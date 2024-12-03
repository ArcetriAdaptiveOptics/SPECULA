from astropy.io import fits

from specula.lib.make_xy import make_xy
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.interp2d import Interp2D
from specula.data_objects.ef import ElectricField
from specula.connections import InputList
from specula.data_objects.layer import Layer
from specula import show_in_profiler

import numpy as np

sec2rad = 4.848e-6
degree2rad = np.pi / 180.


class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self,
                 source_dict,
                 pixel_pupil: int,
                 pixel_pitch: float,
                 target_device_idx=None, 
                 precision=None,
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 pupil_position=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')

        self.pixel_pupil_size = pixel_pupil
        self.pixel_pitch = pixel_pitch
        self.source_dict = source_dict
        if pupil_position is not None:
            self.pupil_position = np.array(pupil_position, dtype=self.dtype)
            if self.pupil_position.size != 2:
                raise ValueError('Pupil position must be an array with 2 elements')
        else:
            self.pupil_position = None
            
        self.doFresnel = doFresnel
        self.wavelengthInNm = wavelengthInNm
        self.propagators = None

        for name, source in source_dict.items():
            ef = ElectricField(self.pixel_pupil_size, self.pixel_pupil_size, self.pixel_pitch, target_device_idx=self.target_device_idx)
            ef.S0 = source.phot_density()
            self.outputs['out_'+name+'_ef'] = ef            
            
        self.inputs['layer_list'] = InputList(type=Layer)

    def doFresnel_setup(self):
   
        if not self.propagators:
                        
            layer_list = self.local_inputs['layer_list']
            
            nlayers = len(layer_list)
            self.propagators = []

            height_layers = np.array([layer.height for layer in self.layer_list], dtype=self.dtype)
            sorted_heights = np.sort(height_layers)
            if not (np.allclose(height_layers, sorted_heights) or np.allclose(height_layers, sorted_heights[::-1])):
                raise ValueError('Layers must be sorted from highest to lowest or from lowest to highest')

            for j in range(nlayers):
                if j < nlayers - 1:
                    self.diff_height_layer = layer_list[j].height - layer_list[j + 1].height
                else:
                    self.diff_height_layer = layer_list[j].height
                
                diameter = self.pixel_pupil_size * self.pixel_pitch
                H = field_propagator(self.pixel_pupil_size, diameter, self.wavelengthInNm, self.diff_height_layer, do_shift=True)
                
                self.propagators.append(H)

    @show_in_profiler('atmo_propagation.trigger_code')
    def trigger_code(self):
        #if self.doFresnel:
        #    self.doFresnel_setup()

        layer_list = self.local_inputs['layer_list']
        for source_name, source in self.source_dict.items():

            output_ef = self.outputs['out_'+source_name+'_ef']
            output_ef.reset()

            for layer in layer_list:

                interpolator = self.interpolators[source][layer]
                
                if interpolator is None:
                    topleft = [(layer.size[0] - self.pixel_pupil_size) // 2, (layer.size[1] - self.pixel_pupil_size) // 2]
                    output_ef.product(layer, subrect=topleft)
                else:
                    
                    if self.magnification_list[layer] is not None:
                        tempA = layer.A
                        tempP = layer.phaseInNm
                        tempP[tempA == 0] = self.xp.mean(tempP[tempA != 0])
                        layer.phaseInNm = tempP

                    output_ef.A *= interpolator.interpolate(layer.A)
                    output_ef.phaseInNm += interpolator.interpolate(layer.phaseInNm)

#                if self.doFresnel:
#                    if self.propagators:
#                        propagator = self.propagators[i]
#                    else:
#                        propagator = None
#                    self.update_ef.physical_prop(self.wavelengthInNm, propagator, temp_array=None)

        for source_name in self.source_dict.keys():
            self.outputs['out_'+source_name+'_ef'].generation_time = self.current_time

        # import matplotlib.pyplot as plt
        # plt.imshow(self.outputs['out_ngs1_source_ef'].A.get())
        # plt.show()
        # import code
        # code.interact(local=dict(locals(), **globals()))
    
    def setup_interpolators(self):
        
        self.interpolators = {}
        for source in self.source_dict.values():
            self.interpolators[source] = {}
            for layer in self.layer_list:
                diff_height = source.height - layer.height
                if (layer.height == 0 or (np.isinf(source.height) and source.r == 0)) and \
                                not self.shiftXY_cond[layer] and \
                                self.pupil_position is None and \
                                layer.rotInDeg == 0 and \
                                self.magnification_list[layer] == 1:
                    self.interpolators[source][layer] = None

                elif diff_height > 0:
                    self.interpolators[source][layer] = self.layer_interpolator(source, layer)
                else:
                    raise ValueError('Invalid layer/source geometry')
 
    def layer_interpolator(self, source, layer):
        pixel_layer = layer.size[0]
        half_pixel_layer = np.array([(pixel_layer - 1) / 2., (pixel_layer - 1) / 2.]) 
        cos_sin_phi =  np.array( [np.cos(source.phi), np.sin(source.phi)]) 
        half_pixel_layer -= layer.shiftXYinPixel

        if self.pupil_position is not None and pixel_layer > self.pixel_pupil_size and np.isinf(source.height):
            pixel_position_s = source.r * layer.height / layer.pixel_pitch
            pixel_position = pixel_position_s * cos_sin_phi + self.pupil_position / layer.pixel_pitch
        elif self.pupil_position is not None and pixel_layer > self.pixel_pupil_size and not np.isinf(source.height):
            pixel_position_s = source.r * source.height / layer.pixel_pitch
            sky_pixel_position = pixel_position_s * cos_sin_phi
            pupil_pixel_position = self.pupil_position / layer.pixel_pitch
            pixel_position = (sky_pixel_position - pupil_pixel_position) * layer.height / source.height + pupil_pixel_position
        else:
            pixel_position_s = source.r * layer.height / layer.pixel_pitch
            pixel_position = pixel_position_s * cos_sin_phi

        if np.isinf(source.height):
            pixel_pupmeta = self.pixel_pupil_size
        else:
            cone_coeff = abs(source.height - abs(layer.height)) / source.height
            pixel_pupmeta = self.pixel_pupil_size * cone_coeff

        if self.magnification_list[layer] != 1.0:
            pixel_pupmeta /= self.magnification_list[layer]

        angle = -layer.rotInDeg % 360
        xx, yy = make_xy(self.pixel_pupil_size, pixel_pupmeta/2., xp=self.xp)
        xx1 = xx + half_pixel_layer[0] + pixel_position[0]
        yy1 = yy + half_pixel_layer[1] + pixel_position[1]
        return Interp2D(layer.size, (self.pixel_pupil_size, self.pixel_pupil_size), xx=xx1, yy=yy1,
                        rotInDeg=angle*180.0/3.1415, xp=self.xp, dtype=self.dtype)

    def run_check(self, time_step):
        # TODO here for no better place, we need something like a "setup()" method called before the loop starts        
        self.layer_list = self.inputs['layer_list'].get(self.target_device_idx)        
        self.shiftXY_cond = {layer: np.any(layer.shiftXYinPixel) for layer in self.layer_list}
        self.magnification_list = {layer: max(layer.magnification, 1.0) for layer in self.layer_list}

        self.setup_interpolators()

        errmsg = ''
        if not (len(self.source_dict) > 0):
            errmsg += 'no source'
        if not (len(self.layer_list) > 0):
            errmsg += 'no layers'
        if not (self.pixel_pupil_size > 0):
            errmsg += 'pixel pupil <= 0'
        return (len(self.source_dict) > 0 and
                len(self.layer_list) > 0 and
                self.pixel_pupil_size > 0), errmsg

    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        super().save(filename, hdr)

        with fits.open(filename, mode='append') as hdul:
            hdul.append(fits.ImageHDU(data=self.phasescreens))

    def read(self, filename):
        super().read(filename)
        self.phasescreens = fits.getdata(filename, ext=1)


