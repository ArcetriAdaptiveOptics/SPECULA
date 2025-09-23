
from specula.lib.make_xy import make_xy
from specula.lib.utils import local_mean_rebin
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.interp2d import Interp2D
from specula.data_objects.electric_field import ElectricField
from specula.connections import InputList
from specula.data_objects.layer import Layer
from specula import cpuArray, show_in_profiler
from specula.data_objects.simul_params import SimulParams

import numpy as np

degree2rad = np.pi / 180.

class AtmoPropagation(BaseProcessingObj):
    '''Atmospheric propagation'''
    def __init__(self,
                 simul_params: SimulParams,
                 source_dict: dict,     # TODO ={},
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 pupil_position=None,
                 mergeLayersContrib: bool=True,
                 reverse_atmo_layer_list: bool=False,
                 target_device_idx=None,
                 precision=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key is set to correctly simulate physical propagation.')

        if not (len(source_dict) > 0):
            raise ValueError('No sources have been set')

        if not (self.pixel_pupil > 0):
            raise ValueError('Pixel pupil must be >0')

        self.mergeLayersContrib = mergeLayersContrib
        self.reverse_atmo_layer_list = reverse_atmo_layer_list
        self.pixel_pupil_size = self.pixel_pupil        
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
        self._block_size = {}

        if self.mergeLayersContrib:
            for name, source in self.source_dict.items():
                ef = ElectricField(self.pixel_pupil_size, self.pixel_pupil_size, self.pixel_pitch, target_device_idx=self.target_device_idx)
                ef.S0 = source.phot_density()
                self.outputs['out_'+name+'_ef'] = ef

        # atmo_layer_list is optional because it can be empty during calibration of an AO system while
        # the common_layer_list is not optional because at least a pupilstop is needed
        self.inputs['atmo_layer_list'] = InputList(type=Layer,optional=True)
        self.inputs['common_layer_list'] = InputList(type=Layer)

        self.airmass = 1. / np.cos(np.radians(self.simul_params.zenithAngleInDeg), dtype=self.dtype)


    def doFresnel_setup(self):

        raise NotImplementedError('Fresnel propagation is not implemented')

        # Missing lib function
        def field_propagator(*args, **kwargs):
            pass

        if not self.propagators:
                        
            layer_list = self.local_inputs['atmo_layer_list'] + self.local_inputs['common_layer_list']
            
            nlayers = len(layer_list)
            self.propagators = []

            height_layers = np.array([layer.height * self.airmass for layer in self.atmo_layer_list + self.common_layer_list], dtype=self.dtype)
            sorted_heights = np.sort(height_layers)
            if not (np.allclose(height_layers, sorted_heights) or np.allclose(height_layers, sorted_heights[::-1])):
                raise ValueError('Layers must be sorted from highest to lowest or from lowest to highest')

            for j in range(nlayers):
                if j < nlayers - 1:
                    self.diff_height_layer = (layer_list[j].height - layer_list[j + 1].height) * self.airmass
                else:
                    self.diff_height_layer = layer_list[j].height * self.airmass
                
                diameter = self.pixel_pupil_size * self.pixel_pitch
                H = field_propagator(self.pixel_pupil_size, diameter, self.wavelengthInNm, self.diff_height_layer, do_shift=True)
                
                self.propagators.append(H)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        for layer in (self.atmo_layer_list + self.common_layer_list):
            if self.magnification_list[layer] is not None and self.magnification_list[layer] != 1:
                # update layer phase filling the missing values to avoid artifacts during interpolation
                mask_valid = layer.A != 0
                local_mean = local_mean_rebin(layer.phaseInNm, mask_valid, self.xp, block_size=self._block_size[layer])
                layer.phaseInNm[~mask_valid] = local_mean[~mask_valid]

    @show_in_profiler('atmo_propagation.trigger_code')
    def trigger_code(self):
        #if self.doFresnel:
        #    self.doFresnel_setup()
        for source_name, source in self.source_dict.items():

            if self.mergeLayersContrib:
                output_ef = self.outputs['out_'+source_name+'_ef']
                output_ef.reset()
            else:
                output_ef_list = self.outputs['out_'+source_name+'_ef']

            for li, layer in enumerate(self.atmo_layer_list + self.common_layer_list):

                if not self.mergeLayersContrib:
                    output_ef = output_ef_list[li]
                    output_ef.reset()

                interpolator = self.interpolators[source][layer]
                if interpolator is None:
                    topleft = [(layer.size[0] - self.pixel_pupil_size) // 2, (layer.size[1] - self.pixel_pupil_size) // 2]
                    output_ef.product(layer, subrect=topleft)
                else:
                    output_ef.A *= interpolator.interpolate(layer.A)
                    output_ef.phaseInNm += interpolator.interpolate(layer.phaseInNm)

#                if self.doFresnel:
#                    if self.propagators:
#                        propagator = self.propagators[i]
#                    else:
#                        propagator = None
#                    self.update_ef.physical_prop(self.wavelengthInNm, propagator, temp_array=None)

    def post_trigger(self):
        super().post_trigger()

        for source_name in self.source_dict.keys():
            self.outputs['out_'+source_name+'_ef'].generation_time = self.current_time

    def setup_interpolators(self):

        self.interpolators = {}
        for source in self.source_dict.values():
            self.interpolators[source] = {}
            for layer in self.atmo_layer_list + self.common_layer_list:
                diff_height = (source.height - layer.height) * self.airmass
                if (layer.height == 0 or (np.isinf(source.height) and source.r == 0)) and \
                                not self.shiftXY_cond[layer] and \
                                self.pupil_position is None and \
                                layer.rotInDeg == 0 and \
                                self.magnification_list[layer] == 1:
                    self.interpolators[source][layer] = None

                elif diff_height > 0:
                    li = self.layer_interpolator(source, layer)
                    if li is None:
                        raise ValueError('FATAL ERROR, the source is not inside the selected FoV for atmosphere layers generation.')
                    else:
                        self.interpolators[source][layer] = li
                else:
                    raise ValueError('Invalid layer/source geometry')

    def layer_interpolator(self, source, layer):
        pixel_layer = layer.size[0]
        half_pixel_layer = np.array([(pixel_layer - 1) / 2., (pixel_layer - 1) / 2.])
        cos_sin_phi =  np.array( [np.cos(source.phi), np.sin(source.phi)])
        half_pixel_layer -= cpuArray(layer.shiftXYinPixel)

        if self.pupil_position is not None and pixel_layer > self.pixel_pupil_size and np.isinf(source.height):
            pixel_position_s = source.r * layer.height * self.airmass / layer.pixel_pitch
            pixel_position = pixel_position_s * cos_sin_phi + self.pupil_position / layer.pixel_pitch
        elif self.pupil_position is not None and pixel_layer > self.pixel_pupil_size and not np.isinf(source.height):
            pixel_position_s = source.r * source.height * self.airmass / layer.pixel_pitch
            sky_pixel_position = pixel_position_s * cos_sin_phi
            pupil_pixel_position = self.pupil_position / layer.pixel_pitch
            pixel_position = (sky_pixel_position - pupil_pixel_position) * layer.height / source.height + pupil_pixel_position
        else:
            pixel_position_s = source.r * layer.height * self.airmass / layer.pixel_pitch
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

        # TODO old code?
        limit0 = (layer.size[0] - self.pixel_pupil_size) /2
        limit1 = (layer.size[1] - self.pixel_pupil_size) /2
        isInside = abs(pixel_position[0]) <= limit0 and abs(pixel_position[1]) <= limit1
        if not isInside:
            return None

        return Interp2D(layer.size, (self.pixel_pupil_size, self.pixel_pupil_size), xx=xx1, yy=yy1,
                        rotInDeg=angle, xp=self.xp, dtype=self.dtype)

    def setup(self):
        super().setup()

        self.atmo_layer_list = self.local_inputs['atmo_layer_list']
        if self.reverse_atmo_layer_list:
            self.atmo_layer_list.reverse()
        self.common_layer_list = self.local_inputs['common_layer_list']

        if self.atmo_layer_list is None:
            self.atmo_layer_list = []

        if self.common_layer_list is None:
            self.common_layer_list = []

        self.nAtmoLayers = len(self.atmo_layer_list)

        if len(self.atmo_layer_list) + len(self.common_layer_list) < 1:
            raise ValueError('At least one layer must be set')

        if not self.mergeLayersContrib:
            for name, source in self.source_dict.items():
                self.outputs['out_'+name+'_ef'] = []
                for _ in range(self.nAtmoLayers):
                    ef = ElectricField(self.pixel_pupil_size, self.pixel_pupil_size, self.pixel_pitch, target_device_idx=self.target_device_idx)
                    ef.S0 = source.phot_density()
                    self.outputs['out_'+name+'_ef'].append(ef)

        self.shiftXY_cond = {layer: np.any(layer.shiftXYinPixel) for layer in self.atmo_layer_list + self.common_layer_list}
        self.magnification_list = {layer: max(layer.magnification, 1.0) for layer in self.atmo_layer_list + self.common_layer_list}

        self._block_size = {}
        for layer in self.atmo_layer_list + self.common_layer_list:
            for div in [5, 4, 3, 2]:
                if layer.size[0] % div == 0:
                    self._block_size[layer] = div
                    break

        self.setup_interpolators()
        self.build_stream()
