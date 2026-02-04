from specula.lib.make_xy import make_xy
from specula.lib.utils import local_mean_rebin
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.interp2d import Interp2D
from specula.data_objects.electric_field import ElectricField
from specula.connections import InputList
from specula.data_objects.layer import Layer
from specula import cpuArray, show_in_profiler
from specula.data_objects.simul_params import SimulParams
import warnings

import numpy as np

degree2rad = np.pi / 180.

class AtmoPropagation(BaseProcessingObj):
    """Atmospheric propagation
    This processing object simulates the propagation of light through atmospheric turbulence layers.
    It can perform both geometric and physical (Fresnel) propagation, depending on the configuration.

    Note
    ----
    - By default, all atmospheric phase screens are referenced to a wavelength of 500 nm.
    - Layer heights are always defined at zenith and projected according to the simulation
      zenith angle (coming from simul_params).

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object containing global settings.
    source_dict : dict
        Dictionary of source objects (e.g., stars, LGS) to be propagated.
    doFresnel : bool, optional
        If True, physical Fresnel propagation is performed. Default is False (geometric propagation).
    wavelengthInNm : float, optional
        Wavelength in nanometers for Fresnel propagation. Required if doFresnel is True. Default is 500.0 nm.
    pupil_position : array-like, optional
        Position of the pupil in pixels. Default is None (centered).
    mergeLayersContrib : bool, optional
        If True, contributions from all layers are merged into a single output per source. Default is True.
    upwards : bool, optional
        If True, propagation is performed upwards (from ground to source). Default is False (downwards).
    padding_factor : int, optional
        Factor for zero padding in Fresnel propagation to avoid numerical issues with FFTs.
    band_limit_factor: float, optional
        Factor in (0,1) for bandlimit filter in angular spectrum propagation.
        If set to 1.0 no bandlimit filter is applied, if set to 0 the full bandlimit filter is applied.
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 source_dict: dict,     # TODO ={},
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 pupil_position=None,
                 mergeLayersContrib: bool=True,
                 upwards: bool=False,
                 padding_factor: int=1,
                 band_limit_factor: float=1.0,
                 target_device_idx=None,
                 precision=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params

        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        if not (len(source_dict) > 0):
            raise ValueError('No sources have been set')

        if not (self.pixel_pupil > 0):
            raise ValueError('Pixel pupil must be >0')

        if doFresnel and wavelengthInNm is None:
            raise ValueError('get_atmo_propagation: wavelengthInNm is required when doFresnel key'
                             ' is set to correctly simulate physical propagation.')
        if padding_factor < 1.0:
            raise ValueError('get_atmo_propagation: padding_factor must be greater than 1.')
        if not (0.0 <= band_limit_factor <= 1.0):
            raise ValueError('get_atmo_propagation: band_limit_factor must be between 0.0 and 1.0, but is set to ' + str(band_limit_factor) + '.')

        self.mergeLayersContrib = mergeLayersContrib
        self.upwards = upwards
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
        self.padding = padding_factor
        self.band_limit_factor = band_limit_factor

        if self.mergeLayersContrib:
            for name, source in self.source_dict.items():
                ef = ElectricField(
                    self.pixel_pupil_size,
                    self.pixel_pupil_size,
                    self.pixel_pitch,
                    target_device_idx=self.target_device_idx
                )
                ef.S0 = source.phot_density()
                self.outputs['out_'+name+'_ef'] = ef

        # atmo_layer_list is optional because it can be empty during calibration of
        # an AO system while the common_layer_list is not optional because at least a
        # pupilstop is needed
        self.inputs['atmo_layer_list'] = InputList(type=Layer,optional=True)
        self.inputs['common_layer_list'] = InputList(type=Layer)

        self.airmass = 1. / np.cos(np.radians(self.simul_params.zenithAngleInDeg), dtype=self.dtype)

    # Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields
    # K. Matsushima, T. Shimobaba
    def field_propagator(self, distanceInM):
        # padded size
        L_pad = self.ef_size_padded * self.pixel_pitch

        df = 1 / L_pad
        fx, fy = self.xp.meshgrid(df * self.xp.arange(-self.ef_size_padded / 2, self.ef_size_padded / 2),
                                  df * self.xp.arange(-self.ef_size_padded / 2, self.ef_size_padded / 2))

        # Bandlimit filter for propagation
        f_limit = L_pad / (self.wavelengthInNm * 1e-9 * np.sqrt(L_pad ** 2 + 4 * distanceInM ** 2))
        W = (fx ** 2 / f_limit ** 2 + (self.wavelengthInNm * 1e-9 * fy) ** 2 <= 1) * (
                fy ** 2 / f_limit ** 2 + (self.wavelengthInNm * 1e-9 * fx) ** 2 <= 1)

        # Reduce propagation distance if bandlimit is too tight to have at least band_limit_factor*self.ef_size_padded values
        if self.xp.sum(W) < (self.ef_size_padded * self.band_limit_factor) ** 2:
            warnings.warn(
                'Propagation distance too large for current band_limit_max in angular spectrum propagation. '
                'Consider increasing zero padding or band_limit_max.',
                RuntimeWarning)
            f_limit = self.ef_size_padded / 2 * df * self.band_limit_factor
            distance_old = distanceInM
            distanceInM = np.sqrt((L_pad / f_limit) ** 2 / (self.wavelengthInNm * 1e-9) ** 2 - L_pad ** 2) / 2
            warnings.warn('Distance for wavelength ' + str(self.wavelengthInNm) + 'nm reduced from ' + str(
                distance_old) + 'm to ' + str(distanceInM) + 'm.', RuntimeWarning)
            W = ((fx / f_limit) ** 2 + (fy * self.wavelengthInNm * 1e-9) ** 2 <= 1) * (
                    (fy / f_limit) ** 2 + (fx * self.wavelengthInNm * 1e-9) ** 2 <= 1)

        # calculate kernel
        k = 2 * np.pi / (self.wavelengthInNm * 1e-9)
        kernel = self.xp.sqrt(
            0j + 1 - abs(fx * self.wavelengthInNm * 1e-9) ** 2 - abs(fy * self.wavelengthInNm * 1e-9) ** 2)
        H_AS = self.xp.exp(1j * k * distanceInM * kernel)

        # Apply bandlimit filter
        if self.band_limit_factor < 1.0:
            H_AS *= W

        return H_AS

    def doFresnel_setup(self):
        self.ef_size_padded = self.pixel_pupil * self.padding

        layer_list = self.common_layer_list + self.atmo_layer_list
        height_layers = np.array([layer.height * self.airmass for layer in layer_list], dtype=self.dtype)

        source_height = self.source_dict[list(self.source_dict)[0]].height * self.airmass
        if np.isinf(source_height):
            raise ValueError('Fresnel propagation to infinity not supported.')
        height_layers = np.append(height_layers, source_height)

        sorted_heights = np.sort(height_layers)
        if not np.allclose(height_layers, sorted_heights):
            raise ValueError('Layers must be sorted from lowest to highest')

        # set up fresnel propagator if height difference is not 0
        height_diffs = np.diff(height_layers)
        self.propagators = [self.field_propagator(diff) if diff != 0 else None for diff in height_diffs]

        # adapt for downwards propagation
        if not self.upwards:
            self.propagators = self.propagators[::-1]
            # no propagation from the source downwards
            self.propagators.pop(0)
            self.propagators.append(None)

        # pre-allocate arrays for propagation
        self.ft_ef1 = self.xp.zeros([self.ef_size_padded, self.ef_size_padded], dtype=self.complex_dtype)
        self.ef_fresnel_padded = self.xp.zeros([self.ef_size_padded, self.ef_size_padded],
                                               dtype=self.complex_dtype)
        self.output_ef_fresnel = self.xp.zeros([self.pixel_pupil, self.pixel_pupil], dtype=self.complex_dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        layer_list = self.common_layer_list + self.atmo_layer_list

        for layer in layer_list:
            if self.magnification_list[layer] is not None and self.magnification_list[layer] != 1:
                # update layer phase filling the missing values to avoid artifacts during interpolation
                mask_valid = layer.A != 0
                local_mean = local_mean_rebin(
                    layer.phaseInNm,
                    mask_valid,
                    self.xp,
                    block_size=self._block_size[layer]
                )
                layer.phaseInNm[~mask_valid] = local_mean[~mask_valid]

    def physical_propagation(self, ef_in, propagator):
        self.ft_ef1[:] = self.xp.fft.fftshift(
            self.xp.fft.fft2(self.xp.fft.fftshift(ef_in, axes=(-2, -1)), s=[self.ef_size_padded, self.ef_size_padded],
                             axes=(-2, -1), norm="ortho"), axes=(-2, -1))
        self.ef_fresnel_padded[:] = self.xp.fft.fftshift(
            self.xp.fft.ifft2(self.xp.fft.fftshift(self.ft_ef1 * propagator, axes=(-2, -1)), norm="ortho",
                              axes=(-2, -1)), axes=(-2, -1))
        self.output_ef_fresnel[:] = self.ef_fresnel_padded[(self.ef_size_padded - self.pixel_pupil) // 2:
                                                           (self.ef_size_padded + self.pixel_pupil) // 2,
                                                           (self.ef_size_padded - self.pixel_pupil) // 2:
                                                           (self.ef_size_padded + self.pixel_pupil) // 2]


    @show_in_profiler('atmo_propagation.trigger_code')
    def trigger_code(self):
        layer_list = self.common_layer_list + self.atmo_layer_list
        if not self.upwards:  # reverse layers for downwards propagation
            layer_list = layer_list[::-1]

        for source_name, source in self.source_dict.items():

            if self.mergeLayersContrib:
                output_ef = self.outputs['out_'+source_name+'_ef']
                output_ef.reset()
            else:
                output_ef_list = self.outputs['out_'+source_name+'_ef']

            for li, layer in enumerate(layer_list):

                if not self.mergeLayersContrib:
                    output_ef = output_ef_list[li]
                    output_ef.reset()

                interpolator = self.interpolators[source][layer]
                if interpolator is None:
                    topleft = [(layer.size[0] - self.pixel_pupil_size) // 2, \
                               (layer.size[1] - self.pixel_pupil_size) // 2]
                    output_ef.product(layer, subrect=topleft)
                else:
                    output_ef.A *= interpolator.interpolate(layer.A)
                    output_ef.phaseInNm += interpolator.interpolate(layer.phaseInNm)

                if self.doFresnel and self.propagators[li] is not None:
                    self.physical_propagation(output_ef.ef_at_lambda(self.wavelengthInNm), self.propagators[li])
                    output_ef.phaseInNm[:] = self.xp.angle(self.output_ef_fresnel) * self.wavelengthInNm / (2 * self.xp.pi)
                    output_ef.A[:] = abs(self.output_ef_fresnel)

    def post_trigger(self):
        super().post_trigger()

        for source_name in self.source_dict.keys():
            self.outputs['out_'+source_name+'_ef'].generation_time = self.current_time

    def setup_interpolators(self):

        self.interpolators = {}
        for source in self.source_dict.values():
            self.interpolators[source] = {}

            layer_list = self.common_layer_list + self.atmo_layer_list

            for layer in layer_list:
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
                        raise ValueError(f'FATAL ERROR, the source [{source.polar_coordinates[0]},'
                                         f'{source.polar_coordinates[1]}] is not inside'
                                         f' the selected FoV for atmosphere layers generation.'
                                         f' Layer height: {layer.height} m, size: {layer.size}.')
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
        if self.doFresnel:
            self.doFresnel_setup()
        self.build_stream()
