from specula.lib.make_xy import make_xy
from specula.lib.utils import local_mean_rebin
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.lib.interp2d import Interp2D
from specula.data_objects.electric_field import ElectricField
from specula.connections import InputList
from specula.data_objects.layer import Layer
from specula.lib.air_refraction import MatharAirRefraction
from specula import cpuArray, show_in_profiler
from specula.data_objects.simul_params import SimulParams
import warnings

import numpy as np

degree2rad = np.pi / 180.

class AtmoPropagation(BaseProcessingObj):
    """
    Atmospheric propagation processing object.
    This processing object simulates the propagation of light through atmospheric turbulence
    layers. It can perform both geometric and physical (Fresnel) propagation, depending on the
    configuration.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 source_dict: dict,     # TODO ={},
                 doFresnel: bool=False,
                 wavelengthInNm: float=500.0,
                 telescope_altitude_m: float=None,
                 enable_chromatic_effect: bool=False,
                 chromatic_reference_wavelengthInNm: float=None,
                 pupil_position=None,
                 mergeLayersContrib: bool=True,
                 upwards: bool=False,
                 padding_factor: int=1,
                 band_limit_factor: float=1.0,
                 target_device_idx=None,
                 precision=None):
        """
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
            If True, physical Fresnel propagation is performed. Default is False
            (geometric propagation).
        wavelengthInNm : float, optional
            Wavelength in nanometers for Fresnel propagation. Required if doFresnel is True.
            Default is 500.0 nm.
        telescope_altitude_m : float, optional
            Telescope altitude above sea level in meters used by chromatic
            anisoplanatism calculations (default: None).
        enable_chromatic_effect : bool, optional
            If True, compute and apply chromatic anisoplanatism shifts for atmospheric layers
            (default: False).
            From Devaney et al. "Chromatic Anisoplanatism in Adaptive Optics" SPIE, 2024 
        chromatic_reference_wavelengthInNm : float, optional
            Reference wavelength in nanometers used for chromatic
            anisoplanatism calculations, typically the WFS wavelength.
            Required when ``enable_chromatic_effect`` is True.
        pupil_position : array-like, optional
            Position of the pupil in pixels. Default is None (centered).
        mergeLayersContrib : bool, optional
            If True, contributions from all layers are merged into a single output per source.
            Default is True.
        upwards : bool, optional
            If True, propagation is performed upwards (from ground to source). Default is False
            (downwards).
        padding_factor : int, optional
            Factor for zero padding in Fresnel propagation to avoid numerical issues with FFTs.
        band_limit_factor: float, optional
            Factor in (0,1) for bandlimit filter in angular spectrum propagation.
            If set to 1.0 no bandlimit filter is applied, if set to 0 the full bandlimit filter
            is applied.
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
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
        self.telescope_altitude_m = telescope_altitude_m
        self.enable_chromatic_effect = enable_chromatic_effect
        self.chromatic_reference_wavelengthInNm = chromatic_reference_wavelengthInNm
        self._air_refraction_model = None
        self.propagators = None
        self._block_size = {}
        self.padding = padding_factor
        self.band_limit_factor = band_limit_factor

        if self.enable_chromatic_effect:
            if self.chromatic_reference_wavelengthInNm is None:
                raise ValueError('chromatic_reference_wavelengthInNm is required when'
                                 ' enable_chromatic_effect is True.')
            if self.telescope_altitude_m is None:
                raise ValueError('telescope_altitude_m is required when'
                                 ' enable_chromatic_effect is True.')
            self._air_refraction_model = MatharAirRefraction()

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

    def input_names(self):
        return {'atmo_layer_list': InputDesc(Layer, 'List of atmospheric turbulence layers (optional)'),
                'common_layer_list': InputDesc(Layer, 'List of common turbulence layers shared across sources')}

    def output_names(self):
        result = {}
        for source_name in self.source_dict:
            result['out_' + source_name + '_ef'] = OutputDesc(ElectricField, f'Electric field output for source {source_name}')
        return result

    # Band-limited angular spectrum method for numerical simulation of free-space propagation in far and near fields
    # K. Matsushima, T. Shimobaba
    def field_propagator(self, distanceInM):
        # padded size
        L_pad = self.ef_size_padded * self.pixel_pitch

        df = 1 / L_pad
        fx, fy = self.xp.meshgrid(df * self.xp.arange(-self.ef_size_padded // 2, self.ef_size_padded // 2),
                                  df * self.xp.arange(-self.ef_size_padded // 2, self.ef_size_padded // 2))
        fsq = fx**2 + fy**2

        # maximal spatial frequency that can propagate in x- and y-direction
        f_limit = L_pad / (self.wavelengthInNm * 1e-9 * np.sqrt(L_pad ** 2 + 4 * distanceInM ** 2))

        # reduce propagation distance if max. spatial frequency is too low - otherwise aliasing occurs
        if f_limit < (self.ef_size_padded / 2 * df * self.band_limit_factor):
            warnings.warn(
                'Propagation distance too large for current band_limit_max in angular spectrum propagation. '
                'Consider increasing zero padding or decreasing band_limit_max.',
                RuntimeWarning)
            f_limit = self.ef_size_padded / 2 * df * self.band_limit_factor
            distance_old = distanceInM
            distanceInM = np.sqrt((L_pad / f_limit) ** 2 / (self.wavelengthInNm * 1e-9) ** 2 - L_pad ** 2) / 2
            warnings.warn('Distance for wavelength ' + str(self.wavelengthInNm) + 'nm reduced from ' + str(
                distance_old) + 'm to ' + str(distanceInM) + 'm.', RuntimeWarning)

        # calculate kernel
        H_AS = self.xp.exp(-1j * np.pi * distanceInM * self.wavelengthInNm * 1e-9 * fsq)

        # Apply bandlimit filter
        if self.band_limit_factor < 1.0:
            W = ((fx / f_limit) ** 2 + (fy * self.wavelengthInNm * 1e-9) ** 2 <= 1) * (
                    (fy / f_limit) ** 2 + (fx * self.wavelengthInNm * 1e-9) ** 2 <= 1)
            H_AS *= W

        return H_AS

    def doFresnel_setup(self):
        self.ef_size_padded = self.pixel_pupil * self.padding

        layer_list = self.common_layer_list + self.atmo_layer_list
        height_layers = np.array([layer.height * self.airmass for layer in layer_list], dtype=self.dtype)

        source_height = self.source_dict[list(self.source_dict)[0]].height * self.airmass
        if np.isinf(source_height):
            raise ValueError('Fresnel propagation to infinity not supported.')

        sorted_heights = np.sort(height_layers)
        if not np.allclose(height_layers, sorted_heights):
            raise ValueError('Layers must be sorted from lowest to highest')

        # set up fresnel propagator if height difference is not 0
        height_diffs = np.diff(height_layers, append=source_height)
        self.propagators = [self.field_propagator(diff) if diff != 0 else None for diff in height_diffs]

        # adapt for downwards propagation
        if not self.upwards:
            self.propagators = self.propagators[::-1]
            # no propagation from the source downwards
            self.propagators.pop(0)
            self.propagators.append(None)

        # pre-allocate arrays for propagation
        self.ef_padded = self.xp.zeros([self.ef_size_padded, self.ef_size_padded], dtype=self.complex_dtype)
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
        # zero padding
        s = (self.ef_size_padded - self.pixel_pupil + 1) // 2
        self.ef_padded[s:s + self.pixel_pupil, s:s + self.pixel_pupil] = ef_in

        self.ft_ef1[:] = self.xp.fft.fftshift(self.xp.fft.fft2(self.ef_padded, norm="ortho"))
        self.ef_fresnel_padded[:] = self.xp.fft.ifft2(self.xp.fft.ifftshift(self.ft_ef1 * propagator), norm="ortho")

        # unpadding
        self.output_ef_fresnel[:] = self.ef_fresnel_padded[s:s + self.pixel_pupil, s:s + self.pixel_pupil]


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

    @staticmethod
    def _pressure_nasa(h_asl):
        if h_asl < 11000.0:
            T_h = 288.08 - 0.00649 * h_asl
            return 1012.9 * (T_h / 288.08)**5.256
        elif h_asl < 25000.0:
            return 226.5 * np.exp(1.73 - 0.000157 * h_asl)
        else:
            T_h = 141.94 + 0.00299 * h_asl
            return 24.88 * (T_h / 216.6)**-11.388

    def compute_chromatic_shifts(self, source, atmo_layer_list):
        """
        Pre-compute the chromatic lateral displacement for each *atmospheric* layer.

        Uses the MatharAirRefraction (Ciddor+Mathar) model to calculate precise 
        refractivity across Visible and Mid-IR bands. Then applies the NASA standard 
        atmospheric pressure profile to compute the exact lateral shift using the 
        Devaney 2024 plane-parallel equations (Eq. 1 and Eq. 6).

        The result is stored in :attr:`chromatic_shifts_m` as a **dict keyed by
        Layer object**, containing the signed lateral displacement in metres.
        Common layers (pupil stop, DM, etc.) are not included and will
        implicitly receive a zero shift in the propagation code.

        This method must be called before the interpolators are built.

        Parameters
        ----------
        atmo_layer_list : list of Layer
            Atmospheric turbulence layers only (not common layers such as
            pupil stops or DMs).
        zenith_angle_deg : float
            Observation zenith angle in degrees.

        Notes
        -----
        If enable_chromatic_effect is False or the two wavelengths are identical,
        all shifts are zero.
        """
        source.chromatic_shifts_m = {}

        if not self.enable_chromatic_effect:
            return
        if self._air_refraction_model is None:
            self._air_refraction_model = MatharAirRefraction()
        if source.wavelengthInNm == self.chromatic_reference_wavelengthInNm:
            return

        # 1. Compute delta refractivity using Standard Conditions (15 C, 101325 Pa, 0% RH)
        n_minus_1_ref = self._air_refraction_model.get_refractive_index(
            self.chromatic_reference_wavelengthInNm * 1e-9)
        n_minus_1_src = self._air_refraction_model.get_refractive_index(source.wavelengthInNm * 1e-9)
        delta_N = n_minus_1_ref - n_minus_1_src

        # 2. Parameters for Devaney 2024 Eq. 1
        zeta_rad = np.radians(self.simul_params.zenithAngleInDeg)
        sec_z = 1.0 / np.cos(zeta_rad)
        tan_z = np.tan(zeta_rad)

        g = 9.8 # m/s^2
        rho_s = 1.225 # kg/m^3

        # Pressure at telescope altitude (P0 in mbar)
        P_0_mbar = self._pressure_nasa(self.telescope_altitude_m)
        # Lateral separation of two rays at the telescope aperture (Devaney Eq 1)
        # Note: Convert mbar to Pascal (1 mbar = 100 Pa)
        delta_b0 = delta_N * sec_z * tan_z * ((P_0_mbar * 100.0) / (g * rho_s))

        for layer in atmo_layer_list:
            # Assuming layer.height is the distance above the telescope
            h_asl = self.telescope_altitude_m + float(layer.height)
            P_h_mbar = self._pressure_nasa(h_asl)

            # Lateral separation at altitude h (Devaney Eq 6)
            source.chromatic_shifts_m[layer] = delta_b0 * (1.0 - (P_h_mbar / P_0_mbar))

    def setup_interpolators(self):

        self.interpolators = {}
        layer_list = self.common_layer_list + self.atmo_layer_list
        for source in self.source_dict.values():
            self.interpolators[source] = {}

            self.compute_chromatic_shifts(source, self.atmo_layer_list)

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

        # Apply pre-computed chromatic lateral displacement.
        # Dispersion always occurs along the elevation axis (typically the Y-axis).
        # We assume the zenith-pointing direction maps to [0, 1] in pixel coordinates.
        chromatic_shift_px = source.chromatic_shifts_m.get(layer, 0.0) / layer.pixel_pitch
        if chromatic_shift_px != 0.0:
            elevation_vector = np.array([0.0, 1.0])
            pixel_position = pixel_position + chromatic_shift_px * elevation_vector

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
