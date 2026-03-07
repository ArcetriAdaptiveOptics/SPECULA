import numpy as np
from specula import fuse, RAD2ASEC
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.zernike_generator import ZernikeGenerator
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.mask import CircularMask
from specula.lib.make_mask import make_mask

@fuse(kernel_name='abs2_cwfs')
def abs2_cwfs(u_fp, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp))

class CurvatureSensor(BaseProcessingObj):
    """
    Curvature Wavefront Sensor (CWFS) propagator processing object.
    This class applies a Zernike Focus aberration (defocus) to the input electric field
    and propagates it to generate intra-focal and extra-focal intensity images.
    """
    def __init__(self,
                 wavelengthInNm: float,
                 wanted_fov: float,
                 pxscale: float,
                 output_resolution: int,
                 defocus_rms_nm: float,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.wavelength_in_nm = wavelengthInNm
        self.wanted_fov = wanted_fov
        self.pxscale = pxscale
        self.output_resolution = output_resolution
        self.defocus_rms_nm = defocus_rms_nm

        self.inputs['in_ef'] = InputValue(type=ElectricField)

        # Output: Two intensities (Intra- and Extra-focal)
        self._out_i1 = Intensity(self.output_resolution, self.output_resolution, 
                                 precision=self.precision, target_device_idx=self.target_device_idx)
        self._out_i2 = Intensity(self.output_resolution, self.output_resolution, 
                                 precision=self.precision, target_device_idx=self.target_device_idx)
        self.outputs['out_i1'] = self._out_i1
        self.outputs['out_i2'] = self._out_i2

        self.fp_plus = None
        self.fp_minus = None
        self.exp_plus = None
        self.exp_minus = None
        self.ef_interpolator = None
        self.ef_resampled = None
        self.detector_mask = None

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']

        # 1. Calculate magnification to achieve the requested pixel scale after FFT
        # Spatial resolution in the FFT plane is dx = lambda / D_padded
        pxscale_rad = self.pxscale / RAD2ASEC
        required_dx = (self.wavelength_in_nm * 1e-9) / (self.output_resolution * pxscale_rad)
        self.magnification = in_ef.pixel_pitch / required_dx

        # Setup EFInterpolator
        self.ef_interpolator = EFInterpolator(
            in_ef=in_ef,
            out_shape=(self.output_resolution, self.output_resolution),
            magnification=self.magnification,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        self.ef_resampled = self.xp.zeros((self.output_resolution, self.output_resolution),
                                          dtype=self.complex_dtype)
        self.fp_plus = self.xp.zeros((self.output_resolution, self.output_resolution),
                                     dtype=self.complex_dtype)
        self.fp_minus = self.xp.zeros((self.output_resolution, self.output_resolution),
                                      dtype=self.complex_dtype)

        # 2. Define the exact area occupied by the pupil in the padded array
        pupil_diameter_pix = in_ef.size[0] * self.magnification
        center = np.ones(2, dtype=self.dtype) * (self.output_resolution / 2.0)

        # Generate a mask so that the ZernikeGenerator creates the parabola exactly
        # confined to the real diameter of the scaled pupil
        mask = CircularMask((self.output_resolution, self.output_resolution), 
                            maskCenter=center, maskRadius=pupil_diameter_pix / 2.0)

        zgen = ZernikeGenerator(mask, self.xp, self.dtype)
        z4 = zgen.getZernike(4) # Z4 Noll = Focus

        # Convert RMS Nanometers to Phase Radians
        k = 2.0 * np.pi / self.wavelength_in_nm
        phase_aberration = z4 * self.defocus_rms_nm * k

        self.exp_plus = self.xp.exp(1j * phase_aberration, dtype=self.complex_dtype)
        self.exp_minus = self.xp.exp(-1j * phase_aberration, dtype=self.complex_dtype)

        # 3. Create a Field Stop mask based on wanted_fov
        fov_pixels = self.wanted_fov / self.pxscale
        self.detector_mask = make_mask(self.output_resolution,
                                       diaratio=fov_pixels / self.output_resolution,
                                       xp=self.xp)

    def trigger_code(self):
        # 1. Retrieve and interpolate input electric field to the new padded grid
        self.ef_interpolator.interpolate()
        self.ef_interpolator.interpolated_ef().ef_at_lambda(self.wavelength_in_nm,
                                                            out=self.ef_resampled)

        # 2. Intrafocal propagation
        ef_plus = self.ef_resampled * self.exp_plus
        self.fp_plus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_plus))
        abs2_cwfs(self.fp_plus, self._out_i1.i, xp=self.xp)
        self._out_i1.i *= self.detector_mask

        # 3. Extrafocal propagation
        ef_minus = self.ef_resampled * self.exp_minus
        self.fp_minus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_minus))
        abs2_cwfs(self.fp_minus, self._out_i2.i, xp=self.xp)
        self._out_i2.i *= self.detector_mask

    def post_trigger(self):
        super().post_trigger()
        # Photometric normalization
        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()

        # Avoid division by zero if image is empty
        sum1 = self._out_i1.i.sum()
        sum2 = self._out_i2.i.sum()

        if sum1 > 0:
            self._out_i1.i *= phot / sum1
        if sum2 > 0:
            self._out_i2.i *= phot / sum2

        self._out_i1.generation_time = self.current_time
        self._out_i2.generation_time = self.current_time
