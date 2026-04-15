import numpy as np
from specula import fuse, RAD2ASEC
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.lib.zernike_generator import ZernikeGenerator
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.mask import CircularMask
from specula.lib.make_mask import make_mask
from specula.lib.toccd import toccd

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
                 number_px: int,
                 defocus_rms_nm: float,
                 fov_ovs_coeff: float = 2.0,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Parameters:
        ----------

        wavelengthInNm: float
            Wavelength of the light in nanometers.
        wanted_fov: float
            Desired field of view in arcseconds.
        pxscale: float
            Desired pixel scale in arcseconds per pixel at the output.
        number_px: int
            Desired output resolution (number of pixels on one side of the square output image).
        defocus_rms_nm: float
            RMS of the defocus aberration in nanometers (controls the strength of the curvature).
        fov_ovs_coeff : float, optional
            Coefficient to determine the oversampling of the FoV.
            A value larger than 1 is recommended to avoid FFT wrapping effects.
            Default is 2.0.
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.wavelength_in_nm = wavelengthInNm
        self.wanted_fov = wanted_fov
        self.pxscale = pxscale
        self.number_px = number_px
        self.defocus_rms_nm = defocus_rms_nm
        self.fov_ovs_coeff = max(1.0, fov_ovs_coeff)

        self.inputs['in_ef'] = InputValue(type=ElectricField)

        # Final requested outputs
        self._out_i1 = Intensity(self.number_px, self.number_px,
                                 precision=self.precision, target_device_idx=self.target_device_idx)
        self._out_i2 = Intensity(self.number_px, self.number_px,
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
        self._temp_i1 = None
        self._temp_i2 = None
        self.actual_internal_pxscale = None

        self.internal_res = 0

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field')}

    @classmethod
    def output_names(cls):
        return {'out_i1': OutputDesc(Intensity, 'Intra-focal intensity image'),
                'out_i2': OutputDesc(Intensity, 'Extra-focal intensity image')}

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']

        # 1. Convert pxscale from arcsec to radians
        # dx_focal = lambda / (N_pad * dx_pupil) -> N_pad = lambda / (dx_focal * dx_pupil)
        pxscale_rad = self.pxscale / RAD2ASEC
        lambda_m = self.wavelength_in_nm * 1e-9

        # Theoretical size to achieve the *exact* requested pxscale at the output
        # (before cropping)
        exact_fft_size = lambda_m / (pxscale_rad * in_ef.pixel_pitch)

        # Oversampling is needed to avoid aliasing when cropping to the wanted FoV,
        # especially for small FoVs.
        internal_size_float = exact_fft_size * self.fov_ovs_coeff
        self.internal_res = int(np.ceil(internal_size_float))
        self.internal_res += (self.internal_res % 2) # Force even

        # Recalculate the "actual" internal pxscale based on the integer size,
        # which will be important when we do toccd to rescale to the final output_resolution
        self.actual_internal_pxscale = lambda_m / (self.internal_res * in_ef.pixel_pitch) * RAD2ASEC

        # 2. Setup EFInterpolator
        self.ef_interpolator = EFInterpolator(
            in_ef=in_ef,
            out_shape=(self.internal_res, self.internal_res),
            magnification=1.0,
            use_out_ef_cache=False, 
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        # Allocate internal working arrays
        self.ef_resampled = self.xp.zeros((self.internal_res, self.internal_res),
                                          dtype=self.complex_dtype)
        self.fp_plus = self.xp.zeros((self.internal_res, self.internal_res),
                                     dtype=self.complex_dtype)
        self.fp_minus = self.xp.zeros((self.internal_res, self.internal_res),
                                      dtype=self.complex_dtype)

        # Temporary intensity arrays for the full padded FoV
        self._temp_i1 = self.xp.zeros((self.internal_res, self.internal_res), dtype=self.dtype)
        self._temp_i2 = self.xp.zeros((self.internal_res, self.internal_res), dtype=self.dtype)

        # 3. Create Aberration Phase (Focus)
        new_pixel_pitch = in_ef.pixel_pitch * (in_ef.size[0] / self.internal_res)
        pupil_diameter_m = in_ef.size[0] * in_ef.pixel_pitch
        pupil_diameter_pix = pupil_diameter_m / new_pixel_pitch

        center = np.ones(2, dtype=self.dtype) * (self.internal_res / 2.0)
        mask = CircularMask((self.internal_res, self.internal_res),
                            maskCenter=center, maskRadius=pupil_diameter_pix / 2.0)

        zgen = ZernikeGenerator(mask, self.xp, self.dtype)
        z4 = zgen.getZernike(4)

        k = 2.0 * np.pi / self.wavelength_in_nm
        phase_aberration = z4 * self.defocus_rms_nm * k

        self.exp_plus = self.xp.exp(1j * phase_aberration, dtype=self.complex_dtype)
        self.exp_minus = self.xp.exp(-1j * phase_aberration, dtype=self.complex_dtype)

        # 4. Create a Field Stop mask based on wanted_fov on the INTERNAL grid
        fov_pixels = self.wanted_fov / self.actual_internal_pxscale
        self.detector_mask = make_mask(self.internal_res,
                                       diaratio=fov_pixels / self.internal_res,
                                       xp=self.xp)


    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.ef_interpolator.interpolate()
        self.ef_interpolator.interpolated_ef().ef_at_lambda(
            self.wavelength_in_nm, out=self.ef_resampled)


    def trigger_code(self):
        # Intrafocal propagation
        ef_plus = self.ef_resampled * self.exp_plus
        self.fp_plus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_plus))
        abs2_cwfs(self.fp_plus, self._temp_i1, xp=self.xp)
        self._temp_i1 *= self.detector_mask

        # Extrafocal propagation
        ef_minus = self.ef_resampled * self.exp_minus
        self.fp_minus[:] = self.xp.fft.fftshift(self.xp.fft.fft2(ef_minus))
        abs2_cwfs(self.fp_minus, self._temp_i2, xp=self.xp)
        self._temp_i2 *= self.detector_mask

        # Crop to Field of View
        # We need to extract the area defined by wanted_fov
        wanted_fov_internal_pixels = int(np.ceil(self.wanted_fov / self.actual_internal_pxscale))
        cut_start = (self.internal_res - wanted_fov_internal_pixels) // 2
        cut_end = cut_start + wanted_fov_internal_pixels

        i1_fov = self._temp_i1[cut_start:cut_end, cut_start:cut_end]
        i2_fov = self._temp_i2[cut_start:cut_end, cut_start:cut_end]

        # Bin/Resample to the exact requested output_resolution using toccd
        # toccd handles the exact re-binning/interpolation preserving flux
        self._out_i1.i[:] = toccd(i1_fov,
                                  (self.number_px, self.number_px),
                                  xp=self.xp)
        self._out_i2.i[:] = toccd(i2_fov,
                                  (self.number_px, self.number_px),
                                  xp=self.xp)


    def post_trigger(self):
        super().post_trigger()

        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()

        sum1 = self._out_i1.i.sum()
        sum2 = self._out_i2.i.sum()

        if sum1 > 0:
            self._out_i1.i *= phot / sum1
        if sum2 > 0:
            self._out_i2.i *= phot / sum2

        self._out_i1.generation_time = self.current_time
        self._out_i2.generation_time = self.current_time
