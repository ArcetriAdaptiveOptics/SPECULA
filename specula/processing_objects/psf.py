
from specula import fuse
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

import numpy as np


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))


class PSF(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,    # TODO =500.0,
                 nd: float=None,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.dim_pup_in_m = self.pixel_pupil * self.pixel_pitch

        if wavelengthInNm <= 0:
            raise ValueError('PSF wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm

        if nd is not None:
            if pixel_size_mas is not None:
                raise ValueError('Cannot set both nd and pixel_size_mas. Use one or the other.')
            self.nd = nd
        elif pixel_size_mas is not None:
            self.nd = PSF.calc_psf_sampling(
                self.pixel_pupil,
                self.pixel_pitch,
                self.wavelengthInNm,
                pixel_size_mas
            )
        else:
            # Default case, use nd as a scaling factor
            self.nd = 1.0
        self.psf_pixel_size = self.calc_psf_pixel_size()

        self.start_time = start_time

        self.sr = BaseValue(target_device_idx=self.target_device_idx)
        self.int_sr = BaseValue(target_device_idx=self.target_device_idx)
        self.psf = BaseValue(target_device_idx=self.target_device_idx)
        self.int_psf = BaseValue(target_device_idx=self.target_device_idx)
        self.ref = None
        self.count = 0
        self.first = True

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self.sr
        self.outputs['out_psf'] = self.psf
        self.outputs['out_int_sr'] = self.int_sr
        self.outputs['out_int_psf'] = self.int_psf

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']
        s = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.int_psf.value = self.xp.zeros(s, dtype=self.dtype)
        self.int_sr.value = 0

        self.out_size = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.ref = Intensity(self.out_size[0], self.out_size[1], target_device_idx=self.target_device_idx)

    def calc_psf_sampling(pixel_pupil: int, pixel_pitch: float, wavelength_nm: float, psf_pixel_size_mas: float):
        """
        Calculate PSF sampling parameters ensuring constraints are met

        Args:
            pixel_pupil: Number of pixels across the pupil
            pixel_pitch: Physical size of each pixel in meters
            wavelength_nm: Wavelength in nanometers
            psf_pixel_size_mas: Desired PSF pixel size in milliarcseconds

        Returns:
            psf_sampling: The calculated sampling factor
        """

        # Calculate pupil diameter in meters
        dim_pup_in_m = pixel_pupil * pixel_pitch

        # Calculate theoretical maximum pixel size (Nyquist limit)
        max_pixel_size_mas = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000

        if psf_pixel_size_mas > max_pixel_size_mas:
            raise ValueError(
                f"Requested PSF pixel size ({psf_pixel_size_mas:.2f} mas) is larger than "
                f"the theoretical maximum ({max_pixel_size_mas:.2f} mas) for this wavelength and pupil size."
            )

        # Calculate required sampling
        required_sampling = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000 / psf_pixel_size_mas

        # Find nearest valid sampling (pixel_pupil * sampling must be integer)
        # Try different integer values for the final PSF size
        best_sampling = required_sampling
        best_error = float('inf')

        for psf_size in range(int(pixel_pupil * required_sampling) - 5, 
                            int(pixel_pupil * required_sampling) + 6):
            if psf_size > 0:
                candidate_sampling = psf_size / pixel_pupil
                candidate_pixel_size = max_pixel_size_mas / candidate_sampling
                error = abs(candidate_pixel_size - psf_pixel_size_mas)

                if error < best_error:
                    best_error = error
                    best_sampling = candidate_sampling

        actual_psf_sampling = best_sampling
        actual_pixel_size_mas = max_pixel_size_mas / actual_psf_sampling

        # Warning if approximation is significant
        error_percent = abs(actual_pixel_size_mas - psf_pixel_size_mas) / psf_pixel_size_mas * 100
        if error_percent > 1.0:
            print(f"Warning: Actual pixel size ({actual_pixel_size_mas:.2f} mas) differs from "
                f"requested ({psf_pixel_size_mas:.2f} mas) by {error_percent:.1f}% due to "
                f"integer sampling constraint.")

        return actual_psf_sampling

    def calc_psf_pixel_size(self):
        """
        Calculate PSF pixel size based on sampling factor or default settings.
        
        Returns:
            pixel_size_mas 
        """

        pixel_size_mas = (self.wavelengthInNm * 1e-9 / self.dim_pup_in_m * 3600 * 180 / np.pi) * 1000 / self.nd

        return pixel_size_mas

    def calc_psf(self, phase, amp, imwidth=None, normalize=False, nocenter=False):
        """
        Calculates a PSF from an electrical field phase and amplitude.

        Parameters:
        phase : ndarray
            2D phase array.
        amp : ndarray
            2D amplitude array (same dimensions as phase).
        imwidth : int, optional
            Width of the output image. If provided, the output will be of shape (imwidth, imwidth).
        normalize : bool, optional
            If set, the PSF is normalized to total(psf).
        nocenter : bool, optional
            If set, avoids centering the PSF and leaves the maximum pixel at [0,0].

        Returns:
        psf : ndarray
            2D PSF (same dimensions as phase).
        """

        # Set up the complex array based on input dimensions and data type
        if imwidth is not None:
            u_ef = self.xp.zeros((imwidth, imwidth), dtype=self.complex_dtype)
            result = amp * self.xp.exp(1j * phase, dtype=self.complex_dtype)
            s = result.shape
            u_ef[:s[0], :s[1]] = result
        else:
            u_ef = amp * self.xp.exp(1j * phase, dtype=self.complex_dtype)
        # Compute FFT (forward)
        u_fp = self.xp.fft.fft2(u_ef)
        # Center the PSF if required
        if not nocenter:
            u_fp = self.xp.fft.fftshift(u_fp)
        # Compute the PSF as the square modulus of the Fourier transform
        psf = psf_abs2(u_fp, xp=self.xp)
        # Normalize if required
        if normalize:
            self.total_psf = self.xp.sum(psf)
            psf /= self.total_psf

        return psf

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        in_ef = self.local_inputs['in_ef']

        # First time, calculate reference PSF.
        if self.first:
            self.ref.i[:] = self.calc_psf(in_ef.A * 0.0, in_ef.A, imwidth=self.out_size[0], normalize=True)
            self.first = False

    def trigger_code(self):
        in_ef = self.local_inputs['in_ef']
        self.psf.value = self.calc_psf(in_ef.phi_at_lambda(self.wavelengthInNm), in_ef.A, imwidth=self.out_size[0], normalize=True)
        self.sr.value = self.psf.value[self.out_size[0] // 2, self.out_size[1] // 2] / self.ref.i[self.out_size[0] // 2, self.out_size[1] // 2]
        print('SR:', self.sr.value, flush=True)

    def post_trigger(self):
        super().post_trigger()
        if self.current_time_seconds >= self.start_time:
            self.count += 1
            self.int_sr.value += self.sr.value
            self.int_psf.value += self.psf.value
        self.psf.generation_time = self.current_time
        self.sr.generation_time = self.current_time

    def finalize(self):
        if self.count > 0:
            self.int_psf.value /= self.count
            self.int_sr.value /= self.count

        self.int_psf.generation_time = self.current_time
        self.int_sr.generation_time = self.current_time