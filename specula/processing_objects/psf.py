
from specula import fuse, show_in_profiler
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue

import numpy as np

@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))


class PSF(BaseProcessingObj):
    def __init__(self,
                 wavelengthInNm: float=None,    # TODO =500.0,
                 wavelengthInNm_list: list = None,
                 nd: float=1,
                 start_time: float=0.0,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)       

        if wavelengthInNm is not None:
            if wavelengthInNm_list is not None:
                raise ValueError('Only one of wavelengthInNm and wavelengthInNm_list can be specified')
            self.wavelengthInNm_list = [wavelengthInNm]
            self.single_wavelength = True
        else:
            if wavelengthInNm_list is None:
                raise ValueError('One of wavelengthInNm and wavelengthInNm_list must be specified')
            self.wavelengthInNm_list = wavelengthInNm_list
            self.single_wavelength = False
 
        self.nd = nd
        self.start_time = start_time

        self.out_sr = BaseValue()
        self.out_int_sr = BaseValue()
        self.out_psf = BaseValue()
        self.out_int_psf = BaseValue()
        self.intsr = [0.0] * len(self.wavelengthInNm_list)
        self.in_ef = None
        self.count = 0

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self.out_sr
        self.outputs['out_psf'] = self.out_psf
        self.outputs['out_int_sr'] = self.out_int_sr
        self.outputs['out_int_psf'] = self.out_int_psf

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
            psf /= self.xp.sum(psf)

        return psf

    def setup(self, loop_dt, loop_iters):
        super().setup(loop_dt, loop_niters)

        self.in_ef = self.inputs['in_ef'].get(target_device_idx=self.target_device_idx)
        s = [int(np.around(dim * self.nd/2)*2) for dim in self.in_ef.size]
        self.center_coord = s[0] // 2, s[1] // 2
        self.out_size = s[0]
        npsf = len(self.wavelengthInNm_list)

        self.out_psf.value = self.xp.zeros((npsf,) + s, dtype=self.dtype)
        self.out_int_psf.value = self.xp.zeros((npsf,) + s, dtype=self.dtype)
        self.out_sr.value = self.xp.zeros(npsf, dtype=self.dtype)
        self.out_int_sr.value = self.xp.zeros(npsf, dtype=self.dtype)

        self.ref_psf = Intensity(s[0], s[1])
        self.ref_psf.i = self.calc_psf(self.in_ef.A * 0.0, self.in_ef.A, imwidth=s[0], normalize=True)

    def trigger_code(self):
        for i, wavelength in enumerate(self.wavelengthInNm_list):
            self.out_psf.value[i] = self.calc_psf(self.in_ef.phi_at_lambda(wavelength), self.in_ef.A, imwidth=self.out_size, normalize=True)
            self.out_sr.value[i] = self.out_psf.value[i, *self.center_coord] / self.ref_psf.i[*self.center_coord]
            print(f'SR @{wavelength} nm: {self.out_sr[i]}')

    def post_trigger(self):
        super().post_trigger()
        self.out_psf.generation_time = self.current_time
        self.out_sr.generation_time = self.current_time
        if self.current_time_seconds >= self.start_time:
            self.count += 1
            for i in range(len(self.wavelengthInNm_list)):
                self.out_int_psf.value[i] += self.out_psf.value[i]
                self.intsr[i] += self.out_sr.value[i]
                self.out_int_sr.value[i] = self.intsr[i] / self.count
            self.out_int_psf.generation_time = self.current_time
            self.out_int_sr.generation_time = self.current_time

