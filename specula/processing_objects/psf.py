
from specula import fuse, show_in_profiler
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.ef import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue

import numpy as np


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))


class PSF(BaseProcessingObj):
    def __init__(self,
                 wavelengthInNm: float,
                 nd: int=1,
                 start_time: float=0.0,
                 target_device_idx: int = None, 
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)        
        self.nd = nd
        self.start_time = start_time
        self.wavelengthInNm = wavelengthInNm

        self.sr = BaseValue()
        self.int_sr = BaseValue()
        self.psf = BaseValue()
        self.int_psf = BaseValue()
        self.in_ef = None
        self.ref = None
        self.intsr = 0.0
        self.count = 0

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self.sr
        self.outputs['out_psf'] = self.psf
#        self.reset_integration()

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

    @property
    def size(self):
        in_ef = self.inputs['in_ef'].get(self.target_device_idx)
        return in_ef.size if in_ef else None

    def run_check(self, time_step, errmsg=''):
        in_ef = self.inputs['in_ef'].get(self.target_device_idx)
        if not in_ef:
            errmsg += ' Input intensity object has not been set'
        if self.wavelengthInNm == 0:
            errmsg += ' PSF wavelength is zero'
        return bool(in_ef) and (self.wavelengthInNm > 0)

    def reset_integration(self):
        self.count = 0
        in_ef = self.local_inputs['in_ef']
        if in_ef:
            self.int_psf.value *= 0
        self.intsr = 0

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.in_ef = self.local_inputs['in_ef']
        if self.psf.value is None:
            s = [dim * self.nd for dim in self.in_ef.size]
            self.int_psf.value = self.xp.zeros(s, dtype=self.dtype)
            self.intsr = 0
        if self.current_time_seconds >= self.start_time:
            self.count += 1
        self.out_size = [np.around(dim * self.nd) for dim in self.in_ef.size]
        if not self.ref:
            self.ref = Intensity(self.out_size[0], self.out_size[1])
            self.ref.i = self.calc_psf(self.in_ef.A * 0.0, self.in_ef.A, imwidth=self.out_size[0], normalize=True)

    @show_in_profiler('psf.trigger')
    def trigger_code(self):
        self.psf.value = self.calc_psf(self.in_ef.phi_at_lambda(self.wavelengthInNm), self.in_ef.A, imwidth=self.out_size[0], normalize=True)
        self.sr.value = self.psf.value[self.out_size[0] // 2, self.out_size[1] // 2] / self.ref.i[self.out_size[0] // 2, self.out_size[1] // 2]
        print('SR:', self.sr.value)

    def post_trigger(self):
        super().post_trigger()
        if self.current_time_seconds >= self.start_time:
            self.intsr += self.sr.value
            self.int_psf.value += self.psf.value
            self.int_sr.value = self.intsr / self.count
            self.int_psf.generation_time = self.current_time
            self.int_sr.generation_time = self.current_time
        self.psf.generation_time = self.current_time
        self.sr.generation_time = self.current_time

