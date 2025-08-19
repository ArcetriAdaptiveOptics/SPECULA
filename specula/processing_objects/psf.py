
from specula import fuse
from specula.lib.calc_psf_geometry import calc_psf_geometry

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

        if wavelengthInNm <= 0:
            raise ValueError('PSF wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm

        self.psf_pixel_size, self.nd = calc_psf_geometry(
                                            simul_params.pixel_pupil,
                                            simul_params.pixel_pitch,
                                            wavelengthInNm,
                                            nd,
                                            pixel_size_mas)
            
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