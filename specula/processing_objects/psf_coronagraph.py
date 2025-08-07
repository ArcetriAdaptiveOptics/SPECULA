
from specula import fuse
from specula.processing_objects.psf import PSF
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams

import numpy as np


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))

class PsfCoronagraph(PSF):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 nd: float=None,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            nd=nd,
            pixel_size_mas=pixel_size_mas,
            start_time=start_time,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Additional outputs for coronagraph
        self.coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx)
        self.int_coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx)

        self.outputs['out_coronagraph_psf'] = self.coronagraph_psf
        self.outputs['out_int_coronagraph_psf'] = self.int_coronagraph_psf

        # Reference complex amplitude for perfect coronagraph
        self.ref_complex_amplitude = None

    def setup(self):
        super().setup()
        # Initialize integrated coronagraph PSF
        self.int_coronagraph_psf.value = self.xp.zeros_like(self.int_psf.value)

    def calc_perfect_coronagraph_amplitude(self, phase, amp, ref_amp):
        """
        Calculate the perfect coronagraph complex amplitude according to:
        A_pc(ρ, t) = A(ρ, t) - √SR(t) * A_dl(ρ, t)
        
        Where:
        - A(ρ, t) is the complex amplitude with phase and amplitude
        - A_dl(ρ, t) is the diffraction-limited reference amplitude
        - SR(t) is the instantaneous Strehl Ratio
        
        Parameters:
        phase : ndarray
            2D phase array
        amp : ndarray  
            2D amplitude array
        ref_amp : ndarray
            2D reference diffraction-limited amplitude
            
        Returns:
        coronagraph_amplitude : ndarray
            Complex amplitude after perfect coronagraph subtraction
        """
        # Calculate current complex amplitude
        current_amplitude = amp * self.xp.exp(1j * phase, dtype=self.complex_dtype)

        # Calculate instantaneous Strehl Ratio
        # SR = |∫ A(ρ,t) * A_dl*(ρ) dρ|² / (∫ |A(ρ,t)|² dρ * ∫ |A_dl(ρ)|² dρ)
        numerator = self.xp.abs(self.xp.sum(current_amplitude * self.xp.conj(ref_amp)))**2
        denominator = self.xp.sum(self.xp.abs(current_amplitude)**2) * self.xp.sum(self.xp.abs(ref_amp)**2)

        if denominator > 0:
            sr_instant = numerator / denominator
        else:
            sr_instant = 0.0
  
        # Perfect coronagraph subtraction
        coronagraph_amplitude = current_amplitude - self.xp.sqrt(sr_instant) * ref_amp

        return coronagraph_amplitude

    def calc_coronagraph_psf(self, phase, amp, ref_amp, imwidth=None, normalize=False, nocenter=False):
        """
        Calculate coronagraph PSF using perfect coronagraph theory.
        
        Parameters:
        phase : ndarray
            2D phase array
        amp : ndarray
            2D amplitude array  
        ref_amp : ndarray
            2D reference diffraction-limited amplitude
        imwidth : int, optional
            Width of output image
        normalize : bool, optional
            If True, normalize PSF
        nocenter : bool, optional
            If True, don't center the PSF
            
        Returns:
        coronagraph_psf : ndarray
            2D coronagraph PSF
        """
        # Get coronagraph complex amplitude
        coronagraph_amplitude = self.calc_perfect_coronagraph_amplitude(phase, amp, ref_amp)

        # Set up the complex array for FFT
        if imwidth is not None:
            u_ef = self.xp.zeros((imwidth, imwidth), dtype=self.complex_dtype)
            s = coronagraph_amplitude.shape
            u_ef[:s[0], :s[1]] = coronagraph_amplitude
        else:
            u_ef = coronagraph_amplitude

        # Compute FFT
        u_fp = self.xp.fft.fft2(u_ef)

        # Center if required
        if not nocenter:
            u_fp = self.xp.fft.fftshift(u_fp)

        # Calculate PSF as intensity
        coronagraph_psf = psf_abs2(u_fp, xp=self.xp)

        # Normalize if required
        if normalize:
            total = self.xp.sum(coronagraph_psf)
            if total > 0:
                coronagraph_psf /= total

        return coronagraph_psf

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        in_ef = self.local_inputs['in_ef']

        # Calculate reference diffraction-limited complex amplitude (first time only)
        if self.ref_complex_amplitude is None:
            # Create perfect amplitude (no phase errors)
            perfect_phase = self.xp.zeros_like(in_ef.A)
            self.ref_complex_amplitude = in_ef.A * self.xp.exp(1j * perfect_phase, dtype=self.complex_dtype)

    def trigger_code(self):
        # Call parent trigger_code for standard PSF calculation
        super().trigger_code()

        in_ef = self.local_inputs['in_ef']

        # Calculate coronagraph PSF
        self.coronagraph_psf.value = self.calc_coronagraph_psf(
            in_ef.phi_at_lambda(self.wavelengthInNm),
            in_ef.A,
            self.ref_complex_amplitude,
            imwidth=self.out_size[0],
            normalize=True
        )

        print(f'SR: {self.sr.value:.6f}, Coronagraph peak suppression: {self.coronagraph_psf.value.max():.2e}', flush=True)

    def post_trigger(self):
        super().post_trigger()

        if self.current_time_seconds >= self.start_time:
            self.int_coronagraph_psf.value += self.coronagraph_psf.value

        self.coronagraph_psf.generation_time = self.current_time

    def finalize(self):
        super().finalize()

        if self.count > 0:
            self.int_coronagraph_psf.value /= self.count

        self.int_coronagraph_psf.generation_time = self.current_time