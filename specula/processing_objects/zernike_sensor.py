from specula.processing_objects.modulated_pyramid import ModulatedPyramid
# from specula.lib.utils import make_subpixel_shift_phase
# from specula import cpuArray
from specula.lib.toccd import toccd

class ZernikeSensor(ModulatedPyramid):
    """
    Zernike Sensor processing object.
    Based on phase-shifting focal-plane spot technique, the class 
    inherits from ModulatedPyramid but replaces the pyramid structure with
    a π/2 (default value) phase-shifting spot in the focal plane.
    """

    def __init__(self,
                 simul_params,
                 wavelengthInNm,
                 fov,
                 pup_diam,
                 output_resolution,
                 spot_radius_lambda: float = 1.0,  # Spot radius in λ/D units, adjusted for ~50% light outside mask
                 phase_shift: float = 0.5, #3.141592653589793 / 2,  # π/2 phase shift
                 fft_res: float = 4.0,
                 target_device_idx=None,
                 precision=None):

        self.spot_radius_lambda = spot_radius_lambda
        self.phase_shift = phase_shift

        # Force modulation to zero (no modulation for Zernike sensor)
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_step=1,
            fft_res=fft_res,
            pup_dist=0,
            pup_margin=0,
            min_pup_dist=0,
            fov_errinf=0.1,
            fov_errsup=2.0,
            target_device_idx=target_device_idx,
            precision=precision
        )
        myexp = self.xp.exp(-1j * self.xp.pi * self.pyr_tlt, dtype=self.complex_dtype)
        self.shifted_masked_exp = self.xp.fft.fftshift(myexp * self.fp_mask)

    def get_pyr_tlt(self, p, c):
        """
        Creates a phase-shifting focal-plane spot of π/2.
        This introduces a π/2 phase shift in a circular region
        centered on the focal plane, replacing the traditional pyramid structure.
        
        Args:
            p: FFT sampling parameter
            c: FFT padding parameter
            
        Returns:
            phase_mask: 2D array with π/2 phase shift in central spot
        """
        A = int((p + c) // 2)
        xx, yy = self.xp.mgrid[-A:A, -A:A].astype(self.dtype)
        # Convert radius from λ/D units to pixels
        # In focal plane, 1 λ/D corresponds to fft_padding/fft_sampling pixels
        fft_sampling = p
        fft_padding = c
        spot_radius_pixels = self.spot_radius_lambda * (1+fft_padding / fft_sampling) #(fft_padding / fft_sampling) 
        rr = self.xp.sqrt((xx+0.5)**2 + (yy+0.5)**2)
        phase_mask = self.xp.where(rr < spot_radius_pixels,
                                   self.phase_shift,
                                   0.0)
        return phase_mask