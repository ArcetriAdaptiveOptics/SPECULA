from specula.processing_objects.modulated_pyramid import ModulatedPyramid

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
                 spot_radius_lambda: float= 1.0,  # Spot radius in λ/D units
                 phase_shift_pi: float = 0.5,  # π/2 phase shift
                 fft_res: float = 4.0,
                 target_device_idx=None,
                 precision=None):

        self.spot_radius_lambda = spot_radius_lambda
        self.phase_shift_pi = phase_shift_pi

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
            fov_errsup=10.0,
            target_device_idx=target_device_idx,
            precision=precision
        )        


    def get_pyr_tlt(self, p, c):
        """
        Creates a phase-shifting focal-plane spot of self.phase_delay π.
        This introduces a self.phase_delay π phase shift in a circular region
        centered on the focal plane, replacing the traditional pyramid structure.
        
        Args:
            p: FFT sampling parameter
            c: FFT padding parameter
            
        Returns:
            phase_mask: 2D array with phase shift in central spot
        """
        A = int((p + c) // 2)

        # Create focal plane coordinates
        xx, yy = self.xp.mgrid[-A:A, -A:A].astype(self.dtype)

        # Convert radius from λ/D units to pixels
        # In focal plane, 1 λ/D corresponds to fft_totsize/fft_sampling pixels
        fft_sampling = p
        fft_padding = c
        spot_radius_pixels = self.spot_radius_lambda * (1+fft_padding/fft_sampling)

        # Calculate distance from center
        rr = self.xp.sqrt((xx+0.5)**2 + (yy+0.5)**2)

        # Create phase mask: self.phase_shift_pi
        phase_mask = self.xp.where(rr <= spot_radius_pixels,
                                   self.phase_shift_pi/2, # phase is multiplied by 2π during super().__init__
                                   0.0)

        return phase_mask