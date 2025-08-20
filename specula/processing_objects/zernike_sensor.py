from specula import RAD2ASEC
from specula.processing_objects.modulated_pyramid import ModulatedPyramid

class ZernikeSensor(ModulatedPyramid):
    """
    A Zernike sensor based on phase-shifting focal-plane spot technique.
    Inherits from ModulatedPyramid but replaces the pyramid structure with
    a π/2 phase-shifting spot in the focal plane.
    """

    def __init__(self,
                 simul_params,
                 wavelengthInNm,
                 fov,
                 pup_diam,
                 output_resolution,
                 spot_radius_lambda=0.5,  # Spot radius in λ/D units
                 target_device_idx=None,
                 precision=None):

        self.spot_radius_lambda = spot_radius_lambda

        # Force modulation to zero (no modulation for Zernike sensor)
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_step=1,
            target_device_idx=target_device_idx,
            precision=precision
        )

    def calc_geometry(self,
        DpupPix,                # number of pixels of input phase array
        pixel_pitch,            # pixel sampling [m] of DpupPix
        lambda_,                # working lambda of the sensor [nm]
        FoV,                    # requested FoV in arcsec
        pup_diam,               # pupil diameter in subapertures
        ccd_side,               # requested output ccd side, in pixels
        fov_errinf=0.1,         # accepted error in reducing FoV, default = 0.1 (-10%)
        fov_errsup=0.5,         # accepted error in enlarging FoV, default = 0.5 (+50%)
        pup_dist=None,          # pupil distance in subapertures, optional
        pup_margin=2,           # zone of respect around pupils for margins, optional, default=2px
        fft_res=3.0,            # requested minimum PSF sampling, 1.0 = 1 pixel / PSF, default=3.0
        min_pup_dist=None,
        NOTEST=False            # skip the time estimation done with a test pyramid
    ):
        # Calculate pup_distance if not given, using the pup_margin
        if pup_dist is None:
            pup_dist = pup_diam + pup_margin * 2

        min_ccd_side = pup_diam
        if ccd_side < min_ccd_side:
            print(f"Error: ccd_side (px) = {ccd_side} is not enough to hold the pupil geometry. Minimum allowed side is {min_ccd_side}")
            return 0

        D = DpupPix * pixel_pitch
        Fov_internal = lambda_ * 1e-9 / D * (D / pixel_pitch) * RAD2ASEC

        minfov = FoV * (1 - fov_errinf)
        maxfov = FoV * (1 + fov_errsup)
        fov_res = 1.0

        if Fov_internal < minfov:
            fov_res = int(minfov / Fov_internal)
            if Fov_internal * fov_res < minfov:
                fov_res += 1

        if Fov_internal > maxfov:
            print("Error: Calculated FoV is higher than maximum accepted FoV.")
            print("Please revise error margin, or the input phase dimension and/or pitch")
            return 0

        if fov_res > 1:
            Fov_internal *= fov_res
            print(f"Interpolated FoV (arcsec): {Fov_internal:.2f}")
            print(f"Warning: reaching the requested FoV requires {fov_res}x interpolation of input phase array.")
            print("Consider revising the input phase dimension and/or pitch to improve performance.")

        fp_masking = FoV / Fov_internal

        if Fov_internal != FoV:
            print(f"FoV reduction from {Fov_internal:.2f} to {FoV:.2f} will be performed with a focal plane mask")

        DpupPixFov = DpupPix * fov_res

        internal_ccd_side = self.xp.around(fft_res * pup_diam / 2) * 2
        fft_res = internal_ccd_side / float(pup_diam)

        totsize = self.xp.around(DpupPixFov * fft_res / 2) * 2
        fft_res = totsize / float(DpupPixFov)

        padding = self.xp.around((DpupPixFov * fft_res - DpupPixFov) / 2) * 2

        results = {
            'fov_res': fov_res,
            'fp_masking': fp_masking,
            'fft_res': fft_res,
            'tilt_scale': fft_res / ((pup_dist / float(pup_diam)) / 2.0),
            'fft_sampling': int(DpupPixFov),
            'fft_padding': int(padding),
            'fft_totsize': int(totsize),
            'wavelengthInNm': lambda_,
            'toccd_side': internal_ccd_side,
            'final_ccd_side': ccd_side
        }

        return results

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

        # Create focal plane coordinates
        xx, yy = self.xp.mgrid[-A:A, -A:A].astype(self.dtype)

        # Convert radius from λ/D units to pixels
        # In focal plane, 1 λ/D corresponds to approximately A/fft_res pixels
        spot_radius_pixels = self.spot_radius_lambda * A / self.fft_res

        # Calculate distance from center
        rr = self.xp.sqrt(xx**2 + yy**2)

        # Create phase mask: π/2 inside circle, 0 outside
        phase_mask = self.xp.where(rr <= spot_radius_pixels,
                                   self.xp.pi / 2,
                                   0.0)

        # Alternative: soft transition spot (Gaussian)
        # sigma = spot_radius_pixels / 2
        # phase_mask = (self.xp.pi / 2) * self.xp.exp(-(rr**2) / (2 * sigma**2))

        return phase_mask / self.tilt_scale