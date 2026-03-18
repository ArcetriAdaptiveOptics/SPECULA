from specula import cpuArray
from specula.processing_objects.slopec import Slopec
from specula.data_objects.slopes import Slopes
from skimage.restoration import unwrap_phase

class CiaoCiaoSlopec(Slopec):
    """
    Slope computer for the CiaoCiao WFS.
    Extracts the phase from an interferogram using the Fourier method:
    1. Computes the FFT of the interferogram.
    2. Isolates the carrier sideband using a Top-Flat Gaussian window.
    3. Shifts the sideband to the center.
    4. Computes the inverse FFT.
    5. Extracts the phase (arctan2) and (optionally) unwraps it.
    6. Converts the phase to OPD.
    7. Averages the OPD over the sector masks (petals) to get the signals (pistons).
    """
    def __init__(self,
                 wavelength_in_nm: float,
                 window_x_in_pix: float,
                 window_y_in_pix: float,
                 window_sigma_in_pix: float,
                 sector_masks: list,
                 unwrap: bool = False,
                 sn: Slopes = None,
                 target_device_idx: int = None,
                 precision: int = None,
                 **kwargs):
        """
        Parameters
        ----------
        wavelength_in_nm : float
            Working wavelength (e.g., 2200 for K band).
        window_x_in_pix, window_y_in_pix : float
            Coordinates of the sideband center in the FFT.
        window_sigma_in_pix : float
            Width of the filtering window (Top Flat Gaussian).
        sector_masks : list of ndarray
            List of 2D boolean masks, one for each pupil sector,
            used to compute the average piston.
        unwrap : bool, optional
            If True, performs 2D phase unwrapping using skimage (runs on CPU).
            Default is False.
        """
        self.wavelength = float(wavelength_in_nm)
        self.window_x = float(window_x_in_pix)
        self.window_y = float(window_y_in_pix)
        self.window_sigma = float(window_sigma_in_pix)
        self.unwrap = bool(unwrap)

        # The masks define the subapertures (sectors)
        self.sector_masks_host = sector_masks
        self._sector_masks_xp = None # Will be loaded onto the device in setup()

        self._window = None

        super().__init__(sn=sn,
                         target_device_idx=target_device_idx,
                         precision=precision,
                         **kwargs)

    def nsubaps(self):
        return len(self.sector_masks_host) if self.sector_masks_host else 1

    def nslopes(self):
        # For CiaoCiao, we consider the average piston of each sector as a "slope"
        return self.nsubaps()

    def setup(self):
        super().setup()

        # Transfer the sector masks to the current device (CPU/GPU)
        self._sector_masks_xp = [self.xp.asarray(m, dtype=bool) for m in self.sector_masks_host]

    def _build_window(self, shape):
        """Precomputes the Top Flat Gaussian Circular Window on the device."""
        x = self.xp.arange(0, shape[1])
        y = self.xp.arange(0, shape[0])
        xx, yy = self.xp.meshgrid(x, y)

        # Top Flat Gaussian: exp( - ( dx^2/2s^2 + dy^2/2s^2 )^2 )
        window = self.xp.exp(
            -((xx - self.window_x)**2 / (2 * self.window_sigma**2) + 
              (yy - self.window_y)**2 / (2 * self.window_sigma**2))**2
        )
        return window.astype(self.complex_dtype)

    def trigger_code(self):
        # Handle temporal accumulation (as in the IDL version)
        if self.weight_int_pixel_dt > 0:
            self.do_accumulation(self.current_time)

        # 1. Retrieve the interferogram (current pixels from the CCD)
        pixels = self.local_inputs['in_pixels'].pixels

        # Initialize the window on the first valid pass
        if self._window is None:
            self._window = self._build_window(pixels.shape)

        # 2. Fourier Transform and shift
        ft_intensity = self.xp.fft.fftshift(self.xp.fft.fft2(pixels, norm='ortho'))

        # 3. Apply the filtering window
        ft_filtered = ft_intensity * self._window

        # 4. Roll / Shift the sideband to the center
        shape = pixels.shape
        shift_y = int(self.xp.rint(shape[0] / 2 - self.window_y))
        shift_x = int(self.xp.rint(shape[1] / 2 - self.window_x))
        ft_roll = self.xp.roll(ft_filtered, (shift_y, shift_x), axis=(0, 1))

        # 5. Inverse FFT
        intensity_filtered = self.xp.fft.ifft2(self.xp.fft.fftshift(ft_roll))

        # 6. Phase extraction
        phase = self.xp.arctan2(intensity_filtered.imag, intensity_filtered.real)

        # 6.b Optional Unwrapping
        if self.unwrap:
            # Move phase to CPU for skimage unwrap_phase
            phase_cpu = cpuArray(phase)
            unwrapped_phase_cpu = unwrap_phase(phase_cpu)
            # Move it back to the current device (CPU/GPU)
            phase = self.to_xp(unwrapped_phase_cpu)

        # 7. Convert to OPD (wrapped or unwrapped)
        opd = phase * self.wavelength / (2 * self.xp.pi)

        # 8. Compute the piston for each sector
        if self._sector_masks_xp is not None:
            slopes_vec = self.xp.zeros(self.nslopes(), dtype=self.dtype)
            for i, mask in enumerate(self._sector_masks_xp):
                slopes_vec[i] = self.xp.mean(opd[mask])

            self.slopes.slopes[:] = slopes_vec

        # Diagnostic outputs: total and subaperture fluxes
        flux_per_sub = self.xp.array([self.xp.sum(pixels[mask]) for mask in self._sector_masks_xp])
        self.flux_per_subaperture_vector.value[:] = flux_per_sub
        self.total_counts.value[0] = self.xp.sum(flux_per_sub)
        self.subap_counts.value[0] = self.xp.mean(flux_per_sub)
