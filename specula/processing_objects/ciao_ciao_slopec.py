from specula import cpuArray
from specula.lib.interp2d import Interp2D
from specula.data_objects.pupilstop import Pupilstop
from specula.processing_objects.slopec import Slopec
from specula.data_objects.slopes import Slopes
from skimage.restoration import unwrap_phase

class CiaoCiaoSlopec(Slopec):
    """
    Slope computer for the CiaoCiao WFS processing object.
    
    Extracts the phase from an interferogram using the Fourier method:
    1. Computes the FFT of the interferogram.
    2. Isolates the carrier sideband using a Top-Flat Gaussian window.
    3. Shifts the sideband to the center.
    4. Computes the inverse FFT.
    5. Extracts the phase (arctan2) and (optionally) unwraps it.
    6. Converts the phase to OPD.
    7. Exports the OPD map as a flattened output vector.
    """
    def __init__(self,
                 wavelength_in_nm: float,
                 window_x_in_pix: float,
                 window_y_in_pix: float,
                 window_sigma_in_pix: float,
                 pupil_mask: Pupilstop,
                 diffRotAngleInDeg: float = 0.0,
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
        pupil_mask : Pupilstop
            Pupil mask defining the valid area. Its ``.A`` amplitude array is
            used. The effective mask is the intersection
            of this mask with a copy rotated by ``diffRotAngleInDeg``, mirroring
            the overlap region seen by the CiaoCiao interferometer.
        diffRotAngleInDeg : float, optional
            Rotation angle in degrees applied to one branch of the interferometer
            (same value as ``diffRotAngleInDeg`` in CiaoCiaoSensor). The effective
            pupil mask is ``mask & rotate(mask, diffRotAngleInDeg)``.
            Default is 0.0 (no rotation applied, mask used as-is).
        unwrap : bool, optional
            If True, performs 2D phase unwrapping using skimage (runs on CPU).
            Default is False.
        """
        self.wavelength = float(wavelength_in_nm)
        self.window_x = float(window_x_in_pix)
        self.window_y = float(window_y_in_pix)
        self.window_sigma = float(window_sigma_in_pix)
        self.unwrap = bool(unwrap)
        self.diffRotAngleInDeg = float(diffRotAngleInDeg)
        self._nslopes = 1
        self._window = None

        super().__init__(sn=sn,
                         target_device_idx=target_device_idx,
                         precision=precision,
                         **kwargs)

        if pupil_mask is not None:
            mask = self.xp.asarray(pupil_mask.A, dtype=self.dtype) > 0.5
            if self.diffRotAngleInDeg != 0.0:
                interp = Interp2D(mask.shape, mask.shape,
                                  rotInDeg=self.diffRotAngleInDeg,
                                  dtype=self.dtype, xp=self.xp)
                rotated = interp.interpolate(mask.astype(self.dtype)) > 0.5
                mask = mask & rotated
            self._pupil_mask_xp = mask
        else:
            self._pupil_mask_xp = None

    def nsubaps(self):
        return 1

    def nslopes(self):
        return self._nslopes

    def setup(self):
        super().setup()

        in_pixels = self.local_inputs['in_pixels']
        nslopes = int(in_pixels.pixels.shape[0] * in_pixels.pixels.shape[1])
        if self.slopes.size != nslopes:
            self.slopes.resize(nslopes)
        self._nslopes = nslopes

        x = self.xp.arange(0, in_pixels.pixels.shape[1])
        y = self.xp.arange(0, in_pixels.pixels.shape[0])
        xx, yy = self.xp.meshgrid(x, y)

        # Top Flat Gaussian: exp( - ( dx^2/2s^2 + dy^2/2s^2 )^2 )
        window = self.xp.exp(
            -((xx - self.window_x)**2 / (2 * self.window_sigma**2) +
              
              (yy - self.window_y)**2 / (2 * self.window_sigma**2))**2
        )

        self._window = window.astype(self.complex_dtype)

    def trigger_code(self):
        # 1. Retrieve the interferogram (current pixels from the CCD)
        pixels = self.local_inputs['in_pixels'].pixels

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

        # 8. Export OPD map as a flattened vector
        self.slopes.slopes[:] = opd.ravel()

        # Diagnostic outputs: mean flux per pixel within the pupil mask
        if self._pupil_mask_xp is not None:
            flux_mean = self.xp.mean(pixels[self._pupil_mask_xp])
        else:
            flux_mean = self.xp.mean(pixels)
        self.flux_per_subaperture_vector.value[:] = flux_mean
        self.total_counts.value[0] = flux_mean
        self.subap_counts.value[0] = flux_mean
