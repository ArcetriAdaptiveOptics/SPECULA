
from specula import fuse
from specula.processing_objects.psf import PSF
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))

class PsfCoronagraph(PSF):
    """
    Perfect coronagraph implementation.
    It includes the standard PSF calculation since it inherits from the PSF class.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object.
    wavelengthInNm : float
        Wavelength at which to compute the PSF [nm].
    nd : float, optional
        Numerical density of the PSF (pixels per lambda/D). If None, it is calculated
        based on the input ElectricField and pixel size.
    pixel_size_mas : float, optional
        Desired pixel size of the PSF in milliarcseconds. If None, it is calculated
        based on the input ElectricField and numerical density.
    start_time : float, optional
        Time (in seconds) after which to start integrating PSF and SR. Default is 0.0.
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None (uses global setting).
    """
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
        self.coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                         precision=precision)
        self.int_coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=precision)
        self.std_coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=precision)

        self.outputs['out_coronagraph_psf'] = self.coronagraph_psf
        self.outputs['out_int_coronagraph_psf'] = self.int_coronagraph_psf
        self.outputs['out_std_coronagraph_psf'] = self.std_coronagraph_psf

        # Reference complex amplitude for perfect coronagraph
        self.ref_complex_amplitude = None
        self._sum_coronagraph_psf_squared = None # For std dev calculation

    def setup(self):
        super().setup()
        # Initialize integrated coronagraph PSF
        self.int_coronagraph_psf.value = self.xp.zeros_like(self.int_psf.value)
        self._sum_coronagraph_psf_squared = self.xp.zeros_like(self.int_psf.value)
        self.std_coronagraph_psf.value = self.xp.zeros_like(self.std_psf.value)

    def calc_coronagraph_psf(self, phase, amp, imwidth=None, normalize=False, nocenter=False):
        """
        Calculate coronagraph PSF using perfect coronagraph theory.
        The perfect coronagraph subtracts the average electric field over the pupil.
        
        Parameters:
        phase : ndarray
            2D phase array
        amp : ndarray
            2D amplitude array
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
        # Step 1: Calculate electric field from incoming phase screen
        electric_field = amp * self.xp.exp(1j * phase, dtype=self.complex_dtype)

        # Step 2: Calculate and subtract the average electric field over the pupil
        # Only consider pixels where amplitude > 0 (inside pupil)
        pupil_mask = amp > 0
        if self.xp.sum(pupil_mask) > 0:
            avg_electric_field = self.xp.sum(electric_field * pupil_mask) / self.xp.sum(pupil_mask)
            electric_field_corrected = electric_field - avg_electric_field * pupil_mask
        else:
            electric_field_corrected = electric_field

        # Set up the complex array based on input dimensions and data type
        if imwidth is not None:
            u_ef = self.xp.zeros((imwidth, imwidth), dtype=self.complex_dtype)
            s = electric_field_corrected.shape
            u_ef[:s[0], :s[1]] = electric_field_corrected
        else:
            u_ef = electric_field_corrected

        # Step 3: Optical Fourier transform to focal plane
        focal_field = self.xp.fft.fft2(u_ef)

        # Center if required
        if not nocenter:
            focal_field = self.xp.fft.fftshift(focal_field)

        # Calculate PSF as square modulus
        coronagraph_psf = psf_abs2(focal_field, xp=self.xp)

        # Normalize if required
        if normalize:
            coronagraph_psf /= self.total_psf

        return coronagraph_psf

    def trigger_code(self):
        # Call parent trigger_code for standard PSF calculation
        super().trigger_code()

        in_ef = self.local_inputs['in_ef']

        # Calculate coronagraph PSF
        self.coronagraph_psf.value = self.calc_coronagraph_psf(
            in_ef.phi_at_lambda(self.wavelengthInNm),
            in_ef.A,
            imwidth=self.out_size[0],
            normalize=True
        )

        print(f'Coronagraph peak suppression: '
              f'{self.coronagraph_psf.value.max()/self.psf.value.max():.2e}',
              flush=True)

    def post_trigger(self):
        super().post_trigger()

        if self.current_time_seconds >= self.start_time:
            self.int_coronagraph_psf.value += self.coronagraph_psf.value
            self._sum_coronagraph_psf_squared += self.coronagraph_psf.value ** 2

        self.coronagraph_psf.generation_time = self.current_time

    def finalize(self):
        super().finalize()

        if self.count > 0:
            self.int_coronagraph_psf.value /= self.count
            self.std_coronagraph_psf.value = self.xp.sqrt(
                self._sum_coronagraph_psf_squared / self.count - self.int_coronagraph_psf.value ** 2
            )

        self.int_coronagraph_psf.generation_time = self.current_time
        self.std_coronagraph_psf.generation_time = self.current_time
