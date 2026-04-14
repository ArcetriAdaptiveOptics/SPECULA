
from specula import fuse
from specula.processing_objects.psf import PSF
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams


@fuse(kernel_name='psf_abs2')
def psf_abs2(v, xp):
    return xp.real(v * xp.conj(v))

class PsfCoronagraph(PSF):
    """
    Perfect coronagraph processing object..
    The implementation includes the standard PSF calculation as it inherits from the PSF class.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object.
    wavelengthInNm : float
        Wavelength at which to compute the PSF [nm].
    nd : float, optional
        Numerical density of the PSF (pixels per lambda/D). If None, it is calculated
        based on the input ElectricField and pixel size.
    use_average_field : bool, optional
        If True, the average electric field over the pupil is subtracted to compute the coronagraph PSF.
        If False, the perfect coronagraph formula is applied for the computation. Default is True (average field removal).
        The perfect coronagraph formula is Equation (1) in Cavarroc et al. 2006
    pixel_size_mas : float, optional
        Desired pixel size of the PSF in milliarcseconds. If None, it is calculated
        based on the input ElectricField and numerical density.
    start_time : float, optional
        Time (in seconds) after which to start integrating PSF and SR. Default is 0.0.
    compute_profile_metrics : bool, optional
        If True, compute coronagraph radial-profile outputs for the instantaneous,
        integrated and standard-deviation coronagraph PSFs.
    compute_metrics_in_trigger : bool, optional
        If True, update those metrics after each trigger as well.
    ee_radius_in_lambda_d : float or array-like, optional
        Radius or radii in units of lambda/D at which to return the encircled energy.
    target_device_idx : int, optional
        Target device index for computation (CPU/GPU). Default is None (uses global setting).
    precision : int, optional
        Precision for computation (0 for double, 1 for single). Default is None (uses global setting).
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 nd: float=None,
                 use_average_field:bool = True,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 compute_profile_metrics: bool=False,
                 compute_metrics_in_trigger: bool=False,
                 ee_radius_in_lambda_d=None,
                 target_device_idx: int = None,
                 precision: int = None,
                 verbose:bool = True,
                ):
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            nd=nd,
            pixel_size_mas=pixel_size_mas,
            start_time=start_time,
            compute_profile_metrics=compute_profile_metrics,
            compute_metrics_in_trigger=compute_metrics_in_trigger,
            ee_radius_in_lambda_d=ee_radius_in_lambda_d,
            target_device_idx=target_device_idx,
            precision=precision,
            verbose=verbose,
        )
        self.use_average_field = use_average_field

        # Additional outputs for coronagraph
        self.coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                         precision=precision)
        self.int_coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=precision)
        self.std_coronagraph_psf = BaseValue(target_device_idx=self.target_device_idx,
                                             precision=precision)
        self.coronagraph_psf_profile = BaseValue(target_device_idx=self.target_device_idx,
                                                 precision=precision)
        self.int_coronagraph_psf_profile = BaseValue(target_device_idx=self.target_device_idx,
                                                     precision=precision)
        self.std_coronagraph_psf_profile = BaseValue(target_device_idx=self.target_device_idx,
                                                     precision=precision)

        self.outputs['out_coronagraph_psf'] = self.coronagraph_psf
        self.outputs['out_int_coronagraph_psf'] = self.int_coronagraph_psf
        self.outputs['out_std_coronagraph_psf'] = self.std_coronagraph_psf
        self.outputs['out_coronagraph_psf_profile'] = self.coronagraph_psf_profile
        self.outputs['out_int_coronagraph_psf_profile'] = self.int_coronagraph_psf_profile
        self.outputs['out_std_coronagraph_psf_profile'] = self.std_coronagraph_psf_profile

        # Reference complex amplitude for perfect coronagraph
        self.ref_complex_amplitude = None
        self._sum_coronagraph_psf_squared = None # For std dev calculation

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil')}

    @classmethod
    def output_names(cls):
        return {
            'out_sr': OutputDesc(BaseValue, 'Instantaneous Strehl ratio'),
            'out_psf': OutputDesc(BaseValue, 'Instantaneous PSF'),
            'out_int_sr': OutputDesc(BaseValue, 'Time-integrated Strehl ratio'),
            'out_int_psf': OutputDesc(BaseValue, 'Time-integrated PSF'),
            'out_std_psf': OutputDesc(BaseValue, 'Standard deviation of PSF over time'),
            'out_psf_profile': OutputDesc(BaseValue, 'Radial profile of the instantaneous PSF'),
            'out_psf_fwhm': OutputDesc(BaseValue, 'FWHM of the instantaneous PSF'),
            'out_encircled_energy': OutputDesc(BaseValue, 'Encircled energy of the instantaneous PSF'),
            'out_encircled_energy_at_radius': OutputDesc(BaseValue, 'Encircled energy at specified radius for instantaneous PSF'),
            'out_int_psf_profile': OutputDesc(BaseValue, 'Radial profile of the integrated PSF'),
            'out_int_psf_fwhm': OutputDesc(BaseValue, 'FWHM of the integrated PSF'),
            'out_int_encircled_energy': OutputDesc(BaseValue, 'Encircled energy of the integrated PSF'),
            'out_int_encircled_energy_at_radius': OutputDesc(BaseValue, 'Encircled energy at specified radius for integrated PSF'),
            'out_coronagraph_psf': OutputDesc(BaseValue, 'Instantaneous coronagraph PSF'),
            'out_int_coronagraph_psf': OutputDesc(BaseValue, 'Time-integrated coronagraph PSF'),
            'out_std_coronagraph_psf': OutputDesc(BaseValue, 'Standard deviation of coronagraph PSF over time'),
            'out_coronagraph_psf_profile': OutputDesc(BaseValue, 'Radial profile of the instantaneous coronagraph PSF'),
            'out_int_coronagraph_psf_profile': OutputDesc(BaseValue, 'Radial profile of the integrated coronagraph PSF'),
            'out_std_coronagraph_psf_profile': OutputDesc(BaseValue, 'Radial profile of the std dev coronagraph PSF'),
        }

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

        # Step 2: Calculate  the field after the perfect coronagraph:
        # if self.use_average_field is True, we subtract the average electric field over the pupil
        # if self.use_average_field is False, we apply the perfect coronagraph formula.
        # The two formulas are equivalent at high angular separations, but the average field
        # removal produces infinite contrast near the PSF core.
        pupil_mask = amp > 0
        if self.xp.sum(pupil_mask) > 0:
            if self.use_average_field is True: # average field removal
                avg_electric_field = self.xp.sum(electric_field * pupil_mask) / self.xp.sum(pupil_mask)
                electric_field_corrected = electric_field - avg_electric_field * pupil_mask
            else: # perfect coronagraph formula (Cavarroc et al. 2006, Eq. 1)
                mean_phase = self.xp.sum(phase * pupil_mask) / self.xp.sum(pupil_mask)
                var_phase = self.xp.sum(((phase - mean_phase) ** 2) * pupil_mask) / self.xp.sum(pupil_mask)
                ec = self.xp.exp(-var_phase, dtype=self.dtype)
                coherent_core = self.xp.sqrt(ec) * self.xp.exp(1j * mean_phase, dtype=self.complex_dtype) * amp
                electric_field_corrected = electric_field - coherent_core * pupil_mask
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

        if self.verbose:
            print(f'Coronagraph peak suppression: '
                f'{self.coronagraph_psf.value.max()/self.psf.value.max():.2e}',
                flush=True)

    def post_trigger(self):
        super().post_trigger()

        if self.current_time_seconds >= self.start_time:
            self.int_coronagraph_psf.value += self.coronagraph_psf.value
            self._sum_coronagraph_psf_squared += self.coronagraph_psf.value ** 2

        self.coronagraph_psf.generation_time = self.current_time

        if self.compute_profile_metrics and self.compute_metrics_in_trigger:
            self._set_radial_profile_output(
                self.coronagraph_psf.value,
                self.coronagraph_psf_profile,
            )

    def finalize(self):
        super().finalize()

        if self.count > 0:
            self.int_coronagraph_psf.value /= self.count
            variance = self._sum_coronagraph_psf_squared / self.count - self.int_coronagraph_psf.value ** 2
            self.std_coronagraph_psf.value = self.xp.sqrt(self.xp.maximum(variance, 0))
            if self.compute_profile_metrics:
                self._set_radial_profile_output(
                    self.int_coronagraph_psf.value,
                    self.int_coronagraph_psf_profile,
                )
                self._set_radial_profile_output(
                    self.std_coronagraph_psf.value,
                    self.std_coronagraph_psf_profile,
                )

        self.int_coronagraph_psf.generation_time = self.current_time
        self.std_coronagraph_psf.generation_time = self.current_time
