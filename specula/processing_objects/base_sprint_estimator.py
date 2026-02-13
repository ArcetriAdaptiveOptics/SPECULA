"""
Base SPRINT Estimator class with common demodulation and iteration logic.
Specific WFS implementations should inherit from this class.
"""

from abc import abstractmethod

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.base_value import BaseValue
from specula.lib.demodulate_signal import demodulate_signal
from specula.processing_objects.dm import DM
from specula.processing_objects.slopec import Slopec
from specula import cpuArray


class BaseSprintEstimator(BaseProcessingObj):
    """
    Base SPRINT (System Parameters Recurrent Invasive Tracking) Estimator.
    
    Online calibration of WFS-DM mis-registration parameters using:
    1. Slope demodulation to extract measured interaction matrix
    2. WFS-specific sensitivity matrix computation (implemented in subclasses)
    3. Iterative parameter refinement with optional integration/forgetting
    
    Based on: Heritier+ 2021, MNRAS "SPRINT: a fast and least-cost 
              online calibration strategy for adaptive optics"
    
    This base class handles:
    - Slope collection and demodulation
    - Iterative estimation loop
    - Integration with gain and forgetting factor
    - Input/output management
    
    Subclasses must implement:
    - _compute_nominal_im(): Compute IM with current misreg parameters
    - _compute_sensitivity_matrices(): Compute sensitivity matrices
    - _validate_wfs(): Check WFS compatibility
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters
    dm : DM
        Deformable mirror object
    slopec : Slopec
        Slope computer object
    source : Source
        Guide star source object
    wfs : BaseProcessingObj
        WFS object (specific type depends on subclass)
    modes_index : list
        List of mode indices to estimate
    carrier_frequencies : list
        Carrier frequencies for each mode [Hz]
    estimation_dt : float
        Time interval between estimations [seconds]
    max_iterations : int
        Maximum iterations per estimation cycle
    convergence_threshold : float
        Relative error threshold for convergence
    initial_misreg : list or None
        Initial mis-registration [shift_x, shift_y, rot, magn(, magn_x, magn_y)]
    apply_absolute_slopes : bool
        Use absolute value of slopes
    enable_wpup_magn_xy : bool
        Enable separate X/Y magnification parameters
    integration_gain : float
        Gain for parameter updates (0 < gain <= 1)
    forgetting_factor : float or None
        Forgetting factor for integration (0 < factor <= 1, 1 = no forgetting)
    target_device_idx : int or None
        GPU device index
    precision : int or None
        Numerical precision
    
    Inputs
    ------
    in_slopes : Slopes
        Current WFS slopes (modulated by pushpull_generator)
    
    Outputs
    -------
    out_intmat : Intmat
        Estimated interaction matrix
    out_misreg_params : BaseValue
        Estimated mis-registration parameters
    out_convergence_error : BaseValue
        Current relative error
    """

    def __init__(self,
                 simul_params: SimulParams,
                 dm: DM,
                 slopec: Slopec,
                 source: Source,
                 wfs: BaseProcessingObj,
                 modes_index: list,
                 carrier_frequencies: list,
                 n_params: int = 4,  # Default: shift_x, shift_y, rotation, magnification
                 estimation_dt: float = 10.0,
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-3,
                 initial_misreg: list = None,
                 apply_absolute_slopes: bool = False,
                 integration_gain: float = 0.5,
                 forgetting_factor: float = 1.0,
                 verbose: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Store references
        self.simul_params = simul_params
        self.dm = dm
        self.slopec = slopec
        self.source = source
        self.wfs = wfs

        # Validate WFS type (implemented in subclass)
        self._validate_wfs()

        # Mode configuration
        if len(carrier_frequencies) != len(modes_index):
            raise ValueError("carrier_frequencies and modes_index must have same length")

        self.modes_index = modes_index
        self.nmodes = len(modes_index)
        self.carrier_frequencies = self.xp.array(carrier_frequencies, dtype=self.dtype)

        # Estimation parameters
        self.estimation_dt = self.seconds_to_t(estimation_dt)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.apply_absolute_slopes = apply_absolute_slopes

        # Integration parameters
        if not 0 < integration_gain <= 1:
            raise ValueError(f"integration_gain must be in (0, 1], got {integration_gain}")
        self.integration_gain = integration_gain

        if not 0 < forgetting_factor <= 1:
            raise ValueError(f"forgetting_factor must be in (0, 1], got {forgetting_factor}")
        self.forgetting_factor = forgetting_factor

        self.verbose = verbose

        # Initialize mis-registration parameters
        self.n_params = n_params
        if initial_misreg is None:
            self.misreg_params = self.xp.zeros(self.n_params, dtype=self.dtype)
        else:
            if len(initial_misreg) != self.n_params:
                raise ValueError(f"initial_misreg must have {self.n_params} elements")
            self.misreg_params = self.to_xp(initial_misreg, dtype=self.dtype)

        # State variables
        self.last_estimation_time = 0
        self.current_error = 0.0
        self.slopes_history = []
        self.time_history = []

        # Pupil parameters (extracted from DM)
        self.pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
        self.pup_mask = None  # Loaded in setup
        self.ifunc_3d = None  # Loaded in setup

        # Create outputs
        self.estimated_intmat = Intmat(
            nmodes=self.nmodes,
            nslopes=0,
            target_device_idx=target_device_idx,
            precision=precision
        )
        self.misreg_output = BaseValue(
            value=self.misreg_params.copy(),
            target_device_idx=target_device_idx,
            precision=precision
        )
        self.error_output = BaseValue(
            value=self.xp.array([0.0], dtype=self.dtype),
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Setup connections
        self.inputs['in_slopes'] = InputValue(type=Slopes)
        self.outputs['out_intmat'] = self.estimated_intmat
        self.outputs['out_misreg_params'] = self.misreg_output
        self.outputs['out_convergence_error'] = self.error_output

    @abstractmethod
    def _validate_wfs(self):
        """Validate that WFS is compatible with this estimator. Raise ValueError if not."""
        pass

    @abstractmethod
    def _compute_nominal_im(self):
        """
        Compute nominal interaction matrix with current mis-registration parameters.
        
        Returns
        -------
        im_nominal : ndarray, shape (nslopes, nmodes)
            Nominal interaction matrix
        """
        pass

    @abstractmethod
    def _compute_sensitivity_matrices(self):
        """
        Compute sensitivity matrices for all mis-registration parameters.
        
        Returns
        -------
        sens_matrices : ndarray, shape (nslopes, nmodes, n_params)
            Sensitivity of each slope/mode to each parameter
        """
        pass

    def setup(self):
        """Initialize slopes size and extract parameters"""
        super().setup()

        # Get initial slopes
        in_slopes = self.local_inputs['in_slopes']
        if in_slopes is None:
            raise ValueError("in_slopes must be connected before setup")

        # Initialize IM size
        self.estimated_intmat.set_nslopes(len(in_slopes.slopes))

        # Extract DM parameters
        self.ifunc_3d = cpuArray(self.dm.ifunc_obj.ifunc_2d_to_3d(normalize=True))
        self.ifunc_3d = self.ifunc_3d[:, :, self.modes_index]  # Extract only requested modes
        self.pup_mask = cpuArray(self.dm.mask)

        if self.verbose: # pragma: no cover
            print(f"\n{self.__class__.__name__} initialized:")
            print(f"  Number of modes: {self.nmodes}")
            print(f"  Number of slopes: {self.estimated_intmat.nslopes}")
            print(f"  Size of pupil: {self.pup_mask.shape}")
            print(f"  Size of DM influence functions: {self.ifunc_3d.shape}")
            print(f"  Estimation interval: {self.t_to_seconds(self.estimation_dt):.2f}s")
            print(f"  Integration gain: {self.integration_gain}")
            print(f"  Forgetting factor: {self.forgetting_factor}")

    def prepare_trigger(self, t):
        """Collect slopes for demodulation"""
        super().prepare_trigger(t)

        in_slopes = self.local_inputs['in_slopes']
        self.slopes_history.append(in_slopes.slopes.copy())
        self.time_history.append(t)

    def trigger_code(self):
        """Main SPRINT estimation logic"""
        t = self.current_time

        # Check if it's time to estimate
        if (t - self.last_estimation_time) < self.estimation_dt:
            return

        if self.verbose: # pragma: no cover
            print(f"\n{'='*60}")
            print(f"SPRINT Estimation at t={self.t_to_seconds(t):.2f}s")
            print(f"{'='*60}")

        # Demodulate slopes
        im_measured = self._demodulate_slopes()
        if im_measured is None:
            if self.verbose: # pragma: no cover
                print("  Not enough data for demodulation yet")
            return

        # Iterative estimation
        self._iterative_estimation(im_measured)

        # Update state
        self.last_estimation_time = t

        # Update outputs
        self.estimated_intmat.generation_time = t
        self.misreg_output.value = self.misreg_params.copy()
        self.misreg_output.generation_time = t
        self.error_output.value = self.xp.array([self.current_error], dtype=self.dtype)
        self.error_output.generation_time = t

    def _demodulate_slopes(self):
        """
        Demodulate slopes history to extract measured IM.
        
        Returns
        -------
        im_measured : ndarray, shape (nslopes, nmodes) or None
        """
        if len(self.slopes_history) < 10:
            return None

        slopes_array = self.xp.stack(self.slopes_history)  # Shape: (nt, nslopes)
        nslopes = slopes_array.shape[1]
        im_measured = self.xp.zeros((nslopes, self.nmodes), dtype=self.dtype)

        dt = self.simul_params.time_step
        sampling_freq = 1.0 / dt

        if self.verbose: # pragma: no cover
            print(f"  Demodulating {len(self.slopes_history)} time samples")
            print(f"  Number of slopes: {nslopes}")
            print(f"  Number of modes: {self.nmodes}")

        # Demodulate each mode (vectorized across all slopes)
        for mode_idx in range(self.nmodes):
            carrier_freq = float(self.carrier_frequencies[mode_idx])

            if self.verbose: # pragma: no cover
                print(f"  Mode {mode_idx}: carrier = {carrier_freq:.2f} Hz")

            # Vectorized demodulation for all slopes at once
            # slopes_array shape: (nt, nslopes)
            # demodulate_signal expects: (nt, nsignals)
            amplitudes, phases = demodulate_signal(
                slopes_array,  # Convert to CPU for demodulation
                carrier_freq,
                sampling_freq,
                xp=self.xp,
                dtype=self.dtype
            )

            # amplitudes shape: (nslopes,)
            # phases shape: (nslopes,)

            # Apply phase correction (vectorized)
            signed_amplitudes = amplitudes * self.xp.sign(self.xp.cos(phases))

            # Store in IM
            im_measured[:, mode_idx] = signed_amplitudes

        # Apply absolute value if requested
        if self.apply_absolute_slopes:
            im_measured = self.xp.abs(im_measured)

        # Clear history
        self.slopes_history = []
        self.time_history = []

        if self.verbose: # pragma: no cover
            print(f"  Demodulated IM shape: {im_measured.shape}")
            print(f"  IM RMS: {float(self.xp.sqrt(self.xp.mean(im_measured**2))):.3e}")

        return im_measured

    def _plot_debug_info(self, im_measured, im_nominal,
                         im_diff, G_opt, iteration):
        """Plot debug information for SH WFS."""
        pass  # Implemented in subclass if needed

    def _iterative_estimation(self, im_measured):
        """
        Iterative estimation loop with integration and forgetting.
        
        Parameters
        ----------
        im_measured : ndarray
            Measured IM from demodulation
        """
        if self.verbose: # pragma: no cover
            print(f"\n  Starting iterative estimation...")
            print(f"  Initial misreg: {cpuArray(self.misreg_params)}")

        params_before = self.misreg_params.copy()

        # Compute reference norm once (for relative error)
        im_measured_norm = float(self.xp.sqrt(self.xp.mean(im_measured**2)))

        for iteration in range(self.max_iterations):
            # Compute nominal IM (subclass-specific)
            im_nominal = self._compute_nominal_im()

            # Compute optical gains
            G_opt = self._compute_optical_gains(im_measured, im_nominal)

            # Mean absolute gain (for error weighting)
            G_mean = float(self.xp.mean(self.xp.abs(G_opt)))

            if self.verbose: # pragma: no cover
                print(f"    Optical gains: {cpuArray(G_opt)}, mean: {G_mean:.3f}")

            # Compute sensitivities (subclass-specific)
            sens_matrices = self._compute_sensitivity_matrices()

            # Corrected difference
            im_diff = self._apply_optical_gain_correction(im_measured, G_opt) - im_nominal

            plot_debug = False  # Set to True to enable debug plotting
            if plot_debug: # pragma: no cover
                self._plot_debug_info(im_measured, im_nominal, im_diff, G_opt, iteration)

            # Estimate correction
            delta_misreg = self._estimate_misreg_correction(im_diff, sens_matrices)

            # Apply integration with forgetting
            self.misreg_params = (self.misreg_params * self.forgetting_factor +
                                 delta_misreg * self.integration_gain)

            # Compute weighted relative error
            # If gains are close to 1.0, this reduces to standard relative error
            error_abs = float(self.xp.sqrt(self.xp.mean(im_diff**2)))
            error_rel = error_abs / im_measured_norm
            error_weighted = error_rel * G_mean  # Weight by mean gain
            self.current_error = error_rel

            if self.verbose: # pragma: no cover
                print(f"    Iteration {iteration+1}: error_rel={error_rel:.3e}, "
                    f"error_weighted={error_weighted:.3e}, params={cpuArray(self.misreg_params)}")

            # Check convergence on weighted error
            if error_weighted < self.convergence_threshold:
                if self.verbose: # pragma: no cover
                    print(f"  Converged after {iteration+1} iterations!")
                break

        # Store final IM
        self.estimated_intmat.intmat = self._compute_nominal_im()

        if self.verbose: # pragma: no cover
            delta_total = self.misreg_params - params_before
            print(f"  Final params: {cpuArray(self.misreg_params)}")
            print(f"  Total change: {cpuArray(delta_total)}")

    def _compute_optical_gains(self, im_measured, im_nominal):
        """Compute optical gains for each mode"""
        rec = self.xp.linalg.pinv(im_nominal)
        return self.xp.diag(rec @ im_measured)

    def _apply_optical_gain_correction(self, im_measured, G_opt):
        """Apply optical gain correction"""
        G_inv = 1.0 / (G_opt + 1e-12)
        return im_measured * G_inv[self.xp.newaxis, :]

    def _estimate_misreg_correction(self, im_diff, sens_matrices):
        """
        Estimate mis-registration correction from IM difference.
        
        Parameters
        ----------
        im_diff : ndarray, shape (nslopes, nmodes)
        sens_matrices : ndarray, shape (nslopes, nmodes, n_params)
        
        Returns
        -------
        delta_misreg : ndarray, shape (n_params,)
        """
        n_params = sens_matrices.shape[2]
        delta_misreg = self.xp.zeros(n_params, dtype=self.dtype)

        if im_diff.ndim == 1:
            im_diff = im_diff[:, self.xp.newaxis]

        # Solve for each parameter
        for p in range(n_params):
            sens_p = sens_matrices[:, :, p]
            deltas = []

            for m in range(self.nmodes):
                sens_col = sens_p[:, m]
                diff_col = im_diff[:, m] if im_diff.shape[1] > m else im_diff[:, 0]

                delta = self.xp.dot(sens_col, diff_col) / \
                       (self.xp.dot(sens_col, sens_col) + 1e-12)
                deltas.append(delta)

            delta_misreg[p] = self.xp.mean(self.xp.array(deltas))

        return delta_misreg
