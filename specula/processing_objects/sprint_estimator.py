import os

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.lib.demodulate_signal import demodulate_signal
from specula import xp, cpuArray

# Import SynIM for sensitivity matrix computation
try:
    import synim.synim as synim
    from synim.params_manager import ParamsManager
    SYNIM_AVAILABLE = True
except ImportError:
    SYNIM_AVAILABLE = False
    print("Warning: SynIM not available. SPRINT estimator will not work.")


class SprintEstimator(BaseProcessingObj):
    """
    SPRINT (System Parameters Recurrent Invasive Tracking) Estimator.
    
    Online calibration of WFS-DM mis-registration parameters using:
    1. Slope demodulation to extract measured interaction matrix
    2. SynIM-based sensitivity matrices
    3. Iterative parameter refinement
    
    Based on: Heritier+ 2021, MNRAS "SPRINT: a fast and least-cost 
              online calibration strategy for adaptive optics"
    
    Inputs
    ------
    in_slopes : Slopes
        Current WFS slopes (modulated by pushpull_generator)
    
    Outputs
    -------
    out_intmat : Intmat
        Estimated interaction matrix with corrected mis-registration
    out_misreg_params : BaseValue
        Estimated mis-registration parameters [shift_x, shift_y, rot, magn]
    """

    def __init__(self,
                 simul_params: SimulParams,
                 params_manager: ParamsManager,  # SynIM params manager
                 wfs_index: int = 0,
                 dm_index: int = 0,
                 carrier_frequencies: list = None,
                 estimation_dt: float = 10.0,  # Demodulation AND estimation interval
                 max_iterations: int = 10,
                 convergence_threshold: float = 1e-3,
                 initial_misreg: list = None,
                 apply_absolute_slopes: bool = False,
                 enable_wpup_magn_xy: bool = False,
                 data_dir: str = None,
                 im_tag: str = None,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if not SYNIM_AVAILABLE:
            raise RuntimeError("SynIM is required for SprintEstimator")

        self.simul_params = simul_params
        self.params_manager = params_manager
        self.wfs_index = wfs_index
        self.dm_index = dm_index
        self.carrier_frequencies = self.xp.array(carrier_frequencies, dtype=self.dtype)
        self.estimation_dt = self.seconds_to_t(estimation_dt)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.apply_absolute_slopes = apply_absolute_slopes
        self.enable_wpup_magn_xy = enable_wpup_magn_xy
        self.data_dir = data_dir or simul_params.root_dir
        self.im_tag = im_tag or 'sprint_estimated_im'
        self.overwrite = overwrite

        # Initialize mis-registration parameters: [shift_x, shift_y, rot, magn(, magn_x, magn_y)]
        n_params = 6 if enable_wpup_magn_xy else 4
        if initial_misreg is None:
            self.misreg_params = self.xp.zeros(n_params, dtype=self.dtype)
        else:
            self.misreg_params = self.to_xp(initial_misreg, dtype=self.dtype)

        # Internal state
        self.iteration_count = 0
        self.converged = False
        self.last_estimation_time = 0

        # Number of modes
        self.nmodes = len(carrier_frequencies)

        # History for demodulation
        self.slopes_history = []
        self.time_history = []

        # Outputs
        self.estimated_intmat = Intmat(nmodes=self.nmodes, nslopes=0, 
                                       target_device_idx=target_device_idx,
                                       precision=precision)
        self.misreg_output = BaseValue(value=self.misreg_params.copy(),
                                       target_device_idx=target_device_idx,
                                       precision=precision)

        # Inputs
        self.inputs['in_slopes'] = InputValue(type=Slopes)

        # Outputs
        self.outputs['out_intmat'] = self.estimated_intmat
        self.outputs['out_misreg_params'] = self.misreg_output

        self.verbose = True

    def setup(self):
        """Initialize slopes size and nominal IM computation"""
        super().setup()

        # Get initial slopes to determine size
        in_slopes = self.local_inputs['in_slopes']
        if in_slopes is None:
            raise ValueError("in_slopes must be connected before setup")

        # Initialize estimated IM size
        self.estimated_intmat.set_nslopes(len(in_slopes.slopes))

        if self.verbose:
            print(f"SPRINT Estimator initialized:")
            print(f"  WFS index: {self.wfs_index}")
            print(f"  DM index: {self.dm_index}")
            print(f"  Number of modes: {self.nmodes}")
            print(f"  Number of slopes: {self.estimated_intmat.nslopes}")
            print(f"  Estimation interval: {self.t_to_seconds(self.estimation_dt):.2f}s")

    def prepare_trigger(self, t):
        """Collect slopes data for demodulation"""
        super().prepare_trigger(t)

        in_slopes = self.local_inputs['in_slopes']

        # Store history for demodulation (only slopes!)
        self.slopes_history.append(in_slopes.slopes.copy())
        self.time_history.append(t)

    def trigger_code(self):
        """Main SPRINT estimation logic"""
        t = self.current_time

        # Check if it's time to perform estimation
        if (t - self.last_estimation_time) < self.estimation_dt:
            return

        if self.verbose:
            print(f"\n=== SPRINT Estimation at t={self.t_to_seconds(t):.2f}s ===")

        # Step 1: Demodulate slopes to extract measured IM
        im_measured = self._demodulate_slopes()

        if im_measured is None:
            if self.verbose:
                print("  Insufficient data for demodulation, skipping")
            return

        # Step 2: Iterative estimation loop
        self._iterative_estimation(im_measured)

        # Update last estimation time
        self.last_estimation_time = t

        # Update outputs
        self.estimated_intmat.generation_time = t
        self.misreg_output.value = self.misreg_params.copy()
        self.misreg_output.generation_time = t

    def _demodulate_slopes(self):
        """
        Demodulate slopes history to extract interaction matrix.
        
        Returns
        -------
        im_measured : ndarray, shape (nslopes, nmodes)
            Measured interaction matrix from demodulated slopes
        """
        if len(self.slopes_history) < 10:  # Need minimum data
            return None

        # Convert history to array
        slopes_array = self.xp.array(self.slopes_history)  # (nt, nslopes)

        nslopes = slopes_array.shape[1]
        im_measured = self.xp.zeros((nslopes, self.nmodes), dtype=self.dtype)

        dt = self.t_to_seconds(self.simul_params.time_step)
        sampling_freq = 1.0 / dt

        # Demodulate each mode separately
        for mode_idx in range(self.nmodes):
            carrier_freq = float(self.carrier_frequencies[mode_idx])

            # Demodulate each slope
            for slope_idx in range(nslopes):
                slope_signal = slopes_array[:, slope_idx]

                # Use factorized demodulation function
                demod_value, _ = demodulate_signal(
                    signal_data=slope_signal,
                    carrier_freq=carrier_freq,
                    sampling_freq=sampling_freq,
                    cumulated=True,
                    verbose=False,
                    xp_module=self.xp
                )

                im_measured[slope_idx, mode_idx] = demod_value

        # Apply absolute value if requested (like IDL code)
        if self.apply_absolute_slopes:
            im_measured = self.xp.abs(im_measured)

        # Clear history to save memory
        self.slopes_history = []
        self.time_history = []

        if self.verbose:
            print(f"  Demodulated IM shape: {im_measured.shape}")
            print(f"  IM RMS: {float(self.xp.sqrt(self.xp.mean(im_measured**2))):.3e}")

        return im_measured

    def _iterative_estimation(self, im_measured):
        """
        Iterative estimation of mis-registration parameters.
        
        Parameters
        ----------
        im_measured : ndarray
            Measured interaction matrix from demodulated slopes
        """
        if self.verbose:
            print(f"  Starting iterative estimation...")
            print(f"  Initial misreg params: {cpuArray(self.misreg_params)}")

        for iteration in range(self.max_iterations):
            # Compute nominal IM with current mis-registration parameters
            im_nominal = self._compute_nominal_im()

            # Compute optical gains
            G_opt = self._compute_optical_gains(im_measured, im_nominal)

            # Compute sensitivity matrices
            sens_matrices = self._compute_sensitivity_matrices()

            # Compute IM difference (corrected for optical gain)
            im_diff = self._apply_optical_gain_correction(im_measured, G_opt) - im_nominal

            # Estimate mis-registration correction
            delta_misreg = self._estimate_misreg_correction(im_diff, sens_matrices)

            # Update mis-registration parameters
            self.misreg_params += delta_misreg

            # Check convergence
            error = float(self.xp.sqrt(self.xp.mean(im_diff**2)) / 
                         self.xp.sqrt(self.xp.mean(im_measured**2)))

            if self.verbose:
                print(f"    Iteration {iteration + 1}: error={error:.3e}, "
                      f"delta={cpuArray(delta_misreg)}")

            if error < self.convergence_threshold:
                self.converged = True
                if self.verbose:
                    print(f"  Converged after {iteration + 1} iterations!")
                break

        # Final update
        self._update_params_manager()

        # Compute final IM and store
        im_final = self._compute_nominal_im()
        self.estimated_intmat.intmat = im_final

        if self.verbose:
            print(f"  Final misreg params: {cpuArray(self.misreg_params)}")
            print(f"  Final error: {error:.3e}")

    def _compute_nominal_im(self):
        """
        Compute nominal IM using SynIM with current mis-registration parameters.
        
        Returns
        -------
        im_nominal : ndarray
            Nominal interaction matrix
        """
        # Update ParamsManager with current mis-registration
        self._update_params_manager()

        # Compute IM using SynIM
        im_nominal = self.params_manager.compute_interaction_matrix(
            wfs_type='ngs',
            wfs_index=self.wfs_index,
            dm_index=self.dm_index,
            verbose=False,
            display=False
        )

        if self.apply_absolute_slopes:
            im_nominal = self.xp.abs(im_nominal)

        return self.to_xp(im_nominal, dtype=self.dtype)

    def _compute_sensitivity_matrices(self):
        """
        Compute sensitivity matrices for all mis-registration parameters.
        
        Returns
        -------
        sens_matrices : ndarray, shape (nslopes, n_params)
            Sensitivity of slopes to each mis-registration parameter
        """
        n_params = len(self.misreg_params)
        nslopes = self.estimated_intmat.nslopes

        sens_matrices = self.xp.zeros((nslopes, n_params), dtype=self.dtype)

        # Define perturbations for each parameter
        perturbations = {
            0: ([1.0, 0.0], 'shift_x'),    # X shift in pixels
            1: ([0.0, 1.0], 'shift_y'),    # Y shift in pixels
            2: (0.1, 'rotation'),          # Rotation in degrees
            3: (0.99, 'magnification'),    # Magnification factor
        }

        if self.enable_wpup_magn_xy:
            perturbations[4] = ([0.99, 1.0], 'magn_x')
            perturbations[5] = ([1.0, 0.99], 'magn_y')

        # Save original parameters
        original_params = self.misreg_params.copy()

        for param_idx, (perturbation, param_name) in perturbations.items():
            # Compute push matrix (positive perturbation)
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] \
                += self._get_perturbation_value(param_idx, perturbation, push=True)
            im_push = self._compute_nominal_im()

            # Compute pull matrix (negative perturbation)
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] \
                += self._get_perturbation_value(param_idx, perturbation, push=False)
            im_pull = self._compute_nominal_im()

            # Sensitivity = (push - pull) / (2 * perturbation_amplitude)
            perturbation_amp = self._get_perturbation_amplitude(param_idx, perturbation)
            sens_matrices[:, param_idx] = self.xp.mean(
                (im_push - im_pull) / (2.0 * perturbation_amp),
                axis=1
            )

        # Restore original parameters
        self.misreg_params = original_params

        return sens_matrices

    def _get_perturbation_value(self, param_idx, perturbation, push):
        """Get perturbation value for push or pull"""
        if param_idx < 2:  # Shifts
            value = self.xp.sqrt(self.xp.sum(self.xp.array(perturbation)**2))
        elif param_idx == 2:  # Rotation
            value = abs(perturbation)
        else:  # Magnifications
            if isinstance(perturbation, list):
                value = self.xp.sqrt(self.xp.sum((1.0 - self.xp.array(perturbation))**2))
            else:
                value = abs(1.0 - perturbation)

        return value if push else -value

    def _get_perturbation_amplitude(self, param_idx, perturbation):
        """Get total perturbation amplitude"""
        return abs(self._get_perturbation_value(param_idx, perturbation, push=True))

    def _compute_optical_gains(self, im_measured, im_nominal):
        """
        Compute optical gains from measured and nominal IMs.
        
        Returns
        -------
        G_opt : ndarray, shape (nmodes,)
            Optical gain for each mode
        """
        # G_opt = diag(im_measured @ pinv(im_nominal))
        rec_nominal = self.xp.linalg.pinv(im_nominal)
        G_matrix = im_measured @ rec_nominal
        G_opt = self.xp.diag(G_matrix)

        return G_opt

    def _apply_optical_gain_correction(self, im_measured, G_opt):
        """Apply optical gain correction to measured IM"""
        # Correct each mode by its optical gain
        G_opt_inv = 1.0 / (G_opt + 1e-12)  # Avoid division by zero
        im_corrected = im_measured * G_opt_inv[self.xp.newaxis, :]

        return im_corrected

    def _estimate_misreg_correction(self, im_diff, sens_matrices):
        """
        Estimate mis-registration correction from IM difference.
        
        Parameters
        ----------
        im_diff : ndarray, shape (nslopes, nmodes)
            Difference between measured and nominal IM
        sens_matrices : ndarray, shape (nslopes, n_params)
            Sensitivity matrices
        
        Returns
        -------
        delta_misreg : ndarray, shape (n_params,)
            Correction to mis-registration parameters
        """
        # Average over modes
        im_diff_mean = self.xp.mean(im_diff, axis=1)

        # Pseudo-inverse solution: delta = pinv(sens_matrices) @ im_diff_mean
        sens_pinv = self.xp.linalg.pinv(sens_matrices)
        delta_misreg = sens_pinv @ im_diff_mean

        return delta_misreg

    def _update_params_manager(self):
        """Update SynIM ParamsManager with current mis-registration parameters"""
        # Extract parameters
        shift_x = float(self.misreg_params[0])
        shift_y = float(self.misreg_params[1])
        rotation = float(self.misreg_params[2])
        magnification = float(self.misreg_params[3])

        # Update WFS parameters in params_manager
        wfs_key = f'sh_ngs{self.wfs_index + 1}'

        # Get original values
        original_shift_x = self.params_manager.params.get(wfs_key, {}).get('xShiftPhInPixel', 0.0)
        original_shift_y = self.params_manager.params.get(wfs_key, {}).get('yShiftPhInPixel', 0.0)
        original_rotation = self.params_manager.params.get(wfs_key, {}).get('rotAnglePhInDeg', 0.0)

        # Apply corrections
        self.params_manager.params[wfs_key]['xShiftPhInPixel'] = original_shift_x + shift_x
        self.params_manager.params[wfs_key]['yShiftPhInPixel'] = original_shift_y + shift_y
        self.params_manager.params[wfs_key]['rotAnglePhInDeg'] = original_rotation + rotation

        # Update DM magnification
        dm_key = f'dm{self.dm_index + 1}'
        original_magn = self.params_manager.params.get(dm_key, {}).get('magnification', 1.0)
        self.params_manager.params[dm_key]['magnification'] = original_magn + magnification

        # Update pupil magnifications if enabled
        if self.enable_wpup_magn_xy:
            magn_x = float(self.misreg_params[4])
            magn_y = float(self.misreg_params[5])

            if 'pupil_distortion' not in self.params_manager.params:
                self.params_manager.params['pupil_distortion'] = {}

            original_x_stretch = self.params_manager.params.get(
                'pupil_distortion', {}).get('x_stretch', 1.0)
            original_y_stretch = self.params_manager.params.get(
                'pupil_distortion', {}).get('y_stretch', 1.0)

            self.params_manager.params['pupil_distortion']['x_stretch'] = \
                original_x_stretch + magn_x
            self.params_manager.params['pupil_distortion']['y_stretch'] = \
                original_y_stretch + magn_y

    def finalize(self):
        """Save final estimated IM"""
        im_path = os.path.join(self.data_dir, self.im_tag)
        if not im_path.endswith('.fits'):
            im_path += '.fits'

        self.estimated_intmat.save(im_path, overwrite=self.overwrite)

        if self.verbose:
            print(f"\nSPRINT Estimator finalized:")
            print(f"  Converged: {self.converged}")
            print(f"  Final mis-registration parameters: {cpuArray(self.misreg_params)}")
            print(f"  Saved IM to: {im_path}")
