import os

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
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.sh import SH
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula import xp, cpuArray, np

# Import SynIM for sensitivity matrix computation
try:
    import synim.synim as synim
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
    
    This is a SCAO-specific implementation that automatically extracts all
    necessary parameters from connected SPECULA objects.
    
    Parameters (passed via YAML with _ref suffix)
    ----------------------------------------------
    dm : DM
        Deformable mirror object for parameter extraction
    slopec : Slopec
        Slope computer for valid subaperture extraction
    source : Source
        Source object for coordinate information
    wfs : BaseProcessingObj (SH or ModulatedPyramid)
        WFS object for geometry extraction
    
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
                 dm: DM,
                 slopec: Slopec,
                 source: Source,
                 wfs: BaseProcessingObj,  # SH or ModulatedPyramid
                 carrier_frequencies: list = None,
                 estimation_dt: float = 10.0,
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
        self.dm = dm
        self.slopec = slopec
        self.source = source
        self.wfs = wfs

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

        # Valid subaperture indices (will be extracted in setup)
        self.idx_valid_sa = None
        
        # Extract pupil mask and diameter
        self.pup_diam_m = self.simul_params.pixel_pupil * self.simul_params.pixel_pitch
        # Will load pupil mask in setup

        # Outputs
        self.estimated_intmat = Intmat(nmodes=self.nmodes, nslopes=0,
                                       target_device_idx=target_device_idx,
                                       precision=precision)
        self.misreg_output = BaseValue(value=self.misreg_params.copy(),
                                       target_device_idx=target_device_idx,
                                       precision=precision)

        # Input connection (only slopes)
        self.inputs['in_slopes'] = InputValue(type=Slopes)

        # Outputs
        self.outputs['out_intmat'] = self.estimated_intmat
        self.outputs['out_misreg_params'] = self.misreg_output

        self.verbose = True

    def setup(self):
        """Initialize slopes size and extract parameters from connected objects"""
        super().setup()

        # Get initial slopes to determine size
        in_slopes = self.local_inputs['in_slopes']
        if in_slopes is None:
            raise ValueError("in_slopes must be connected before setup")

        # Initialize estimated IM size
        self.estimated_intmat.set_nslopes(len(in_slopes.slopes))

        # Extract valid subaperture indices
        self._extract_valid_subapertures()
        
        # Extract pupil mask from DM
        self.pup_mask = cpuArray(self.dm.mask)

        if self.verbose:
            print(f"SPRINT Estimator initialized:")
            print(f"  WFS type: {self._get_wfs_type()}")
            print(f"  Number of modes: {self.nmodes}")
            print(f"  Number of slopes: {self.estimated_intmat.nslopes}")
            print(f"  Valid subapertures: {self.idx_valid_sa.shape[0] if self.idx_valid_sa is not None else 'Unknown'}")
            print(f"  Estimation interval: {self.t_to_seconds(self.estimation_dt):.2f}s")
            print(f"  Source coordinates: {self.source.polar_coordinates}")

    def _extract_valid_subapertures(self):
        """Extract valid subaperture indices from slopec"""

        if isinstance(self.slopec, ShSlopec):
            # Shack-Hartmann case
            subapdata = self.slopec.subapdata
            # Convert display_map to (i,j) indices
            display_map = cpuArray(subapdata.display_map)
            nx = subapdata.nx
            idx_i = display_map // nx
            idx_j = display_map % nx
            self.idx_valid_sa = np.column_stack((idx_i, idx_j))

        elif isinstance(self.slopec, PyrSlopec):
            # Pyramid case
            pupdata = self.slopec.pupdata

            if self.slopec.slopes_from_intensity:
                # For intensity mode, use complete mask
                complete_mask = cpuArray(pupdata.complete_mask())
                idx_i, idx_j = np.where(complete_mask > 0)
            else:
                # For slopes mode, use single mask
                single_mask = cpuArray(pupdata.single_mask())
                idx_i, idx_j = np.where(single_mask > 0)

            self.idx_valid_sa = np.column_stack((idx_i, idx_j))
        else:
            raise ValueError(f"Unknown slopec type: {type(self.slopec)}")

    def _get_wfs_type(self):
        """Determine WFS type from connected object"""
        if isinstance(self.wfs, SH):
            return 'sh'
        elif isinstance(self.wfs, ModulatedPyramid):
            return 'pyramid'
        else:
            raise ValueError(f"Unknown WFS type: {type(self.wfs)}")

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
            print(f"\n{'='*60}")
            print(f"SPRINT Estimation at t={self.t_to_seconds(t):.2f}s")
            print(f"{'='*60}")

        # Step 1: Demodulate slopes to extract measured IM
        im_measured = self._demodulate_slopes()

        if im_measured is None:
            if self.verbose:
                print("  Not enough data for demodulation yet")
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
        slopes_array = self.xp.stack(self.slopes_history)  # (nt, nslopes)

        nslopes = slopes_array.shape[1]
        im_measured = self.xp.zeros((nslopes, self.nmodes), dtype=self.dtype)

        dt = self.t_to_seconds(self.simul_params.time_step)
        sampling_freq = 1.0 / dt

        # Demodulate each mode separately
        for mode_idx in range(self.nmodes):
            carrier_freq = float(self.carrier_frequencies[mode_idx])

            # Demodulate each slope using vectorized operations
            for slope_idx in range(nslopes):
                signal = cpuArray(slopes_array[:, slope_idx])

                # Use SPECULA's demodulate_signal function
                amplitude, _ = demodulate_signal(
                    signal,
                    carrier_freq,
                    sampling_freq
                )

                im_measured[slope_idx, mode_idx] = self.to_xp(amplitude)

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
            print(f"\n  Starting iterative estimation...")
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
                print(f"    Iteration {iteration+1}: error = {error:.3e}, "
                      f"delta = {cpuArray(delta_misreg)}")

            if error < self.convergence_threshold:
                if self.verbose:
                    print(f"  Converged after {iteration+1} iterations!")
                break

        # Compute final IM and store
        im_final = self._compute_nominal_im()
        self.estimated_intmat.intmat = im_final

        if self.verbose:
            print(f"\n  Final misreg params: {cpuArray(self.misreg_params)}")
            print(f"  Final error: {error:.3e}")

    def _compute_nominal_im(self):
        """
        Compute nominal IM using SynIM with current mis-registration parameters.
        
        Returns
        -------
        im_nominal : ndarray
            Nominal interaction matrix
        """
        # Extract current mis-registration parameters
        shift_x = float(self.misreg_params[0])
        shift_y = float(self.misreg_params[1])
        rotation = float(self.misreg_params[2])
        magnification = 1.0 + float(self.misreg_params[3])

        # Get WFS parameters
        if isinstance(self.wfs, SH):
            wfs_nsubaps = self.wfs.subap_on_diameter
            wfs_fov_arcsec = self.wfs.subap_wanted_fov
        else:  # ModulatedPyramid
            wfs_nsubaps = self.wfs.pup_diam
            wfs_fov_arcsec = self.wfs.fov

        # Get source coordinates
        gs_pol_coo = tuple(cpuArray(self.source.polar_coordinates))
        gs_height = self.source.height if self.source.height != float('inf') else float('inf')

        # Compute IM using SynIM directly
        im_nominal = synim.interaction_matrix(
            pup_diam_m=self.pup_diam_m,
            pup_mask=self.pup_mask,
            dm_array=cpuArray(self.dm.ifunc).transpose(1, 0, 2),  # SynIM convention
            dm_mask=cpuArray(self.dm.mask).T,  # SynIM convention
            dm_height=0.0,  # Ground DM
            dm_rotation=0.0,  # DM rotation (if any)
            gs_pol_coo=gs_pol_coo,
            gs_height=gs_height,
            wfs_nsubaps=wfs_nsubaps,
            wfs_rotation=rotation,  # Apply estimated rotation
            wfs_translation=(shift_x, shift_y),  # Apply estimated shift
            wfs_mag_global=magnification,  # Apply estimated magnification
            wfs_fov_arcsec=wfs_fov_arcsec,
            idx_valid_sa=self.idx_valid_sa,
            verbose=False,
            specula_convention=True
        )

        if self.apply_absolute_slopes:
            im_nominal = self.xp.abs(self.to_xp(im_nominal))
        else:
            im_nominal = self.to_xp(im_nominal)

        return im_nominal.astype(self.dtype)

    def _compute_sensitivity_matrices(self):
        """
        Compute sensitivity matrices for all mis-registration parameters.
        
        Returns
        -------
        sens_matrices : ndarray, shape (nslopes, nmodes, n_params)
            Sensitivity of each slope/mode to each mis-registration parameter
        """
        n_params = len(self.misreg_params)
        nslopes = self.estimated_intmat.nslopes

        # Store per-mode sensitivities: (nslopes, nmodes, n_params)
        sens_matrices = self.xp.zeros((nslopes, self.nmodes, n_params), dtype=self.dtype)

        # Define perturbations for each parameter
        perturbations = {
            0: (1.0, 'shift_x'),          # X shift in pixels
            1: (1.0, 'shift_y'),          # Y shift in pixels
            2: (0.1, 'rotation'),         # Rotation in degrees
            3: (0.01, 'magnification'),   # Magnification factor (1% change)
        }

        if self.enable_wpup_magn_xy:
            perturbations[4] = (0.01, 'magn_x')  # X magnification
            perturbations[5] = (0.01, 'magn_y')  # Y magnification

        # Save original parameters
        original_params = self.misreg_params.copy()

        for param_idx, (perturbation, param_name) in perturbations.items():
            # Compute push matrix (positive perturbation)
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] += perturbation
            im_push = self._compute_nominal_im()

            # Compute pull matrix (negative perturbation)
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] -= perturbation
            im_pull = self._compute_nominal_im()

            # Sensitivity = (push - pull) / (2 * perturbation)
            sens_matrices[:, :, param_idx] = (im_push - im_pull) / (2.0 * perturbation)

        # Restore original parameters
        self.misreg_params = original_params

        return sens_matrices

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
        sens_matrices : ndarray, shape (nslopes, nmodes, n_params)
            Sensitivity matrices
        
        Returns
        -------
        delta_misreg : ndarray, shape (n_params,)
            Correction to mis-registration parameters
        """
        n_params = sens_matrices.shape[2]
        delta_misreg = self.xp.zeros(n_params, dtype=self.dtype)

        # For each parameter, solve: sens[:,:,p] @ delta_p = im_diff
        # Average solution over all modes
        for param_idx in range(n_params):
            # Extract sensitivity for this parameter (nslopes, nmodes)
            sens_p = sens_matrices[:, :, param_idx]

            # Solve for each mode separately, then average
            deltas = []
            for mode_idx in range(self.nmodes):
                # sens_p[:, mode_idx] @ delta = im_diff[:, mode_idx]
                sens_col = sens_p[:, mode_idx]
                diff_col = im_diff[:, mode_idx]

                # Least squares solution
                delta = self.xp.dot(sens_col, diff_col) / (self.xp.dot(sens_col, sens_col) + 1e-12)
                deltas.append(delta)

            # Average over modes
            delta_misreg[param_idx] = self.xp.mean(self.xp.array(deltas))

        return delta_misreg

    def finalize(self):
        """Save final estimated IM"""
        im_path = os.path.join(self.data_dir, self.im_tag)
        if not im_path.endswith('.fits'):
            im_path += '.fits'

        self.estimated_intmat.save(im_path, overwrite=self.overwrite)

        if self.verbose:
            print(f"\nSaved estimated interaction matrix to: {im_path}")
            print(f"Final mis-registration parameters:")
            print(f"  X shift: {float(self.misreg_params[0]):.3f} pixels")
            print(f"  Y shift: {float(self.misreg_params[1]):.3f} pixels")
            print(f"  Rotation: {float(self.misreg_params[2]):.3f} degrees")
            print(f"  Magnification: {float(self.misreg_params[3]):.6f}")
            if self.enable_wpup_magn_xy:
                print(f"  X magnification: {float(self.misreg_params[4]):.6f}")
                print(f"  Y magnification: {float(self.misreg_params[5]):.6f}")
