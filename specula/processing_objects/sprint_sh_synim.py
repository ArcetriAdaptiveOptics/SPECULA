"""
SPRINT Estimator for Shack-Hartmann WFS using SynIM for IM computation.
"""

import synim.synim as synim
from specula.processing_objects.base_sprint_estimator import BaseSprintEstimator
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula import cpuArray, np


class SprintShSynim(BaseSprintEstimator):
    """
    SPRINT Estimator for Shack-Hartmann WFS.
    
    Uses SynIM library to compute interaction matrices and sensitivity matrices
    for Shack-Hartmann wavefront sensors.
    
    Mis-registration parameters:
    - [0]: shift_x (pixels)
    - [1]: shift_y (pixels)
    - [2]: rotation (degrees)
    - [3]: magnification (fractional, added to 1.0)
    
    If enable_wpup_magn_xy=True (not yet implemented in SynIM):
    - [4]: magn_x (fractional)
    - [5]: magn_y (fractional)
    
    Parameters
    ----------
    enable_wpup_magn_xy : bool
        Enable separate X/Y magnification parameters (default: False)
    
    All other parameters inherited from BaseSprintEstimator.
    """

    def __init__(self,
                 simul_params,
                 dm,
                 slopec,
                 source,
                 wfs,
                 modes_index,
                 carrier_frequencies,
                 enable_wpup_magn_xy=False,
                 estimation_dt=10.0,
                 max_iterations=10,
                 convergence_threshold=1e-3,
                 initial_misreg=None,
                 apply_absolute_slopes=False,
                 integration_gain=0.9,
                 forgetting_factor=1.0,
                 target_device_idx=None,
                 precision=None):
        """
        Initialize SH SPRINT estimator with SynIM backend.
        
        Parameters
        ----------
        enable_wpup_magn_xy : bool
            Enable separate X/Y magnification (future feature)
        """
        # Store before calling super().__init__
        self.enable_wpup_magn_xy = enable_wpup_magn_xy

        # Calculate number of parameters for this WFS type
        n_params = 6 if enable_wpup_magn_xy else 4

        # Call parent constructor with n_params
        super().__init__(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            modes_index=modes_index,
            carrier_frequencies=carrier_frequencies,
            n_params=n_params,
            estimation_dt=estimation_dt,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            initial_misreg=initial_misreg,
            apply_absolute_slopes=apply_absolute_slopes,
            integration_gain=integration_gain,
            forgetting_factor=forgetting_factor,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.idx_valid_sa = None

    def _validate_wfs(self):
        """Validate that WFS is Shack-Hartmann"""
        if not isinstance(self.wfs, SH):
            raise ValueError(f"SprintEstimator requires SH WFS, got {type(self.wfs)}")

        if not isinstance(self.slopec, ShSlopec):
            raise ValueError(f"SprintEstimator requires ShSlopec, got {type(self.slopec)}")

    def setup(self):
        """Initialize with SH-specific parameters"""
        super().setup()

        # Extract valid subapertures from ShSlopec
        subapdata = self.slopec.subapdata
        display_map = cpuArray(subapdata.display_map)
        nx = subapdata.nx
        idx_i = display_map // nx
        idx_j = display_map % nx
        self.idx_valid_sa = np.column_stack((idx_i, idx_j))

        if self.verbose:
            print(f"  WFS type: Shack-Hartmann")
            print(f"  Subapertures: {self.wfs.subap_on_diameter}x{self.wfs.subap_on_diameter}")
            print(f"  Valid subapertures: {len(self.idx_valid_sa)}")
            print(f"  FOV: {self.wfs.subap_wanted_fov:.2f} arcsec")
            print(f"  Number of misreg params: {self.n_params}")
            if self.enable_wpup_magn_xy:
                print(f"  Using separate X/Y magnification")

    def _compute_nominal_im(self):
        """Compute nominal IM using SynIM"""
        # Extract current mis-registration
        shift_x = float(self.misreg_params[0])
        shift_y = float(self.misreg_params[1])
        rotation = float(self.misreg_params[2])
        magnification = 1.0 + float(self.misreg_params[3])

        # Get source parameters
        gs_pol_coo = tuple(cpuArray(self.source.polar_coordinates))
        gs_height = self.source.height if self.source.height != float('inf') else float('inf')

        # Compute IM with SynIM
        im_nominal = synim.interaction_matrix(
            pup_diam_m=self.pup_diam_m,
            pup_mask=self.pup_mask,
            dm_array=cpuArray(self.ifunc_3d),
            dm_mask=cpuArray(self.dm.mask).T,
            dm_height=0.0,
            dm_rotation=0.0,
            gs_pol_coo=gs_pol_coo,
            gs_height=gs_height,
            wfs_nsubaps=self.wfs.subap_on_diameter,
            wfs_rotation=rotation,
            wfs_translation=(shift_x, shift_y),
            wfs_mag_global=magnification,
            wfs_fov_arcsec=self.wfs.subap_wanted_fov,
            idx_valid_sa=self.idx_valid_sa,
            verbose=False,
            specula_convention=True
        )

        im_nominal = self.to_xp(im_nominal, dtype=self.dtype)

        if self.apply_absolute_slopes:
            im_nominal = self.xp.abs(im_nominal)

        return im_nominal

    def _compute_sensitivity_matrices(self):
        """Compute sensitivity matrices using mis-registration push-pull"""
        n_params = len(self.misreg_params)
        nslopes = self.estimated_intmat.nslopes

        sens_matrices = self.xp.zeros((nslopes, self.nmodes, n_params), dtype=self.dtype)

        # Define perturbations
        perturbations = {
            0: (1.0, 'shift_x'),
            1: (1.0, 'shift_y'),
            2: (0.1, 'rotation'),
            3: (0.01, 'magnification'),
        }

        if self.enable_wpup_magn_xy:
            perturbations[4] = (0.01, 'magn_x')
            perturbations[5] = (0.01, 'magn_y')

        original_params = self.misreg_params.copy()

        for param_idx, (delta, name) in perturbations.items():
            # Push
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] += delta
            im_push = self._compute_nominal_im()

            # Pull
            self.misreg_params = original_params.copy()
            self.misreg_params[param_idx] -= delta
            im_pull = self._compute_nominal_im()

            # Sensitivity
            sens_matrices[:, :, param_idx] = (im_push - im_pull) / (2.0 * delta)

        self.misreg_params = original_params

        return sens_matrices
