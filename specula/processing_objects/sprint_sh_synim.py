"""
SPRINT Estimator for Shack-Hartmann WFS using SynIM for IM computation.
"""

from specula.lib.synim_utils import compute_im_synim
from specula.data_objects.slopes import Slopes
from specula.processing_objects.base_sprint_estimator import BaseSprintEstimator
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
from specula import cpuArray, np

import matplotlib.pyplot as plt

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
                 integration_gain=0.5,
                 forgetting_factor=1.0,
                 verbose: bool = False,
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
            verbose=verbose,
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

        if self.verbose: # pragma: no cover
            print(f"  WFS type: Shack-Hartmann")
            print(f"  Subapertures: {self.wfs.subap_on_diameter}x{self.wfs.subap_on_diameter}")
            print(f"  Valid subapertures: {len(self.idx_valid_sa)}")
            print(f"  FOV: {self.wfs.subap_wanted_fov:.2f} arcsec")
            print(f"  Number of misreg params: {self.n_params}")
            if self.enable_wpup_magn_xy:
                print(f"  Using separate X/Y magnification")

    def _compute_nominal_im(self):
        """Compute nominal IM using SynIM"""
        im_nominal = compute_im_synim(
            misreg_params=self.misreg_params,
            pup_diam_m=self.pup_diam_m,
            pup_mask=self.pup_mask,
            ifunc_3d=self.ifunc_3d,
            dm_mask=self.dm.mask,
            source_polar_coords=self.source.polar_coordinates,
            source_height=self.source.height,
            wfs_nsubaps=self.wfs.subap_on_diameter,
            wfs_fov_arcsec=self.wfs.subap_wanted_fov,
            idx_valid_sa=self.idx_valid_sa,
            apply_absolute_slopes=self.apply_absolute_slopes,
            verbose=self.verbose
        )

        return self.to_xp(im_nominal, dtype=self.dtype)

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

    def _im_2d_map(self, im_mode): # pragma: no cover
        """Convert interaction matrix to 2D map for visualization."""
        # Load subapdata
        if isinstance(self.slopec, ShSlopec):
            # Shack-Hartmann case
            subapdata = self.slopec.subapdata

            # Create Slopes object for IM mode
            sl = Slopes(length=im_mode.shape[0])
            sl.set_value(im_mode)
            sl.single_mask = subapdata.single_mask()
            sl.display_map = subapdata.display_map

            # Create 2D frames
            frame_x = sl.xp.zeros_like(sl.single_mask, dtype=sl.dtype)
            frame_y = sl.xp.zeros_like(sl.single_mask, dtype=sl.dtype)

            # Remap to 2D
            sl.x_remap2d(frame_x, sl.display_map)
            sl.y_remap2d(frame_y, sl.display_map)
        else:
            raise NotImplementedError("2D IM map is only implemented for"
                                      "Shack-Hartmann WFS in this example")

        return frame_x, frame_y

    def _plot_debug_info(self, im_measured, im_nominal,
                         im_diff, G_opt, iteration): # pragma: no cover
        """Plot debug information for SH WFS."""

        print(f"G_opt: {cpuArray(G_opt)}")
        print(f"Mis-reg parameters: {cpuArray(self.misreg_params)}")

        plt.figure(figsize=(12, 5))
        plt.plot(im_measured[:,0]/G_opt[0], label='Measured IM (demodulated)')
        plt.plot(im_nominal[:,0], label='Nominal IM (current params)')
        plt.plot(im_diff[:,0], label='IM Difference (corrected)')
        plt.legend()
        plt.title(f"Iteration {iteration+1}")
        plt.xlabel("Slope index")
        plt.ylabel("Amplitude")
        plt.grid()

        #2D plot of IM
        im_2d_measured = self._im_2d_map(im_measured[:,0]/G_opt[0])
        im_2d_nominal = self._im_2d_map(im_nominal[:,0])
        im_2d_diff = self._im_2d_map(im_diff[:,0])
        # Calculate common colorbar limits
        all_data = [
            cpuArray(im_2d_measured[0]),
            cpuArray(im_2d_nominal[0]),
            cpuArray(im_2d_diff[0])
        ]
        vmin = min(data.min() for data in all_data)
        vmax = max(data.max() for data in all_data)

        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.title("Measured IM (demodulated)")
        plt.imshow(cpuArray(im_2d_measured[0]), cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.subplot(1,3,2)
        plt.title("Nominal IM (current params)")
        plt.imshow(cpuArray(im_2d_nominal[0]), cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.subplot(1,3,3)
        plt.title("IM Difference (corrected)")
        plt.imshow(cpuArray(im_2d_diff[0]), cmap='viridis', vmin=vmin, vmax=vmax)
        plt.colorbar()

        plt.tight_layout()
        plt.show()
