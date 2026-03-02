"""
SPRINT Estimator for Pyramid WFS.
"""

from specula.processing_objects.base_sprint_estimator import BaseSprintEstimator
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.dm import DM
from specula.base_value import BaseValue
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula import cpuArray, np

import matplotlib.pyplot as plt

class SprintPyr(BaseSprintEstimator):
    """
    SPRINT Estimator for Pyramid WFS.
    
    It computes interaction matrices and sensitivity matrices
    for Pyramid wavefront sensors.
    
    Mis-registration parameters:
    - [0]: shift_x (pixels)
    - [1]: shift_y (pixels)
    - [2]: rotation (degrees)
    - [3]: magnification (fractional, added to 1.0)
    
    Anisotropic magnification is not currently implemented in the Pyramid case.
    
    Parameters
    ----------  
    push_amp : float
        Amplitude of the push-pull perturbation for sensitivity matrix estimation (default: 10 nm)
    
    All parameters inherited from BaseSprintEstimator.
    """

    def __init__(self,
                 simul_params,
                 dm,
                 slopec,
                 source,
                 wfs,
                 modes_index,
                 carrier_frequencies,
                 pupil_mask: Pupilstop = None,
                 push_amp=10,
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
        Initialize Pyramid SPRINT estimator.
        """

        # Number of parameters for this WFS type
        n_params = 4

        # Call parent constructor with n_params
        super().__init__(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            modes_index=modes_index,
            carrier_frequencies=carrier_frequencies,
            pupil_mask=pupil_mask,
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

        if self.source.polar_coordinates[0] != 0.0:
            raise ValueError("SprintPyr currently only supports on-axis sources")

        self.pyr_params = {}
        self.internal_wfs = None
        self.internal_command = None
        self.internal_dm = None
        self.internal_pixels = None
        self.internal_slopec = None
        self.internal_ef = None
        self.push_amp = push_amp

        self.idx_valid_sa = None

        # Define perturbations
        self.perturbations = {
            0: (1.0, 'shift_x'),
            1: (1.0, 'shift_y'),
            2: (0.1, 'rotation'),
            3: (0.01, 'magnification')
        }

    def _validate_wfs(self):
        """Validate that WFS is Pyramid"""
        if not isinstance(self.wfs, ModulatedPyramid):
            raise ValueError(f"SprintEstimator requires Pyramid WFS, got {type(self.wfs)}")

        if not isinstance(self.slopec, PyrSlopec):
            raise ValueError(f"SprintEstimator requires PyrSlopec, got {type(self.slopec)}")

    def setup(self):
        """Initialize with Pyramid-specific parameters and build internal pipeline"""
        super().setup()

        # 1. Build Internal DM (sharing IFuncs with the main DM)
        self.internal_dm = DM(
            simul_params=self.simul_params,
            height=getattr(self.dm, 'height', 0.0),
            ifunc=self.dm.ifunc_obj,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.internal_command = BaseValue(self.xp.zeros(self.dm.nmodes, dtype=self.dtype),
                                          target_device_idx=self.target_device_idx)
        self.internal_dm.inputs['in_command'].set(self.internal_command)
        self.internal_dm.setup()

        # 2. Build Internal Pyramid WFS
        self._build_pyr_params()
        self.internal_wfs = ModulatedPyramid(**self.pyr_params,
                                             target_device_idx=self.target_device_idx,
                                             precision=self.precision)

        # Create an ElectricField object to link DM and WFS
        self.internal_ef = ElectricField(
            self.simul_params.pixel_pupil,
            self.simul_params.pixel_pupil,
            pixel_pitch=self.simul_params.pixel_pitch,
            S0=1e10,  # High flux for accurate IM estimation
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        self.internal_ef.A = self.pupil_mask
        self.internal_wfs.inputs['in_ef'].set(self.internal_ef)
        self.internal_wfs.setup()

        # Bulid Pixels object to interface WFS and Slopec
        ccd_side = self.internal_wfs.final_ccd_side
        self.internal_pixels = Pixels(dimx=ccd_side, dimy=ccd_side,
                                      target_device_idx=self.target_device_idx,
                                      precision=self.precision)

        # 3. Build Internal Slopec (sharing PupData with the main Slopec)
        self.internal_slopec = PyrSlopec(
            pupdata=self.slopec.pupdata,
            shlike=self.slopec.shlike,
            norm_factor=self.slopec.norm_factor,
            slopes_from_intensity=self.slopec.slopes_from_intensity,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )
        # Link WFS output (Intensity) to Slopec input (Pixels)
        self.internal_slopec.inputs['in_pixels'].set(self.internal_pixels)
        self.internal_slopec.setup()

        # Extract valid subapertures for display/logging
        pupil_idx = self.slopec.pupdata.pupil_idx
        self.idx_valid_sa = self.xp.concatenate([pupil_idx(i)[pupil_idx(i) >= 0] for i in range(4)])

        if self.verbose: # pragma: no cover
            print(f"  WFS type: Pyramid")
            print(f"  FOV: {self.wfs.fov:.2f} arcsec")
            print(f"  Valid subapertures mapping size: {len(self.idx_valid_sa)}")
            print(f"  Number of misreg params: {self.n_params}")

    def _build_pyr_params(self):
        """Set up parameters for the internal Pyramid simulator based on the WFS configuration."""
        wfs = self.wfs
        self.pyr_params.clear()

        # Use existing properties from the original WFS
        self.pyr_params['simul_params'] = self.simul_params
        self.pyr_params['wavelengthInNm'] = wfs.wavelength_in_nm
        self.pyr_params['fov'] = wfs.fov
        self.pyr_params['pup_diam'] = wfs.pup_diam
        self.pyr_params['output_resolution'] = wfs.final_ccd_side
        self.pyr_params['mod_amp'] = getattr(wfs, 'mod_amp', 3.0)
        self.pyr_params['mod_step'] = getattr(wfs, 'mod_steps', None)
        self.pyr_params['mod_type'] = getattr(wfs, 'mod_type', 'circular')

        # We need to force extrapolation for the internal WFS to initialize the interpolator,
        # even if the starting mis-registration parameters would not require it.
        self.pyr_params['force_extrapolation'] = True

        # Current mis-registration parameters (initially from input or zeros)
        self.pyr_params['xShiftPhInPixel'] = float(self.misreg_params[0])
        self.pyr_params['yShiftPhInPixel'] = float(self.misreg_params[1])
        self.pyr_params['rotAnglePhInDeg'] = float(self.misreg_params[2])
        self.pyr_params['magnification'] = 1.0 + float(self.misreg_params[3])

    def _compute_nominal_im(self):
        """Compute nominal IM using a push-pull sequence on the internal pipeline."""

        # 1. Update mis-registration parameters on the interpolator
        self.internal_wfs.ef_interpolator.update_parameters(
                            xShiftPhInPixel=float(self.misreg_params[0]),
                            yShiftPhInPixel=float(self.misreg_params[1]),
                            rotAnglePhInDeg=float(self.misreg_params[2]),
                            magnification=1.0 + float(self.misreg_params[3])
                            )

        im_nominal = self.xp.zeros((int(self.internal_slopec.nslopes()), self.nmodes),
                                   dtype=self.dtype)
        step = 1
        time_step = 0.001

        # Push-pull for each mode in modes_index
        for i, mode_idx in enumerate(self.modes_index):
            for sign in [1, -1]:
                # time
                current_time = self.internal_wfs.seconds_to_t(step * time_step)

                # A. Apply push or pull command to the internal DM
                cmd = self.xp.zeros(self.dm.nmodes, dtype=self.dtype)
                cmd[mode_idx] = sign * self.push_amp
                self.internal_command.set_value(cmd)
                self.internal_command.generation_time = current_time
                self.internal_dm.check_ready(current_time)
                self.internal_dm.trigger_code()

                # B. Convert DM shape to electric field and feed to the internal WFS
                self.internal_ef.phaseInNm = self.internal_dm.outputs['out_layer'].phaseInNm
                self.internal_ef.generation_time = current_time

                # C. Propagate through the internal WFS to get the intensity pattern
                self.internal_wfs.check_ready(current_time)
                self.internal_wfs.trigger_code()
                self.internal_wfs.post_trigger()

                # C2. Transfer the calculated data to the Pixels object of the Slopec
                # Extract the calculated intensity array (.i) and pass it to the pixels (.pixels)
                intensity = self.internal_wfs.outputs['out_i'].i
                # Normalize and scale to 12-bit range
                intensity_norm = intensity / intensity.max() * 2**12
                self.internal_pixels.pixels[:] = intensity_norm
                self.internal_pixels.generation_time = current_time

                # D. Calculate the slopes
                self.internal_slopec.check_ready(current_time)
                self.internal_slopec.trigger_code()
                self.internal_slopec.post_trigger()

                # Save the result of the push or pull
                if sign == 1:
                    slopes_push = self.internal_slopec.outputs['out_slopes'].slopes.copy()
                else:
                    slopes_pull = self.internal_slopec.outputs['out_slopes'].slopes.copy()

                step += 1

            # Interaction matrix: discrete derivative using self.push_amp
            im_nominal[:, i] = (slopes_push - slopes_pull) / (2.0 * self.push_amp)

        return im_nominal

    def _im_2d_map(self, im_mode): # pragma: no cover
        """Convert interaction matrix to 2D map for visualization (Pyramid)."""

        # Create a temporary Slopes object for the IM mode
        sl = Slopes(length=im_mode.shape[0],
                    target_device_idx=self.target_device_idx,
                    precision=self.precision)
        sl.set_value(im_mode)

        pupdata = self.internal_slopec.pupdata

        if self.internal_slopec.slopes_from_intensity:
            sl.single_mask = pupdata.complete_mask()
            sl.display_map = pupdata.display_map
        else:
            # For Pyramid, we need to reconstruct the single_mask for the full frame based
            # on the pupil data.
            # The original single_mask from pupdata is cropped to the valid subapertures,
            # which is not suitable for visualizing the full 2D map of the IM mode.
            full_mask = self.xp.zeros(pupdata.framesize, dtype=self.dtype)

            # Extract the valid subaperture indices for each of the 4 pupils and mark
            # them on the full mask
            idx0 = pupdata.pupil_idx(0)
            idx0_valid = idx0[idx0 >= 0]

            # Write 1s in the full mask at the positions corresponding to the valid
            # subapertures of pupil 0
            self.xp.put(full_mask, idx0_valid, 1)

            sl.single_mask = full_mask
            sl.display_map = idx0_valid

        # get2d() handles the reshaping based on the lengths of mask/display_map.
        # It returns a single 2D array if slopes_from_intensity=True,
        # or a list of two 2D arrays [frame_x, frame_y] for standard slopes.
        frames2d = sl.get2d()

        # Handle both single frame and separate x/y frames cases.
        if isinstance(frames2d, list):
            frames2d = [cpuArray(frame) for frame in frames2d]
        else:
            frames2d = cpuArray(frames2d)
            if frames2d.ndim == 3 and frames2d.shape[0] == 2:
                n = frames2d.shape[2]
                combined = np.zeros((n,n//2))
                combined[:n//2,:] = frames2d[0][:n//2, :n//2]
                combined[n//2:,:] = frames2d[1][:n//2, :n//2]
                frames2d = [combined]
            else:
                frames2d = [frames2d]

        return frames2d

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
