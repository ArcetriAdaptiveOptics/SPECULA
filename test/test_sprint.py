import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula.data_objects.simul_params import SimulParams
from specula.data_objects.source import Source
from specula.data_objects.ifunc import IFunc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.ccd import CCD
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.sprint_sh_synim import SprintShSynim
import synim.synim as synim

from test.specula_testlib import cpu_and_gpu


def create_test_system():
    """Create a minimal SCAO system for testing SPRINT"""

    # Simulation parameters
    simul_params = SimulParams(
        time_step=1e-3,  # 1ms
        pixel_pupil=160,
        pixel_pitch=8.0/160  # 8m telescope
    )

    # Source (on-axis NGS)
    source = Source(
        polar_coordinates=[0.0, 0.0],  # On-axis
        magnitude=8.0,
        wavelengthInNm=500.0,
        target_device_idx=-1,
        precision=1
    )

    # DM - Zernike modes
    n_modes = 50
    ifunc = IFunc(
        type_str='zernike',
        npixels=simul_params.pixel_pupil,
        nmodes=n_modes,
        obsratio=0.0,
        target_device_idx=-1,
        precision=1
    )

    dm = DM(
        simul_params=simul_params,
        ifunc=ifunc,
        height=0.0,
        target_device_idx=-1,
        precision=1
    )

    # Create SubapData (valid subapertures) - based on test_sh_slopec.py
    subap_on_diameter = 8
    subap_npx = 12
    n_subap = subap_on_diameter

    mask_subap = np.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))
    idxs = {}
    map_dict = {}

    count = 0
    for i in range(subap_on_diameter):
        for j in range(subap_on_diameter):
            mask_subap *= 0
            mask_subap[i*subap_npx:(i+1)*subap_npx, j*subap_npx:(j+1)*subap_npx] = 1
            idxs[count] = np.where(mask_subap == 1)
            map_dict[count] = j * subap_on_diameter + i
            count += 1

    v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
    m = np.zeros(len(idxs), dtype=int)
    for k, idx in idxs.items():
        v[k] = np.ravel_multi_index(idx, mask_subap.shape)
        m[k] = map_dict[k]

    subapdata = SubapData(
        idxs=v,
        display_map=m,
        nx=n_subap,
        ny=n_subap,
        target_device_idx=-1,
        precision=1
    )

    # SH WFS - based on test_sh.py
    pxscale_arcsec = 0.5
    wfs = SH(
        wavelengthInNm=500,
        subap_wanted_fov=subap_npx * pxscale_arcsec,
        sensor_pxscale=pxscale_arcsec,
        subap_on_diameter=subap_on_diameter,
        subap_npx=subap_npx,
        target_device_idx=-1,
        precision=1
    )

    # Create input electric field for WFS setup
    ef = ElectricField(
        simul_params.pixel_pupil,
        simul_params.pixel_pupil,
        simul_params.pixel_pitch,
        S0=1,
        target_device_idx=-1,
        precision=1
    )
    ef.generation_time = 1

    # Connect and setup WFS
    wfs.inputs['in_ef'].set(ef)
    wfs.setup()

    # CCD detector - based on test_ccd.py
    ccd_size = subap_on_diameter * subap_npx
    ccd = CCD(
        simul_params=simul_params,
        size=(ccd_size, ccd_size),
        dt=simul_params.time_step,  # Same as simulation time step
        bandw=300.0,
        photon_noise=False,  # Start without noise for testing
        readout_noise=False,
        target_device_idx=-1,
        precision=1
    )

    # Connect CCD to WFS
    ccd.inputs['in_i'].set(wfs.outputs['out_i'])
    ccd.setup()

    # Slopec - based on test_sh_slopec.py
    slopec = ShSlopec(
        subapdata=subapdata,
        target_device_idx=-1,
        precision=1
    )

    # Connect slopec to CCD pixels
    slopec.inputs['in_pixels'].set(ccd.outputs['out_pixels'])
    slopec.setup()

    return simul_params, source, dm, wfs, ccd, slopec


def generate_reference_im(simul_params, source, dm, wfs, slopec):
    """Generate reference IM using SynIM (no mis-registration)"""

    # Extract parameters
    pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
    pup_mask = np.array(dm.mask)

    # Get 3D influence functions
    ifunc_3d = np.array(dm._ifunc.ifunc_2d_to_3d(normalize=True))

    # Get valid subapertures from slopec
    subapdata = slopec.subapdata
    display_map = np.array(subapdata.display_map)
    nx = subapdata.nx
    idx_i = display_map // nx
    idx_j = display_map % nx
    idx_valid_sa = np.column_stack((idx_i, idx_j))

    # Source coordinates
    gs_pol_coo = tuple(np.array(source.polar_coordinates))
    gs_height = source.height if source.height != float('inf') else float('inf')

    # Get WFS parameters from SH object
    subap_on_diameter = wfs.subap_on_diameter
    subap_fov = wfs.subap_wanted_fov  # This is in arcsec

    # Compute reference IM
    im_ref = synim.interaction_matrix(
        pup_diam_m=pup_diam_m,
        pup_mask=pup_mask,
        dm_array=ifunc_3d,
        dm_mask=pup_mask.T,
        dm_height=0.0,
        dm_rotation=0.0,
        gs_pol_coo=gs_pol_coo,
        gs_height=gs_height,
        wfs_nsubaps=subap_on_diameter,
        wfs_rotation=0.0,
        wfs_translation=(0.0, 0.0),
        wfs_mag_global=1.0,
        wfs_fov_arcsec=subap_fov,
        idx_valid_sa=idx_valid_sa,
        verbose=False,
        specula_convention=True
    )

    return im_ref


def generate_misregistered_im(simul_params, source, dm, wfs, slopec,
                              shift_x, shift_y, rotation, magnification):
    """Generate IM with known mis-registration"""

    # Extract parameters (same as generate_reference_im)
    pup_diam_m = simul_params.pixel_pupil * simul_params.pixel_pitch
    pup_mask = np.array(dm.mask)
    ifunc_3d = np.array(dm._ifunc.ifunc_2d_to_3d(normalize=True))

    subapdata = slopec.subapdata
    display_map = np.array(subapdata.display_map)
    nx = subapdata.nx
    idx_i = display_map // nx
    idx_j = display_map % nx
    idx_valid_sa = np.column_stack((idx_i, idx_j))

    gs_pol_coo = tuple(np.array(source.polar_coordinates))
    gs_height = source.height if source.height != float('inf') else float('inf')

    subap_on_diameter = wfs.subap_on_diameter
    subap_fov = wfs.subap_wanted_fov

    # Compute IM with mis-registration
    im_misreg = synim.interaction_matrix(
        pup_diam_m=pup_diam_m,
        pup_mask=pup_mask,
        dm_array=ifunc_3d,
        dm_mask=pup_mask.T,
        dm_height=0.0,
        dm_rotation=0.0,
        gs_pol_coo=gs_pol_coo,
        gs_height=gs_height,
        wfs_nsubaps=subap_on_diameter,
        wfs_rotation=rotation,
        wfs_translation=(shift_x, shift_y),
        wfs_mag_global=1.0 + magnification,
        wfs_fov_arcsec=subap_fov,
        idx_valid_sa=idx_valid_sa,
        verbose=False,
        specula_convention=True
    )

    return im_misreg


def generate_sinusoidal_slopes(im, carrier_frequencies, duration, dt, noise_level=0.0):
    """
    Generate time series of slopes with sinusoidal modulation.
    
    Parameters
    ----------
    im : ndarray, shape (nslopes,) or (nslopes, nmodes)
        Interaction matrix (single mode or multiple modes)
    carrier_frequencies : list
        Carrier frequencies for each mode [Hz]
    duration : float
        Duration of signal [seconds]
    dt : float
        Time step [seconds]
    noise_level : float
        RMS of Gaussian noise to add
    
    Returns
    -------
    slopes_time : ndarray, shape (nt, nslopes)
        Time series of slopes
    time : ndarray, shape (nt,)
        Time vector
    """
    # Handle both 1D (single mode) and 2D (multiple modes) cases
    if im.ndim == 1:
        im = im[:, np.newaxis]  # Convert to (nslopes, 1)

    nslopes, nmodes = im.shape
    nt = int(duration / dt)
    time = np.arange(nt) * dt

    # Initialize slopes array
    slopes_time = np.zeros((nt, nslopes))

    # For each mode, add sinusoidal component
    for mode_idx in range(nmodes):
        freq = carrier_frequencies[mode_idx]
        # Unit amplitude sine wave
        modulation = np.sin(2 * np.pi * freq * time)

        # Add contribution from this mode to all slopes
        slopes_time += np.outer(modulation, im[:, mode_idx])

    # Add noise if requested
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, slopes_time.shape)
        slopes_time += noise

    return slopes_time, time


class TestSprintShSynim(unittest.TestCase):

    verbose = False  # Set to True for detailed output during tests

    @cpu_and_gpu
    def test_sprint_estimation_small(self, target_device_idx, xp):
        """Test SPRINT estimation with small mis-registration"""
        self._run_sprint_test(2.0, 1.5, 1.0, 0.02, target_device_idx, xp)

    @cpu_and_gpu
    def test_sprint_estimation_medium(self, target_device_idx, xp):
        """Test SPRINT estimation with medium mis-registration"""
        self._run_sprint_test(5.0, -3.0, 3.0, 0.05, target_device_idx, xp)

    def _run_sprint_test(self, shift_x, shift_y, rotation, magnification,
                        target_device_idx, xp):
        """Helper method to run SPRINT test with given parameters"""

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing SPRINT with mis-registration:")
            print(f"  shift_x={shift_x:.2f} px, shift_y={shift_y:.2f} px")
            print(f"  rotation={rotation:.2f} deg, magnification={magnification:.4f}")
            print(f"  target_device={target_device_idx}")
            print(f"{'='*70}")

        # Create test system
        simul_params, source, dm, wfs, ccd, slopec = create_test_system()

        # Generate reference IM (no mis-registration)
        if self.verbose: # pragma: no cover
            print("\nGenerating reference IM...")
        im_ref_full = generate_reference_im(simul_params, source, dm, wfs, slopec)

        # Select 1 mode only
        mode_idx = 30
        im_ref = im_ref_full[:, mode_idx:mode_idx+1]  # Keep 2D shape (nslopes, 1)

        if self.verbose: # pragma: no cover
            print(f"Reference IM shape: {im_ref.shape}")
            print(f"Reference IM RMS: {np.sqrt(np.mean(im_ref**2)):.3e}")

        # Generate mis-registered IM
        if self.verbose: # pragma: no cover
            print("\nGenerating mis-registered IM...")
        im_misreg_full = generate_misregistered_im(
            simul_params, source, dm, wfs, slopec,
            shift_x, shift_y, rotation, magnification
        )

        # Select same mode
        im_misreg = im_misreg_full[:, mode_idx:mode_idx+1]  # Keep 2D shape (nslopes, 1)

        if self.verbose: # pragma: no cover
            print(f"Mis-registered IM shape: {im_misreg.shape}")
            print(f"Mis-registered IM RMS: {np.sqrt(np.mean(im_misreg**2)):.3e}")
            print(f"IM difference RMS: {np.sqrt(np.mean((im_misreg - im_ref)**2)):.3e}")

        # Define carrier frequencies for single mode
        carrier_frequencies = [100.0]  # Single frequency

        if self.verbose: # pragma: no cover
            print(f"\nCarrier frequencies: {carrier_frequencies}")

        # Generate time series of slopes
        duration = 0.01  # seconds
        dt = simul_params.time_step
        noise_level = 0.0  # Start without noise

        if self.verbose: # pragma: no cover
            print(f"\nGenerating sinusoidal slopes...")
            print(f"  Duration: {duration}s, dt: {dt*1e3:.2f}ms")
            print(f"  Noise level: {noise_level}")

        slopes_time, time = generate_sinusoidal_slopes(
            im_misreg, carrier_frequencies, duration, dt, noise_level
        )

        if self.verbose: # pragma: no cover
            print(f"  Generated {slopes_time.shape[0]} time steps")
            print(f"  Slopes RMS: {np.sqrt(np.mean(slopes_time**2)):.3e}")

        slopes_time *= 100.0  # Scale up for testing optical gain estimation too

        # Create SPRINT estimator
        if self.verbose: # pragma: no cover
            print("\nCreating SPRINT estimator...")
        sprint = SprintShSynim(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            modes_index=[mode_idx],  # Only the mode we are testing
            carrier_frequencies=carrier_frequencies,
            estimation_dt=duration-0.001,  # Estimate once after full period
            max_iterations=20,
            convergence_threshold=1e-2,
            initial_misreg=None,  # Start from zero
            apply_absolute_slopes=False,
            enable_wpup_magn_xy=False,
            integration_gain=0.9,
            target_device_idx=target_device_idx,
            precision=1
        )

        # Connect SPRINT to slopec output
        sprint.inputs['in_slopes'].set(slopec.outputs['out_slopes'])

        # Setup
        sprint.setup()

        # Feed slopes time series by manually setting slopes in slopec output
        if self.verbose: # pragma: no cover
            print("\nFeeding slopes to SPRINT...")
        dummy_slopes = slopec.outputs['out_slopes']

        for t_idx, t in enumerate(time):
            # Update slopes directly
            dummy_slopes.slopes[:] = slopes_time[t_idx, :]
            t_internal = sprint.seconds_to_t(t)
            dummy_slopes.generation_time = t_internal

            # Trigger SPRINT
            sprint.check_ready(t_internal)
            sprint.trigger_code()

        # Get estimated parameters
        estimated_params = specula.cpuArray(sprint.misreg_params)

        if self.verbose: # pragma: no cover
            print(f"\n{'='*70}")
            print("ESTIMATION RESULTS:")
            print(f"{'='*70}")
            print(f"                     True       Estimated    Error       Rel.Error")
            print(f"shift_x (px):     {shift_x:8.3f}   {estimated_params[0]:8.3f}   "
                f"{estimated_params[0]-shift_x:8.3f}   "
                f"{abs(estimated_params[0]-shift_x)/abs(shift_x)*100:6.2f}%")
            print(f"shift_y (px):     {shift_y:8.3f}   {estimated_params[1]:8.3f}   "
                f"{estimated_params[1]-shift_y:8.3f}   "
                f"{abs(estimated_params[1]-shift_y)/abs(shift_y)*100:6.2f}%")
            print(f"rotation (deg):   {rotation:8.3f}   {estimated_params[2]:8.3f}   "
                f"{estimated_params[2]-rotation:8.3f}   "
                f"{abs(estimated_params[2]-rotation)/abs(rotation)*100:6.2f}%")
            print(f"magnification:    {magnification:8.5f}   {estimated_params[3]:8.5f}   "
                f"{estimated_params[3]-magnification:8.5f}   "
                f"{abs(estimated_params[3]-magnification)/abs(magnification)*100:6.2f}%")
            print(f"{'='*70}")

        # Get estimated IM (single mode)
        im_estimated = specula.cpuArray(sprint.estimated_intmat.intmat)

        # Ensure shapes match for comparison
        if im_estimated.ndim == 2 and im_estimated.shape[1] == 1:
            im_estimated = im_estimated[:, 0]
        if im_ref.ndim == 2 and im_ref.shape[1] == 1:
            im_ref_1d = im_ref[:, 0]
            im_misreg_1d = im_misreg[:, 0]
        else:
            im_ref_1d = im_ref
            im_misreg_1d = im_misreg

        # Compare IMs
        im_diff = im_estimated - im_ref_1d
        residual_rms = np.sqrt(np.mean(im_diff**2))
        initial_rms = np.sqrt(np.mean((im_misreg_1d - im_ref_1d)**2))

        if self.verbose: # pragma: no cover
            print(f"\nINTERACTION MATRIX QUALITY:")
            print(f"  Initial RMS error:   {initial_rms:.3e}")
            print(f"  Residual RMS error:  {residual_rms:.3e}")
            if residual_rms > 0:
                print(f"  Improvement factor:  {initial_rms/residual_rms:.2f}x")
            else:
                print(f"  Perfect reconstruction!")
            print(f"{'='*70}\n")

        # Assertions - allow 20% error on parameters (relaxed for robustness)
        self.assertLess(
            abs(estimated_params[0] - shift_x) / abs(shift_x), 0.1,
            f"shift_x error too large: "
            f"{abs(estimated_params[0] - shift_x) / abs(shift_x) * 100:.1f}%"
        )
        self.assertLess(
            abs(estimated_params[1] - shift_y) / abs(shift_y), 0.1,
            f"shift_y error too large: "
            f"{abs(estimated_params[1] - shift_y) / abs(shift_y) * 100:.1f}%"
        )
        self.assertLess(
            abs(estimated_params[2] - rotation) / abs(rotation), 0.1,
            f"rotation error too large: "
            f"{abs(estimated_params[2] - rotation) / abs(rotation) * 100:.1f}%"
        )
        self.assertLess(
            abs(estimated_params[3] - magnification) / abs(magnification), 0.1,
            f"magnification error too large: "
            f"{abs(estimated_params[3] - magnification) / abs(magnification) * 100:.1f}%"
        )
