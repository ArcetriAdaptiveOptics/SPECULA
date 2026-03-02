import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula.data_objects.simul_params import SimulParams
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.source import Source
from specula.data_objects.ifunc import IFunc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes
from specula.processing_objects.dm import DM
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.ccd import CCD
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.sprint_pyr import SprintPyr
from test.test_sprint import generate_sinusoidal_slopes

from test.specula_testlib import cpu_and_gpu


def create_test_system():
    """Create a minimal SCAO system for testing SPRINT"""

    # Simulation parameters
    simul_params = SimulParams(
        time_step=1e-3,  # 1ms
        pixel_pupil=80,
        pixel_pitch=8.0/80  # 8m telescope
    )

    # Source (on-axis NGS)
    source = Source(
        polar_coordinates=[0.0, 0.0],  # On-axis
        magnitude=8.0,
        wavelengthInNm=750.0,
        target_device_idx=-1,
        precision=1
    )

    # Pupil stop (circular, inscribed in 80x80 grid)
    pupil_diam = 60  # pixels
    pupil_mask = Pupilstop(
        simul_params=simul_params,
        mask_diam=1.0,
        obs_diam=0.1,
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

    # Pyramid WFS - parameters will be set in SPRINT estimator, but we need to create the object
    output_resolution = 60  # pixels (per pyramid image)
    pupil_diam = 20  # pixels (diameter of each pupil image)
    pup_dist = 24  # pixels (distance between pupil centers)

    wfs = ModulatedPyramid(
        simul_params=simul_params,
        wavelengthInNm=750.0,
        fov=1.5,  # arcsec
        pup_diam=pupil_diam,  # pixels
        pup_dist=pup_dist,  # pixels
        output_resolution=output_resolution,  # pixels
        mod_amp=2.0,  # lambda/D
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

    # CCD detector
    ccd = CCD(
        simul_params=simul_params,
        size=(output_resolution, output_resolution),
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
    pupdata = create_pyramid_pupdata(framesize=output_resolution,
                                     pupil_diam=pupil_diam,
                                     pupil_dist=pup_dist)
    slopec = PyrSlopec(
        pupdata=pupdata,
        target_device_idx=-1,
        precision=1
    )

    # Connect slopec to CCD pixels
    slopec.inputs['in_pixels'].set(ccd.outputs['out_pixels'])
    slopec.setup()

    return simul_params, pupil_mask, source, dm, wfs, ccd, slopec


def create_pyramid_pupdata(framesize=80, pupil_diam=30, pupil_dist=36,
                           target_device_idx=-1, precision=1):
    """
    Create PupData for a 4-faced pyramid in a square frame.
    framesize: side of the frame (pixels)
    pupil_diam: diameter of each pupil (pixels)
    pupil_dist: distance between the centers of the pupils (pixels)
    """
    # Centers of the 4 pupils (A, B, C, D)
    cx = np.array([
        framesize//2 - pupil_dist//2,  # A (top left)
        framesize//2 + pupil_dist//2,  # B (top right)
        framesize//2 - pupil_dist//2,  # C (bottom left)
        framesize//2 + pupil_dist//2   # D (bottom right)
    ])
    cy = np.array([
        framesize//2 - pupil_dist//2,  # A (top left)
        framesize//2 - pupil_dist//2,  # B (top right)
        framesize//2 + pupil_dist//2,  # C (bottom left)
        framesize//2 + pupil_dist//2   # D (bottom right)
    ])
    radius = np.full(4, pupil_diam/2)
    ind_pup = []
    yy, xx = np.indices((framesize, framesize))
    for i in range(4):
        mask = (xx - cx[i])**2 + (yy - cy[i])**2 <= (pupil_diam/2)**2
        ind = np.where(mask)
        ind_pup.append(np.ravel_multi_index(ind, (framesize, framesize)))
    # Pad ind_pup arrays to the same length
    maxlen = max(len(x) for x in ind_pup)
    ind_pup_arr = np.full((maxlen, 4), -1, dtype=int)
    for i in range(4):
        ind_pup_arr[:len(ind_pup[i]), i] = ind_pup[i]
    return PupData(
        ind_pup=ind_pup_arr,
        radius=radius,
        cx=cx,
        cy=cy,
        framesize=[framesize, framesize],
        target_device_idx=target_device_idx,
        precision=precision
    )


def generate_ims(simul_params, pupil_mask, source, dm, wfs, slopec,
                 mode_idx, shift_x, shift_y, rotation, magnification):
    """Generate IM with known mis-registration"""

    sprint = SprintPyr(
        simul_params=simul_params,
        dm=dm,
        slopec=slopec,
        source=source,
        wfs=wfs,
        pupil_mask=pupil_mask,
        modes_index=[mode_idx],
        carrier_frequencies=[10],
        estimation_dt=1.0,
        max_iterations=20,
        convergence_threshold=1e-2,
        initial_misreg=None,
        apply_absolute_slopes=False,
        integration_gain=0.9,
        target_device_idx=-1,
        precision=1
    )
    slopes = Slopes(2, target_device_idx=-1, precision=1)
    sprint.inputs['in_slopes'].set(slopes)
    sprint.setup()

    # IM without misregistration
    sprint.misreg_params = [0.0, 0.0, 0.0, 0.0]  # Set to known values
    im_ref = sprint._compute_nominal_im()

    # IM with misregistration
    sprint.misreg_params = [shift_x, shift_y, rotation, magnification]
    im_misreg = sprint._compute_nominal_im()

    return im_ref, im_misreg


class TestSprintPyr(unittest.TestCase):

    verbose = False  # Set to True for detailed output during tests

    @cpu_and_gpu
    def test_sprint_estimation_small(self, target_device_idx, xp):
        """Test SPRINT estimation with small mis-registration"""
        self._run_sprint_test(1.0, 0.5, 1.0, 0.02, target_device_idx, xp)

    @cpu_and_gpu
    def test_sprint_estimation_medium(self, target_device_idx, xp):
        """Test SPRINT estimation with medium mis-registration"""
        self._run_sprint_test(1.0, -1.0, 20.0, 0.05, target_device_idx, xp)

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
        simul_params, pupil_mask, source, dm, wfs, ccd, slopec = create_test_system()

        # Select 1 mode only
        mode_idx = 30  # Mode to test (arbitrary choice within range)

        # Generate reference IM (no mis-registration)
        if self.verbose: # pragma: no cover
            print("\nGenerating reference and mis-registered IM...")
        im_ref_full, im_misreg_full = generate_ims(simul_params=simul_params,
                                                   source=source,
                                                   pupil_mask=pupil_mask,
                                                   dm=dm,
                                                   wfs=wfs,
                                                   slopec=slopec,
                                                   mode_idx=mode_idx,
                                                   shift_x=shift_x,
                                                   shift_y=shift_y,
                                                   rotation=rotation,
                                                   magnification=magnification)

        im_ref = im_ref_full[:, 0]  # Keep 2D shape (nslopes, 1)
        im_misreg = im_misreg_full[:, 0]  # Keep 2D shape (nslopes, 1)

        if self.verbose: # pragma: no cover
            print(f"Reference IM shape: {im_ref.shape}")
            print(f"Reference IM RMS: {np.sqrt(np.mean(im_ref**2)):.3e}")

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
        sprint = SprintPyr(
            simul_params=simul_params,
            dm=dm,
            slopec=slopec,
            source=source,
            wfs=wfs,
            pupil_mask=pupil_mask,
            modes_index=[mode_idx],  # Only the mode we are testing
            carrier_frequencies=carrier_frequencies,
            estimation_dt=duration-0.001,  # Estimate once after full period
            max_iterations=20,
            convergence_threshold=1e-2,
            initial_misreg=None,  # Start from zero
            apply_absolute_slopes=False,
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
