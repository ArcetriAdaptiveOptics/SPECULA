import unittest
import specula
specula.init(0)  # Default target device

import numpy as np
from specula import cpuArray
from specula.calib_manager import CalibManager
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.data_objects.pupilstop import Pupilstop
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec

from test.specula_testlib import cpu_and_gpu
import os
from astropy.io import fits

class TestShSlopecMorfeo(unittest.TestCase):

    def setUp(self):
        """Set up test parameters matching MORFEO LGS1 configuration from params_morfeo_focus_ref.yml"""
        # Parameters from sh_lgs1 in params_morfeo_focus_ref.yml
        self.wavelengthInNm = 589  # LGS wavelength
        self.subap_wanted_fov = 16.1  # arcsec (from sh_lgs1)
        self.sensor_pxscale = 1.15  # arcsec/pixel (from sh_lgs1)
        self.subap_on_diameter = 68  # subapertures on diameter (from sh_lgs1)
        self.subap_npx = 14  # pixels per subaperture (from sh_lgs1)
        self.fov_ovs_coeff = 1.52  # (from sh_lgs1)
        self.rotAnglePhInDeg = 6.2  # (from sh_lgs1)

        # Parameters from main in params_morfeo_focus_ref.yml
        self.pixel_pupil = 480  # pixels (from main)
        self.pixel_pitch = 0.0802  # meters (from main)

        # Electric field parameters
        self.S0 = 100.0  # photons/s/m^2/nm

        # Reference data directory
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'data')

        # Calibration manager setup
        self.root_dir = os.path.join(os.path.dirname(__file__), 'calib')
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

    def load_pupilstop(self, target_device_idx):
        """Load Pupilstop from disk using calibration manager"""
        cm = CalibManager(self.root_dir)
        # Load the pupilstop from MORFEO configuration
        pupilstop_tag = 'EELT480pp0.0803m_obs0.283_spider2023'
        pupilstop = Pupilstop.restore(
            cm.filename('pupilstop', pupilstop_tag + '.fits'),
            target_device_idx=target_device_idx
        )
        print(f"Loaded pupilstop from calibration: {pupilstop_tag}")
        return pupilstop

    def load_subap_data(self, target_device_idx):
        """Load SubapData from disk using calibration manager"""
        cm = CalibManager(self.root_dir)
        # Load the subapdata_object from slopec_lgs1
        subapdata_tag = 'maory_np_ps480p0.080_shs68x68_wl589_fv16.1_np14_th0.50_rot6.2'
        return SubapData.restore(
            cm.filename('subapdata', subapdata_tag),
            target_device_idx=target_device_idx
        )

    def create_simplified_subap_data(self, target_device_idx):
        """Create simplified SubapData for testing when calibration files are not available"""
        # Create a simplified 4x4 subaperture pattern for testing
        test_subap_on_diameter = 4
        test_subap_npx = 14

        idxs_list = []
        display_map = np.arange(test_subap_on_diameter * test_subap_on_diameter)

        total_pixels = test_subap_on_diameter * test_subap_npx

        count = 0
        for i in range(test_subap_on_diameter):
            for j in range(test_subap_on_diameter):
                # Create indices for this subaperture
                x_start = i * test_subap_npx
                y_start = j * test_subap_npx

                indices = []
                for y in range(y_start, y_start + test_subap_npx):
                    for x in range(x_start, x_start + test_subap_npx):
                        if y < total_pixels and x < total_pixels:
                            indices.append(y * total_pixels + x)

                idxs_list.append(np.array(indices, dtype=np.int32))
                count += 1

        # Convert to format expected by SubapData
        max_indices = max(len(idx) for idx in idxs_list)
        idxs = np.zeros((len(idxs_list), max_indices), dtype=np.int32)

        for i, idx_array in enumerate(idxs_list):
            idxs[i, :len(idx_array)] = idx_array

        return SubapData(
            idxs=idxs,
            display_map=display_map,
            nx=test_subap_on_diameter,
            ny=test_subap_on_diameter,
            target_device_idx=target_device_idx
        )

    def load_reference_data(self, verbose=False):
        """Load reference phase cube, intensity and slopes from FITS files"""
        # Load reference phase cube
        phase_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_phase.fits')
        with fits.open(phase_file) as hdul:
            ref_phase_cube = hdul[0].data.copy()

        # Load reference intensity
        intensity_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_intensity.fits')
        with fits.open(intensity_file) as hdul:
            ref_intensity_cube = hdul[0].data.copy()

        # Load reference slopes
        slopes_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_slopes.fits')
        with fits.open(slopes_file) as hdul:
            ref_slopes_cube = hdul[0].data.copy()

        if verbose:
            print("Loaded reference data:")
            print(f"  Phase shape: {ref_phase_cube.shape}")
            print(f"  Intensity shape: {ref_intensity_cube.shape}")
            print(f"  Slopes shape: {ref_slopes_cube.shape}")

        return ref_phase_cube, ref_intensity_cube, ref_slopes_cube

    @unittest.skipUnless(os.getenv('RUN_MORFEO_TEST', '0') == '1',
                        "Set RUN_MORFEO_TEST=1 to run this test")
    # Example command to run this specific test:
    # RUN_MORFEO_TEST=1 python -m pytest test_sh_slopec_morfeo.py::TestShSlopecMorfeo::test_morfeo_lgs1_pipeline_with_3d_phase_array
    @cpu_and_gpu
    def test_morfeo_lgs1_pipeline_with_3d_phase_array(self, target_device_idx, xp):
        """Test complete LGS1 pipeline with 3D phase array from reference data"""

        verbose = False
        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm

        # Load reference data from FITS files
        ref_phase_cube, ref_intensity_cube, ref_slopes_cube = self.load_reference_data(verbose=verbose)

        # Use the reference phase cube
        phase_cube = ref_phase_cube
        n_frames = phase_cube.shape[0]  # Update n_frames to match reference data

        # Load pupilstop mask from calibration
        pupilstop = self.load_pupilstop(target_device_idx)
        mask = pupilstop.get_value()  # Get the amplitude mask

        # Initialize SH with MORFEO LGS1 parameters
        sh = SH(wavelengthInNm=self.wavelengthInNm,
                subap_wanted_fov=self.subap_wanted_fov,
                sensor_pxscale=self.sensor_pxscale,
                subap_on_diameter=self.subap_on_diameter,
                subap_npx=self.subap_npx,
                fov_ovs_coeff=self.fov_ovs_coeff,
                rotAnglePhInDeg=self.rotAnglePhInDeg,
                laser_launch_tel=None,
                target_device_idx=target_device_idx)

        # Load or create subaperture data
        subapdata = self.load_subap_data(target_device_idx)

        # Initialize slope computer with MORFEO LGS1 parameters
        slopec = ShSlopec(subapdata=subapdata,
                         target_device_idx=target_device_idx)

        # Create electric field with pupilstop mask
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                          S0=self.S0, target_device_idx=target_device_idx)

        # Apply pupilstop mask
        ef.A[:] = xp.array(mask)

        sh.inputs['in_ef'].set(ef)

        pixels = Pixels(self.subap_on_diameter*self.subap_npx, self.subap_on_diameter*self.subap_npx,
                        target_device_idx=target_device_idx)

        # Run slope computation
        slopec.inputs['in_pixels'].set(pixels)

        # Storage for results
        intensities = []
        slopes_list = []

        # Process each frame from the reference phase cube
        for frame_idx in range(n_frames):
            t = frame_idx + 1

            print(f"Processing frame {frame_idx + 1}/{n_frames}")

            # Set phase from reference cube
            ef.phaseInNm[:] = xp.array(phase_cube[frame_idx])
            ef.generation_time = t

            if plot_debug and frame_idx == 0:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(cpuArray(ef.A), cmap='gray')
                plt.title(f'Amplitude Frame {frame_idx + 1}')
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(cpuArray(ef.phaseInNm), cmap='jet')
                plt.title(f'Phase Frame {frame_idx + 1} (nm)')
                plt.colorbar()

            # Run SH simulation
            if frame_idx == 0:
                sh.setup()
            sh.check_ready(t)
            sh.trigger()
            sh.post_trigger()

            # Get intensity and convert to pixels
            intensity = sh.outputs['out_i']

            if plot_debug and frame_idx == 0:
                plt.figure()
                plt.imshow(cpuArray(intensity.i), cmap='hot')
                plt.colorbar()
                plt.title(f'Intensity Frame {frame_idx + 1}')

            pixels.set_value(intensity.i)
            pixels.generation_time = t

            # Run slope computation
            if frame_idx == 0:
                slopec.setup()
            slopec.check_ready(t)
            slopec.trigger()
            slopec.post_trigger()

            if plot_debug and frame_idx == 0:
                plt.figure()
                plt.plot(cpuArray(slopec.outputs['out_slopes'].slopes))
                plt.title(f'Slopes Frame {frame_idx + 1}')
                plt.show()

            # Store results
            intensities.append(cpuArray(intensity.i.copy()))
            slopes_list.append(cpuArray(slopec.outputs['out_slopes'].slopes.copy()))

            if verbose:
                print(f"  Intensity sum = {np.sum(intensities[-1]):.2e}, "
                    f"Slopes RMS = {np.std(slopes_list[-1]):.3f}")

        # Convert to arrays
        intensity_cube = np.stack(intensities)
        slopes_cube = np.stack(slopes_list)

        # comparison plot with reference data
        if plot_debug:
            frame_to_plot = 0
            plt.figure(figsize=(20, 10))

            # Intensity comparison
            plt.subplot(2, 2, 1)
            intensity_max = np.max(ref_intensity_cube[frame_to_plot])
            vmin = intensity_max * 1e-6
            vmax = intensity_max
            plt.imshow(ref_intensity_cube[frame_to_plot], cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.title('Reference Intensity')
            plt.colorbar()
            plt.subplot(2, 2, 2)
            intensity_diff = np.abs(intensity_cube[frame_to_plot] - ref_intensity_cube[frame_to_plot])
            # Use log scale with clipping to avoid log(0)
            plt.imshow(intensity_diff, cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.title('Intensity Difference (log scale)')
            plt.colorbar()

            # Slopes comparison
            plt.subplot(2, 2, 3)
            plt.plot(ref_slopes_cube[frame_to_plot], label='Reference')
            plt.plot(slopes_cube[frame_to_plot], label='Current')
            plt.title('Slopes: Reference vs Current')
            plt.xlabel('Slope Index')
            plt.ylabel('Slope Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.subplot(2, 2, 4)
            slopes_diff = slopes_cube[frame_to_plot] - ref_slopes_cube[frame_to_plot]
            plt.plot(ref_slopes_cube[frame_to_plot], label='Reference')
            plt.plot(slopes_diff, label='Difference')
            plt.title('Slopes: Reference vs Difference')
            plt.xlabel('Slope Index')
            plt.ylabel('Slope Value')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.tight_layout()
            plt.show()

        # Calculate RMS
        intensity_rms = np.sqrt(np.mean(intensity_cube**2))
        ref_intensity_rms = np.sqrt(np.mean(ref_intensity_cube**2))
        intensity_diff_rms = np.sqrt(np.mean((intensity_cube - ref_intensity_cube)**2))

        slopes_rms = np.sqrt(np.mean(slopes_cube**2))
        ref_slopes_rms = np.sqrt(np.mean(ref_slopes_cube**2))
        slopes_diff_rms = np.sqrt(np.mean((slopes_cube - ref_slopes_cube)**2))

        intensity_rms_ratio = intensity_diff_rms / ref_intensity_rms
        slopes_rms_ratio = slopes_diff_rms / ref_slopes_rms

        # Calculate max
        intensity_max_val = np.max(np.abs(ref_intensity_cube))
        intensity_max_diff = np.max(np.abs(intensity_cube - ref_intensity_cube))
        intensity_max_ratio = intensity_max_diff / intensity_max_val

        slopes_max_val = np.max(np.abs(np.abs(ref_slopes_cube)))
        slopes_max_diff = np.max(np.abs(slopes_cube - ref_slopes_cube))
        slopes_max_ratio = slopes_max_diff / slopes_max_val

        # Compare with reference data
        if verbose:
            print("\nComparing with reference data...")

            print("Intensity comparison:")
            print(f"  Current shape: {intensity_cube.shape}")
            print(f"  Reference shape: {ref_intensity_cube.shape}")
            print(f"  Current sum: {np.sum(intensity_cube):.2e}")
            print(f"  Reference sum: {np.sum(ref_intensity_cube):.2e}")
            print(f"  Difference sum (with absolute values): {np.sum(np.abs(intensity_cube - ref_intensity_cube)):.2e}")

            print(f"  Current RMS: {intensity_rms:.6f}")
            print(f"  Reference RMS: {ref_intensity_rms:.6f}")
            print(f"  Difference RMS: {intensity_diff_rms:.6f}")
            print(f"  Max intensity: {np.max(intensity_cube):.6f}")
            print(f"  Max reference intensity: {np.max(ref_intensity_cube):.6f}")
            print(f"  Max difference: {np.max(np.abs(intensity_cube - ref_intensity_cube)):.6f}")

            print("Slopes comparison:")
            print(f"  Current shape: {slopes_cube.shape}")
            print(f"  Reference shape: {ref_slopes_cube.shape}")

            print(f"  Current RMS: {slopes_rms:.6f}")
            print(f"  Reference RMS: {ref_slopes_rms:.6f}")
            print(f"  Difference RMS: {slopes_diff_rms:.6f}")
            print(f"  Max slopes: {np.max(np.abs(slopes_cube)):.6f}")
            print(f"  Max reference slopes: {np.max(np.abs(ref_slopes_cube)):.6f}")
            print(f"  Max difference: {np.max(np.abs(slopes_cube - ref_slopes_cube)):.6f}")

        # Test assertions with custom tolerances

        # (1) Max difference vs max value with different tolerances
        if verbose:
            print("\nMax difference tests:")
            print(f"  Intensity max diff ratio: {intensity_max_ratio:.4f} (should be < 0.02)")
            print(f"  Slopes max diff ratio: {slopes_max_ratio:.4f} (should be < 0.05)")

        self.assertLess(intensity_max_ratio, 0.02,
                       f"Intensity max difference ({intensity_max_ratio:.4f}) exceeds 2% of max value")
        self.assertLess(slopes_max_ratio, 0.05,
                       f"Slopes max difference ({slopes_max_ratio:.4f}) exceeds 5% of max value")

        # (2) RMS vs RMS difference with different tolerances
        if verbose:
            print("\nRMS difference tests:")
            print(f"  Intensity RMS diff ratio: {intensity_rms_ratio:.4f} (should be < 0.01)")
            print(f"  Slopes RMS diff ratio: {slopes_rms_ratio:.4f} (should be < 0.002)")

        self.assertLess(intensity_rms_ratio, 0.01,
                       f"Intensity RMS difference ({intensity_rms_ratio:.4f}) exceeds 2% of reference RMS")
        self.assertLess(slopes_rms_ratio, 0.02,
                       f"Slopes RMS difference ({slopes_rms_ratio:.4f}) exceeds 2% of reference RMS")
        if verbose:
            print("OK: Successfully compared with reference data (custom tolerances)")

    def load_reference_data_crop(self, verbose=False):
        """Load reference phase cube (cropped), intensity and slopes from FITS files for reduced test"""
        # Load reference phase cube (original, will be cropped)
        phase_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_phase.fits')
        with fits.open(phase_file) as hdul:
            ref_phase_cube_full = hdul[0].data.copy()

        # Load cropped reference intensity (already cropped to 17x17 subapertures)
        intensity_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_crop_intensity.fits')
        with fits.open(intensity_file) as hdul:
            ref_intensity_cube_crop = hdul[0].data.copy()

        # Load cropped reference slopes (already cropped to match 17x17 subapertures)
        slopes_file = os.path.join(self.test_data_dir, 'ref_test_morfeo_crop_slopes.fits')
        with fits.open(slopes_file) as hdul:
            ref_slopes_cube_crop = hdul[0].data.copy()

        # Crop the phase cube to match the reduced test (second quarter)
        # Crop each frame in the phase cube
        ref_phase_cube_crop = ref_phase_cube_full[:, 120:240, 120:240]

        if verbose:
            print("Loaded cropped reference data:")
            print(f"  Phase shape: {ref_phase_cube_crop.shape} (cropped from {ref_phase_cube_full.shape})")
            print(f"  Intensity shape: {ref_intensity_cube_crop.shape}")
            print(f"  Slopes shape: {ref_slopes_cube_crop.shape}")

        return ref_phase_cube_crop, ref_intensity_cube_crop, ref_slopes_cube_crop

    def load_reduced_subap_data(self, target_device_idx):
        """Load SubapData from disk using calibration manager"""
        cm = CalibManager(self.root_dir)
        # Load the subapdata_object from slopec_lgs1
        subapdata_tag = 'maory_np_ps480p0.080_shs68x68_wl589_fv16.1_np14_th0.50_rot6.2_reduced_17x17'
        return SubapData.restore(
            cm.filename('subapdata', subapdata_tag),
            target_device_idx=target_device_idx
        )

    @cpu_and_gpu
    def test_morfeo_lgs1_pipeline_with_cropped_data(self, target_device_idx, xp):
        """Test complete LGS1 pipeline with cropped phase and reduced subapertures (17x17)"""

        verbose = False
        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm

        print("=== Testing MORFEO LGS1 pipeline with cropped data ===")

        # Load cropped reference data from FITS files
        ref_phase_cube_crop, ref_intensity_cube_crop, ref_slopes_cube_crop = self.load_reference_data_crop(
            verbose=verbose
        )

        # Use the cropped phase cube
        phase_cube_crop = ref_phase_cube_crop
        n_frames = phase_cube_crop.shape[0]

        # Reduced parameters for cropped test
        cropped_pixel_pupil = 120  # 480 // 4
        cropped_subap_on_diameter = 17  # 68 // 4

        if verbose:
            print(f"Test parameters:")
            print(f"  Original pupil: {self.pixel_pupil}x{self.pixel_pupil}")
            print(f"  --> Cropped: {cropped_pixel_pupil}x{cropped_pixel_pupil}")
            print(f"  Original subapertures: {self.subap_on_diameter}x{self.subap_on_diameter}")
            print(f"  --> Cropped: {cropped_subap_on_diameter}x{cropped_subap_on_diameter}")
            print(f"  Frames to process: {n_frames}")

        # Load pupilstop mask from calibration and crop it
        pupilstop_full = self.load_pupilstop(target_device_idx)
        mask_full = cpuArray(pupilstop_full.get_value())

        # Crop the pupilstop mask to match cropped phase
        mask_crop = mask_full[120:240, 120:240]

        if verbose:
            print(f"Pupilstop cropped from {mask_full.shape} to {mask_crop.shape}")

        # Initialize SH with ORIGINAL parameters (SH will generate full images, then we crop)
        sh = SH(wavelengthInNm=self.wavelengthInNm,
                subap_wanted_fov=self.subap_wanted_fov,
                sensor_pxscale=self.sensor_pxscale,
                subap_on_diameter=cropped_subap_on_diameter,
                subap_npx=self.subap_npx,
                fov_ovs_coeff=self.fov_ovs_coeff,
                rotAnglePhInDeg=self.rotAnglePhInDeg,
                laser_launch_tel=None,
                target_device_idx=target_device_idx)

        # Load reduced subaperture data (17x17)
        subapdata_reduced = self.load_reduced_subap_data(target_device_idx)

        # Initialize slope computer with REDUCED subaperture data
        slopec = ShSlopec(subapdata=subapdata_reduced,
                        target_device_idx=target_device_idx)

        # Create electric field with CROPPED dimensions
        ef = ElectricField(cropped_pixel_pupil, cropped_pixel_pupil, self.pixel_pitch,
                        S0=self.S0, target_device_idx=target_device_idx)

        # Apply cropped pupilstop mask
        ef.A[:] = xp.array(mask_crop)

        sh.inputs['in_ef'].set(ef)

        # Create pixels with FULL dimensions (SH generates full image)
        pixels = Pixels(cropped_subap_on_diameter*self.subap_npx, cropped_subap_on_diameter*self.subap_npx,
                        target_device_idx=target_device_idx)

        # Connect slope computer
        slopec.inputs['in_pixels'].set(pixels)

        # Storage for results
        intensities = []
        slopes_list = []

        # Process each frame from the cropped reference phase cube
        for frame_idx in range(n_frames):
            t = frame_idx + 1

            if verbose and frame_idx % max(1, n_frames // 5) == 0:
                print(f"Processing frame {frame_idx + 1}/{n_frames}")

            # Set cropped phase from reference cube
            ef.phaseInNm[:] = xp.array(phase_cube_crop[frame_idx])
            ef.generation_time = t

            if plot_debug and frame_idx == 0:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.imshow(cpuArray(ef.A), cmap='gray')
                plt.title(f'Cropped Amplitude Frame {frame_idx + 1}')
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.imshow(cpuArray(ef.phaseInNm), cmap='jet')
                plt.title(f'Cropped Phase Frame {frame_idx + 1} (nm)')
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.imshow(mask_full, cmap='gray')
                plt.title('Original Pupilstop')
                plt.colorbar()
                plt.show()

            # Run SH simulation (generates full intensity image)
            if frame_idx == 0:
                sh.setup()
            sh.check_ready(t)
            sh.trigger()
            sh.post_trigger()

            # Get full intensity and crop it to match reference
            intensity_full = sh.outputs['out_i'].i

            if plot_debug and frame_idx == 0:
                vmax = np.max(ref_intensity_cube_crop[frame_idx])
                vmin = vmax * 1e-6

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(cpuArray(intensity_full), cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.title('Full Intensity')
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.imshow(ref_intensity_cube_crop[frame_idx], cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.title('Reference Cropped Intensity')
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.imshow(np.abs(cpuArray(intensity_full) - ref_intensity_cube_crop[frame_idx]),
                           cmap='hot', norm=LogNorm(vmin=vmin, vmax=vmax))
                plt.title('Intensity Difference')
                plt.colorbar()
                plt.show()

            # Set full pixels for slope computation
            pixels.set_value(intensity_full)
            pixels.generation_time = t

            # Run slope computation (uses reduced subapertures)
            if frame_idx == 0:
                slopec.setup()
            slopec.check_ready(t)
            slopec.trigger()
            slopec.post_trigger()

            if plot_debug and frame_idx == 0:
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(cpuArray(slopec.outputs['out_slopes'].slopes), 'o-', label='Current')
                plt.plot(ref_slopes_cube_crop[frame_idx], 'x-', alpha=0.7, label='Reference')
                plt.title(f'Slopes Frame {frame_idx + 1}')
                plt.legend()
                plt.grid(True)

                plt.subplot(1, 2, 2)
                slopes_diff = cpuArray(slopec.outputs['out_slopes'].slopes) - ref_slopes_cube_crop[frame_idx]
                plt.plot(slopes_diff, 'r-', alpha=0.7)
                plt.title('Slopes Difference')
                plt.grid(True)
                plt.show()

            # Store results
            intensities.append(cpuArray(intensity_full))
            slopes_list.append(cpuArray(slopec.outputs['out_slopes'].slopes.copy()))

            if verbose and frame_idx < 3:
                print(f"  Frame {frame_idx + 1}: Intensity sum = {np.sum(intensities[-1]):.2e}, "
                    f"Slopes RMS = {np.std(slopes_list[-1]):.3f}")

        # Convert to arrays
        intensity_cube = np.stack(intensities)
        slopes_cube = np.stack(slopes_list)

        if verbose:
            print("\nComparing with cropped reference data...")

        # Calculate metrics (same as main test)
        intensity_rms = np.sqrt(np.mean(intensity_cube**2))
        ref_intensity_rms = np.sqrt(np.mean(ref_intensity_cube_crop**2))
        intensity_diff_rms = np.sqrt(np.mean((intensity_cube - ref_intensity_cube_crop)**2))
        intensity_rms_ratio = intensity_diff_rms / ref_intensity_rms

        slopes_rms = np.sqrt(np.mean(slopes_cube**2))
        ref_slopes_rms = np.sqrt(np.mean(ref_slopes_cube_crop**2))
        slopes_diff_rms = np.sqrt(np.mean((slopes_cube - ref_slopes_cube_crop)**2))
        slopes_rms_ratio = slopes_diff_rms / ref_slopes_rms

        intensity_max_val = np.max(np.abs(ref_intensity_cube_crop))
        intensity_max_diff = np.max(np.abs(intensity_cube - ref_intensity_cube_crop))
        intensity_max_ratio = intensity_max_diff / intensity_max_val

        slopes_max_val = np.max(np.abs(ref_slopes_cube_crop))
        slopes_max_diff = np.max(np.abs(slopes_cube - ref_slopes_cube_crop))
        slopes_max_ratio = slopes_max_diff / slopes_max_val

        if verbose:
            print("Intensity comparison:")
            print(f"  Current RMS: {intensity_rms:.6f}")
            print(f"  Reference RMS: {ref_intensity_rms:.6f}")
            print(f"  RMS ratio: {intensity_rms_ratio:.6f}")
            print(f"  Max ratio: {intensity_max_ratio:.6f}")

            print("Slopes comparison:")
            print(f"  Current RMS: {slopes_rms:.6f}")
            print(f"  Reference RMS: {ref_slopes_rms:.6f}")
            print(f"  RMS ratio: {slopes_rms_ratio:.6f}")
            print(f"  Max ratio: {slopes_max_ratio:.6f}")

        # Test assertions
        self.assertLess(slopes_rms_ratio, 0.06,
                    f"Slopes RMS difference ({slopes_rms_ratio:.4f}) exceeds 2% of reference RMS")

        # Basic sanity checks
        self.assertEqual(intensity_cube.shape[0], n_frames)
        self.assertEqual(slopes_cube.shape[0], n_frames)
        self.assertEqual(slopes_cube.shape[1], subapdata_reduced.n_subaps * 2)

        if verbose:
            print("\nOk: Cropped data test completed successfully:")
            print(f"  Processed {n_frames} frames")
            print(f"  Used {subapdata_reduced.n_subaps} subapertures (17x17 grid)")
            print(f"  Phase cropped to {cropped_pixel_pupil}x{cropped_pixel_pupil} pixels")
            print("  Results match cropped reference data")

        # Final comparison plot
        if plot_debug:
            frame_to_plot = 0
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(ref_intensity_cube_crop[frame_to_plot], cmap='hot')
            plt.title('Reference Intensity (Cropped)')
            plt.colorbar()

            plt.subplot(2, 3, 2)
            plt.imshow(intensity_cube[frame_to_plot], cmap='hot')
            plt.title('Current Intensity')
            plt.colorbar()

            plt.subplot(2, 3, 3)
            diff = np.abs(intensity_cube[frame_to_plot] - ref_intensity_cube_crop[frame_to_plot])
            plt.imshow(diff, cmap='hot')
            plt.title('Intensity Difference')
            plt.colorbar()

            plt.subplot(2, 3, 4)
            plt.plot(ref_slopes_cube_crop[frame_to_plot], 'b-', alpha=0.7, label='Reference')
            plt.plot(slopes_cube[frame_to_plot], 'r-', alpha=0.7, label='Current')
            plt.title('Slopes Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 3, 5)
            slopes_diff = slopes_cube[frame_to_plot] - ref_slopes_cube_crop[frame_to_plot]
            plt.plot(slopes_diff, 'g-', alpha=0.7)
            plt.title('Slopes Difference')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()