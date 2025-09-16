import unittest
import specula
specula.init(0)  # Default target device

import numpy as np
from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.calib_manager import CalibManager
from specula.processing_objects.atmo_infinite_evolution import AtmoInfiniteEvolution
from specula.processing_objects.wave_generator import WaveGenerator
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
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

        # Test data
        self.n_frames = 5  # Reduced for testing
        self.test_data_dir = "test_data"

        # Calibration manager setup
        self.root_dir = '/raid1/guido/PASSATA/MAORYC/'
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        # Atmospheric parameters from params_morfeo_focus_ref.yml
        self.atmo_L0 = 25.0  # [m] Outer scale
        self.atmo_heights = [0.0]  # [m] layer heights
        self.atmo_Cn2 = [1.0]  # Cn2 weights
        self.atmo_fov = 160  # arcsec
        self.seeing = 0.65  # arcsec

    def create_launcher(self, target_device_idx):
        """Create LaserLaunchTelescope matching MORFEO launcher1 configuration"""
        return LaserLaunchTelescope(
            spot_size=1.8,
            target_device_idx=target_device_idx
        )

    def load_subap_data(self, target_device_idx):
        """Load SubapData from disk using calibration manager"""
        try:
            cm = CalibManager(self.root_dir)
            # Try to load the subapdata_object from slopec_lgs1
            subapdata_tag = 'maory_np_ps480p0.080_shs68x68_wl589_fv16.1_np14_th0.50_rot6.2'
            return SubapData.restore(
                cm.filename('subapdata', subapdata_tag),
                target_device_idx=target_device_idx
            )
        except FileNotFoundError:
            # If calibration file not found, create a simplified version for testing
            print("Warning: Calibration file not found, creating simplified subap data")
            return self.create_simplified_subap_data(target_device_idx)

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

    def get_phase_cube_filename(self):
        """Generate filename for atmospheric phase cube"""
        return os.path.join(
            self.test_data_dir,
            f"atmo_phase_cube_L0{self.atmo_L0}_seeing{self.seeing}_"
            f"pupil{self.pixel_pupil}_frames{self.n_frames}.fits"
        )

    def load_or_create_atmospheric_phase_cube(self, target_device_idx, xp):
        """Load atmospheric phase cube from disk or create and save it"""

        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

        phase_cube_file = self.get_phase_cube_filename()

        # Try to load existing phase cube
        if os.path.exists(phase_cube_file):
            print(f"Loading existing atmospheric phase cube from: {phase_cube_file}")
            with fits.open(phase_cube_file) as hdul:
                phase_cube = hdul[0].data
            print(f"Loaded phase cube with shape: {phase_cube.shape}")
            print(f"Phase RMS: {np.std(phase_cube):.1f} nm, Range: [{np.min(phase_cube):.1f}, {np.max(phase_cube):.1f}] nm")
            return phase_cube
        else:
            # Create new phase cube and save it
            print(f"Creating new atmospheric phase cube...")
            phase_cube = self.create_atmospheric_phase_cube(target_device_idx, xp)

            print(f"Saving atmospheric phase cube to: {phase_cube_file}")
            fits.writeto(phase_cube_file, phase_cube.astype(np.float32), overwrite=True)
            print(f"Phase cube saved successfully")

            return phase_cube

    def create_atmospheric_phase_cube(self, target_device_idx, xp):
        """Create realistic atmospheric phase cube using AtmoInfiniteEvolution"""
        print("Creating atmospheric phase cube using AtmoInfiniteEvolution...")

        # Create SimulParams matching MORFEO configuration
        simul_params = SimulParams(
            pixel_pupil=self.pixel_pupil,
            pixel_pitch=self.pixel_pitch,
            time_step=1  # 1 millisecond steps
        )

        # Create wave generators for atmospheric parameters
        seeing_gen = WaveGenerator(constant=self.seeing, target_device_idx=target_device_idx)

        # Create simple wind model (constant wind speed and direction for reproducibility)
        wind_speeds = np.full(len(self.atmo_heights), 10.0)  # 10 m/s for all layers
        wind_directions = np.linspace(0, 360, len(self.atmo_heights))  # Different directions for each layer

        wind_speed_gen = WaveGenerator(constant=wind_speeds.tolist(), target_device_idx=target_device_idx)
        wind_direction_gen = WaveGenerator(constant=wind_directions.tolist(), target_device_idx=target_device_idx)

        # Create AtmoInfiniteEvolution with MORFEO parameters
        atmo = AtmoInfiniteEvolution(
            simul_params=simul_params,
            L0=self.atmo_L0,
            heights=self.atmo_heights,
            Cn2=self.atmo_Cn2,
            fov=self.atmo_fov,
            seed=42,  # Fixed seed for reproducibility
            target_device_idx=target_device_idx
        )

        # Connect atmospheric inputs
        atmo.inputs['seeing'].set(seeing_gen.output)
        atmo.inputs['wind_speed'].set(wind_speed_gen.output)
        atmo.inputs['wind_direction'].set(wind_direction_gen.output)

        # Setup all objects
        objects = [seeing_gen, wind_speed_gen, wind_direction_gen, atmo]
        for obj in objects:
            obj.setup()

        # Storage for phase screens
        phase_cube = []

        # Generate frames
        for frame_idx in range(self.n_frames):
            t = (frame_idx + 1) * 2e6

            print(f"  Generating atmospheric frame {frame_idx + 1}/{self.n_frames}")

            # Trigger all objects
            for obj in objects:
                obj.check_ready(t)
                obj.trigger()
                obj.post_trigger()

            layer = atmo.outputs['layer_list'][0]  # Only one layer in this test
            phase_cube.append(cpuArray(layer.phaseInNm))

        phase_cube = np.stack(phase_cube)
        print(f"Created atmospheric phase cube with shape: {phase_cube.shape}")
        print(f"Phase RMS: {np.std(phase_cube):.1f} nm, Range: [{np.min(phase_cube):.1f}, {np.max(phase_cube):.1f}] nm")

        return phase_cube

    @cpu_and_gpu
    def test_morfeo_lgs1_pipeline_with_3d_phase_array(self, target_device_idx, xp):
        """Test complete LGS1 pipeline with 3D phase array from disk"""

        plot_debug = True
        if plot_debug:
            import matplotlib.pyplot as plt

        # Load or create atmospheric phase cube (will be saved/loaded from disk)
        phase_cube = self.load_or_create_atmospheric_phase_cube(target_device_idx, xp)

        # Create LaserLaunchTelescope
        launcher = self.create_launcher(target_device_idx)

        # Initialize SH with MORFEO LGS1 parameters
        sh = SH(wavelengthInNm=self.wavelengthInNm,
                subap_wanted_fov=self.subap_wanted_fov,
                sensor_pxscale=self.sensor_pxscale,
                subap_on_diameter=self.subap_on_diameter,
                subap_npx=self.subap_npx,
                fov_ovs_coeff=self.fov_ovs_coeff,
                rotAnglePhInDeg=self.rotAnglePhInDeg,
                laser_launch_tel=launcher,
                target_device_idx=target_device_idx)

        # Load or create subaperture data
        subapdata = self.load_subap_data(target_device_idx)

        # Initialize slope computer with MORFEO LGS1 parameters
        slopec = ShSlopec(subapdata=subapdata,
                         weight_int_pixel_dt=0.2,  # from slopec_lgs1
                         window_int_pixel=True,    # from slopec_lgs1
                         target_device_idx=target_device_idx)

        # Create electric field with constant amplitude and phase from cube
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch,
                            S0=self.S0, target_device_idx=target_device_idx)

        # Apply circular mask to simulate pupil
        mask = make_mask(self.pixel_pupil, obsratio=0.0, xp=np)
        ef.A[:] = xp.array(mask)

        sh.inputs['in_ef'].set(ef)

        pixels = Pixels(self.subap_on_diameter*self.subap_npx, self.subap_on_diameter*self.subap_npx,
                        target_device_idx=target_device_idx)

        # Run slope computation
        slopec.inputs['in_pixels'].set(pixels)

        # Storage for results
        intensities = []
        slopes_list = []

        # Process each frame from the pre-computed phase cube
        for frame_idx in range(self.n_frames):
            t = frame_idx + 1

            print(f"Processing frame {frame_idx + 1}/{self.n_frames}")

            #  phase from cube
            ef.phaseInNm[:] = xp.array(phase_cube[frame_idx])  # Use i-th frame from phase cube
            ef.generation_time = t

            if plot_debug:
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

            if plot_debug:
                plt.figure()
                plt.imshow(cpuArray(intensity.i), cmap='hot')
                plt.colorbar()
                plt.title(f'Intensity Frame {frame_idx + 1}')
                plt.show()

            pixels.set_value(intensity.i)
            pixels.generation_time = t

            # Run slope computation
            slopec.inputs['in_pixels'].set(pixels)
            if frame_idx == 0:
                slopec.setup()
            slopec.check_ready(t)
            slopec.trigger()
            slopec.post_trigger()

            if plot_debug:
                plt.figure()
                plt.plot(cpuArray(slopec.outputs['out_slopes'].slopes))
                plt.title(f'Slopes Frame {frame_idx + 1}')
                plt.show()

            # Store results
            intensities.append(cpuArray(intensity.i.copy()))
            slopes_list.append(cpuArray(slopec.outputs['out_slopes'].slopes.copy()))

            print(f"  Intensity sum = {np.sum(intensities[-1]):.2e}, "
                  f"Slopes RMS = {np.std(slopes_list[-1]):.3f}")

        # Convert to arrays
        intensity_cube = np.stack(intensities)
        slopes_cube = np.stack(slopes_list)

        # # Basic sanity checks
        # self.assertEqual(intensity_cube.shape[0], self.n_frames)
        # self.assertEqual(slopes_cube.shape[0], self.n_frames)
        # self.assertEqual(slopes_cube.shape[1], subapdata.n_subaps * 2)

        # # Check that intensities are positive and finite
        # self.assertTrue(np.all(intensity_cube >= 0))
        # self.assertTrue(np.all(np.isfinite(intensity_cube)))

        # # Check that slopes are finite
        # self.assertTrue(np.all(np.isfinite(slopes_cube)))

        # # Verify that we have non-trivial slopes (atmospheric turbulence should create slopes)
        # slopes_rms = np.std(slopes_cube)
        # self.assertGreater(slopes_rms, 0.001, "Slopes should have non-trivial RMS due to atmospheric turbulence")

        # Save or compare reference data for intensities and slopes
        ref_intensity_file = os.path.join(self.test_data_dir, "morfeo_lgs1_intensity_ref.fits")
        ref_slopes_file = os.path.join(self.test_data_dir, "morfeo_lgs1_slopes_ref.fits")

        if not os.path.exists(ref_intensity_file):
            fits.writeto(ref_intensity_file, intensity_cube.astype(np.float32), overwrite=True)
            fits.writeto(ref_slopes_file, slopes_cube.astype(np.float32), overwrite=True)
            print(f"Saved reference data:")
            print(f"  Intensity: {ref_intensity_file}")
            print(f"  Slopes: {ref_slopes_file}")
        else:
            # Compare with reference data
            with fits.open(ref_intensity_file) as hdul:
                ref_intensity = hdul[0].data
            with fits.open(ref_slopes_file) as hdul:
                ref_slopes = hdul[0].data

            np.testing.assert_allclose(intensity_cube, ref_intensity, rtol=1e-10)
            np.testing.assert_allclose(slopes_cube, ref_slopes, rtol=1e-10)
            print("Successfully compared with reference data")

        print(f"\nTest completed successfully:")
        print(f"  Phase cube shape: {phase_cube.shape}")
        print(f"  Intensity cube shape: {intensity_cube.shape}")
        print(f"  Slopes cube shape: {slopes_cube.shape}")
        print(f"  Phase RMS: {np.std(phase_cube):.1f} nm")
        print(f"  Intensity sum per frame: {np.mean([np.sum(frame) for frame in intensities]):.2e}")
        print(f"  Slopes RMS: {np.std(slopes_cube):.3f}")


if __name__ == '__main__':
    unittest.main()