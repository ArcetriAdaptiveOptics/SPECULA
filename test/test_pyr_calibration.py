import unittest
import os
import shutil
import specula
specula.init(0)

from specula.simul import Simul
from test.specula_testlib import cpu_and_gpu, assert_HDU_contents_match
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator
from specula import cpuArray
import numpy as np

class TestPyrPupdataCalibration(unittest.TestCase):
    """Test Pyramid PupData calibration by comparing generated calibration files with reference ones"""

    def setUp(self):
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        
        """Set up test by ensuring calibration directory exists"""
        # Make sure the calib directory exists
        os.makedirs(os.path.join(self.calibdir, 'pupils'), exist_ok=True)

        self.pupdata_ref_path = os.path.join(self.datadir, 'scao_pupdata_ref.fits')
        self.pupdata_path = os.path.join(self.calibdir, 'pupils', 'scao_pupdata.fits')

        self._cleanFiles()
        self.cwd = os.getcwd()

    def _cleanFiles(self):
        if os.path.exists(self.pupdata_path):
            os.remove(self.pupdata_path)

    def tearDown(self):
        self._cleanFiles()
        os.chdir(self.cwd)

    def test_pyr_pupdata_calibration(self):
        """Test Pyramid PupData calibration by comparing generated calibration file with reference"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Check if reference file exists
        self.assertTrue(os.path.exists(self.pupdata_ref_path), f"Reference file {self.pupdata_ref_path} does not exist")

        # Run the simulation for calibration
        yml_files = ['params_scao_pyr_test.yml','params_scao_pyr_test_pupdata.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Check if the calibration file was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Pyramid PupData calibration file was not generated")

        # Compare the generated file with the reference file
        assert_HDU_contents_match(self.pupdata_path, self.pupdata_ref_path)

        print("Pyramid PupData calibration matches reference!")

    @cpu_and_gpu
    def test_generate_indices_modes(self, target_device_idx, xp):
        """Test _generate_indices with different radii in both modes"""

        verbose = False  # Set to True to see detailed output
        plot_debug = False  # Set to True to visualize results

        # Create test data with different radii
        centers = xp.array([[50, 50], [150, 50], [50, 150], [150, 150]], dtype=float)
        radii = xp.array([20, 25, 30, 35], dtype=float)  # Different radii
        image_shape = (200, 200)

        # Test INTENSITY mode (slopes_from_intensity=True)
        calibrator_intensity = PyrPupdataCalibrator(
            data_dir="/tmp",
            slopes_from_intensity=True,
            target_device_idx=target_device_idx
        )
        calibrator_intensity.central_obstruction_ratio = 0.0
        ind_pup_intensity = calibrator_intensity._generate_indices(centers, radii, image_shape)

        # Test SLOPES mode (slopes_from_intensity=False)
        calibrator_slopes = PyrPupdataCalibrator(
            data_dir="/tmp", 
            slopes_from_intensity=False,
            target_device_idx=target_device_idx
        )
        calibrator_slopes.central_obstruction_ratio = 0.0
        ind_pup_slopes = calibrator_slopes._generate_indices(centers, radii, image_shape)

        if plot_debug:
            import matplotlib.pyplot as plt
            # 2D array for visualization
            pup_intensity = np.zeros(image_shape, dtype=int)
            pup_slopes = np.zeros(image_shape, dtype=int)
            for i in range(4):
                # 2D array needs to be flattened for indexing
                pup_intensity.ravel()[ind_pup_intensity[:, i]] = i + 1
                pup_slopes.ravel()[ind_pup_slopes[:, i]] = i + 1
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("INTENSITY Mode Pupils")
            plt.imshow(pup_intensity, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("SLOPES Mode Pupils")
            plt.imshow(pup_slopes, cmap='jet', interpolation='nearest')
            plt.colorbar()
            plt.show()

        # Helper function to check if indices are translation-equivalent
        def are_translation_equivalent(indices1, indices2, image_shape, xp=xp):
            """Check if indices2 can be obtained by integer translation of indices1"""
            h, w = image_shape

            # Remove -1 padding
            valid1 = indices1[indices1 >= 0]
            valid2 = indices2[indices2 >= 0]

            if len(valid1) != len(valid2):
                return False

            if len(valid1) == 0:
                return True

            # Convert to 2D coordinates
            y1, x1 = xp.unravel_index(valid1, (h, w))
            y2, x2 = xp.unravel_index(valid2, (h, w))

            # Calculate translation vector
            dx = x2[0] - x1[0]
            dy = y2[0] - y1[0]

            # Check if all points follow the same translation
            expected_x = x1 + dx
            expected_y = y1 + dy

            return np.allclose(cpuArray(x2), cpuArray(expected_x)) and np.allclose(cpuArray(y2), cpuArray(expected_y))

        # Test INTENSITY mode: should have DIFFERENT geometries due to different radii
        intensity_pixel_counts = [len(ind_pup_intensity[:, i][ind_pup_intensity[:, i] >= 0]) for i in range(4)]
        self.assertGreater(len(set(intensity_pixel_counts)), 1, 
                        "INTENSITY mode should produce different geometries with different radii")

        # Test SLOPES mode: all should be translation-equivalent
        for i in range(1, 4):
            self.assertTrue(are_translation_equivalent(ind_pup_slopes[:, 0], ind_pup_slopes[:, i], image_shape, xp=xp), 
                        f"SLOPES mode: Pupil {i} should be translation-equivalent to Pupil 0")

    @unittest.skipIf(int(os.getenv('CREATE_REF', 0)) < 1, "This test is only used to create reference files")
    def test_create_reference_file(self):
        """Create reference file for Pyramid PupData calibration"""

        # Change to test directory 
        os.chdir(os.path.dirname(__file__))

        # Run the simulation for calibration
        yml_files = ['params_scao_pyr_test.yml','params_scao_pyr_test_pupdata.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Check if the calibration file was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Pyramid PupData calibration file was not generated")

        # Copy file to reference directory
        shutil.copy(self.pupdata_path, self.pupdata_ref_path)
        print("Reference file created and saved to test/data/")
        print("Please commit this file to the repository for future tests")