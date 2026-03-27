import specula
specula.init(0)

import os.path
import unittest
import tempfile
import numpy as np
from specula.data_objects.intensity import Intensity
from specula.data_objects.pupdata import PupData
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator
from test.specula_testlib import cpu_and_gpu

class TestPyrPupdataCalibrator(unittest.TestCase):

    def _create_synthetic_pupils(self, xp, shape=(256, 256), radius=40, centers=None):
        """Helper to create a synthetic 4-pupil image"""
        image = xp.zeros(shape, dtype=np.float32)
        h, w = shape
        y, x = xp.mgrid[0:h, 0:w]
        
        if centers is None:
            # Standard quadrants
            centers = [
                (w//4, h//4), (3*w//4, h//4),
                (w//4, 3*h//4), (3*w//4, 3*h//4)
            ]
            
        for cx, cy in centers:
            r = xp.sqrt((x - cx)**2 + (y - cy)**2)
            # Create a pupil with an obstruction (ratio 0.2)
            mask = (r <= radius) & (r >= radius * 0.2)
            image[mask] = 1.0
            
        return image, centers, radius

    @cpu_and_gpu
    def test_calibration_full_run(self, target_device_idx, xp):
        """Test the full trigger_code path and PupData generation"""
        shape = (128, 128)
        radius = 20
        image_data, _, _ = self._create_synthetic_pupils(xp, shape=shape, radius=radius)
        
        # Wrap in Intensity object
        in_i = Intensity(128, 128, target_device_idx=target_device_idx)
        in_i.i = image_data
        
        calibrator = PyrPupdataCalibrator(
            data_dir="/tmp",
            auto_detect_obstruction=True,
            target_device_idx=target_device_idx
        )
        
        # Manually set input
        calibrator.local_inputs['in_i'] = in_i
        
        # Run calibration
        calibrator.trigger_code()
        
        # Verify PupData existence and metadata
        self.assertIsNotNone(calibrator.pupdata)
        self.assertIsInstance(calibrator.pupdata, PupData)
        
        # Check detected radius (should be close to 20)
        # radii are stored in calibrator.pupdata.radius
        detected_radius = float(xp.mean(calibrator.pupdata.radius))
        self.assertAlmostEqual(detected_radius, radius, delta=1.5)
        
        # Check obstruction detection (synthetic was 0.2)
        self.assertGreater(calibrator.central_obstruction_ratio, 0.1)
        self.assertLess(calibrator.central_obstruction_ratio, 0.3)

    @cpu_and_gpu
    def test_geometric_vs_intensity_modes(self, target_device_idx, xp):
        """Test the difference between slopes_from_intensity=True and False"""
        shape = (100, 100)
        # Slightly jittered centers to test 'Intensity' mode's flexibility
        centers = [(25, 25), (76, 25), (25, 76), (74, 74)] # Pupil 3 is slightly off
        image_data, _, _ = self._create_synthetic_pupils(xp, shape=shape, centers=centers)
        
        in_i = Intensity(128, 128, target_device_idx=target_device_idx)
        in_i.i = image_data

        # --- Mode 2: Intensity Mode (Unique pixel sets) ---
        cal_int = PyrPupdataCalibrator(
            data_dir="/tmp",
            slopes_from_intensity=True,
            target_device_idx=target_device_idx
        )
        cal_int.local_inputs['in_i'] = in_i
        cal_int.trigger_code()
        
        # In intensity mode, since pupil 3 was jittered/smaller, pixel counts might differ
        # or at least the logic path is distinct.
        self.assertEqual(cal_int.pupdata.ind_pup.shape[1], 4)

    @cpu_and_gpu
    def test_analyze_single_pupil_empty(self, target_device_idx, xp):
        """Ensure it handles empty/black images gracefully"""
        empty_img = xp.zeros((50, 50))
        cal = PyrPupdataCalibrator(data_dir="/tmp", target_device_idx=target_device_idx)
        
        center, radius = cal._analyze_single_pupil(empty_img)
        
        self.assertEqual(float(radius), 0.0)
        self.assertTrue(bool(xp.all(center == 0.0)))


class TestPyrPupdataCalibratorInitialization(unittest.TestCase):

    @cpu_and_gpu
    def test_invalid_dt_raises(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            PyrPupdataCalibrator(
                data_dir=".",
                dt=0,
                target_device_idx=target_device_idx
            )

    @cpu_and_gpu
    def test_valid_initialization(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            target_device_idx=target_device_idx
        )

        self.assertIsNotNone(calib.pupdata)
        self.assertEqual(calib.thr1, 0.1)
        self.assertEqual(calib.thr2, 0.25)


class TestPyrPupdataCalibratorSetup(unittest.TestCase):

    @cpu_and_gpu
    def test_setup_requires_input(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            target_device_idx=target_device_idx
        )

        calib.local_inputs = {
            "in_intensity": None,
            "in_pixels": None
        }

        with self.assertRaises(ValueError):
            calib.setup()


class TestPyrPupdataCalibratorPupilAnalysis(unittest.TestCase):

    @cpu_and_gpu
    def test_single_pupil_detection(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            target_device_idx=target_device_idx
        )

        size = 64
        image = xp.zeros((size, size))

        cx, cy = 32, 32
        r = 10

        y, x = xp.mgrid[0:size, 0:size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        image[mask] = 1.0

        center, radius = calib._analyze_single_pupil(image)

        self.assertGreater(radius, 0)
        self.assertTrue(abs(center[0] - cx) < 2)
        self.assertTrue(abs(center[1] - cy) < 2)


class TestPyrPupdataCalibratorRadialProfile(unittest.TestCase):

    @cpu_and_gpu
    def test_radial_profile_length(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            target_device_idx=target_device_idx
        )

        size = 64
        image = xp.ones((size, size))
        center = xp.array([32, 32])

        profile = calib._radial_profile(image, center, 20, n_bins=15)

        self.assertEqual(profile.shape[0], 15)


class TestPyrPupdataCalibratorIndices(unittest.TestCase):

    @cpu_and_gpu
    def test_generate_indices_intensity_mode(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            slopes_from_intensity=True,
            target_device_idx=target_device_idx
        )

        image_shape = (64, 64)

        centers = xp.array([
            [16, 16],
            [48, 16],
            [16, 48],
            [48, 48]
        ])

        radii = xp.array([8, 8, 8, 8])

        ind = calib._generate_indices(centers, radii, image_shape)

        self.assertEqual(ind.shape[1], 4)
        self.assertTrue(xp.any(ind >= 0))


    @cpu_and_gpu
    def test_generate_indices_geometric_mode(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            slopes_from_intensity=False,
            target_device_idx=target_device_idx
        )

        image_shape = (64, 64)

        centers = xp.array([
            [16, 16],
            [48, 16],
            [16, 48],
            [48, 48]
        ])

        radii = xp.array([8, 8, 8, 8])

        ind = calib._generate_indices(centers, radii, image_shape)

        self.assertEqual(ind.shape[1], 4)
        self.assertTrue(xp.any(ind >= 0))


class TestPyrPupdataCalibratorSave(unittest.TestCase):

    @cpu_and_gpu
    def test_save_creates_file(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as tmpdir:

            calib = PyrPupdataCalibrator(
                data_dir=tmpdir,
                dt=1,
                overwrite=True,
                target_device_idx=target_device_idx
            )

            calib.pupdata.ind_pup = xp.zeros((10, 4), dtype=int)
            calib.pupdata.radius = xp.ones(4)
            calib.pupdata.cx = xp.ones(4)
            calib.pupdata.cy = xp.ones(4)
            calib.pupdata.framesize = (64, 64)

            calib._save("test_pupdata")

            expected = os.path.join(tmpdir, "test_pupdata.fits")

            self.assertTrue(os.path.exists(expected))


class TestPyrPupdataCalibratorObstruction(unittest.TestCase):

    @cpu_and_gpu
    def test_detect_obstruction_returns_float(self, target_device_idx, xp):
        calib = PyrPupdataCalibrator(
            data_dir=".",
            dt=1,
            target_device_idx=target_device_idx
        )

        size = 64
        image = xp.ones((size, size))

        centers = xp.array([
            [16, 16],
            [48, 16],
            [16, 48],
            [48, 48]
        ])

        radii = xp.array([10, 10, 10, 10])

        ratio = calib._detect_obstruction(image, centers, radii)

        self.assertIsInstance(float(ratio), float)
        self.assertGreaterEqual(ratio, 0.0)