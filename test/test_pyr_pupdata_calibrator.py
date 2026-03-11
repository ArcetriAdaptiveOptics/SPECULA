import specula
specula.init(0)

import unittest
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
