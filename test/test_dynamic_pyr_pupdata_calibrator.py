import specula
specula.init(0)

import unittest
import numpy as np
from specula.data_objects.intensity import Intensity
from specula.processing_objects.dynamic_pyr_pupdata_calibrator import DynamicPyrPupdataCalibrator
from test.specula_testlib import cpu_and_gpu
from test.test_pyr_pupdata_calibrator import TestPyrPupdataCalibrator

class TestDynamicPyrPupdataCalibrator(unittest.TestCase):

    @cpu_and_gpu
    def test_exception_catch(self, target_device_idx, xp):
        """Test that invalid parameters trigger exceptions that are catched"""
        shape = (128, 128)
        radius = 20
        image_data, _, _ = TestPyrPupdataCalibrator()._create_synthetic_pupils(xp, shape=shape, radius=radius)
        
        # Wrap in Intensity object
        in_i = Intensity(128, 128, target_device_idx=target_device_idx)
        in_i.i = image_data
        
        calibrator = DynamicPyrPupdataCalibrator(
            data_dir="/tmp",
            thr1 = 2.0, # invalid
            auto_detect_obstruction=True,
            target_device_idx=target_device_idx
        )
        
        # Manually set input
        calibrator.local_inputs['in_i'] = in_i
        
        # Run calibration
        calibrator.trigger_code()
        assert calibrator.status_string != 'OK'

        
