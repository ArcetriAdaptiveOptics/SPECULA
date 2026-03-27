import specula
specula.init(0)

import unittest
import numpy as np
from specula.base_value import BaseValue
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

    @cpu_and_gpu
    def test_interactive_inputs(self, target_device_idx, xp):
        """Test that interactive inputs are processed"""

        calibrator = DynamicPyrPupdataCalibrator(
            data_dir="/tmp",
            auto_detect_obstruction=True,
            target_device_idx=target_device_idx
        )

        # Float input
        thr1 = BaseValue(value=3.1415)
        thr1.generation_time = 42
        calibrator.inputs['in_thr1'].set(thr1)
        calibrator.check_ready(42)
        assert calibrator.thr1 == 3.1415

        # String input converted to float
        thr1 = BaseValue(value='0.123')
        thr1.generation_time = 42
        calibrator.inputs['in_thr1'].set(thr1)
        calibrator.check_ready(42)

        assert calibrator.thr1 == 0.123

        # Float input
        thr2 = BaseValue(value=3.1415)
        thr2.generation_time = 42
        calibrator.inputs['in_thr2'].set(thr2)
        calibrator.check_ready(42)
        assert calibrator.thr2 == 3.1415

        # String input converted to float
        thr2 = BaseValue(value='0.123')
        thr2.generation_time = 42
        calibrator.inputs['in_thr2'].set(thr2)
        calibrator.check_ready(42)

        assert calibrator.thr2 == 0.123

        
