import specula
specula.init(0)

import os
import shutil
import unittest
import numpy as np
from specula.base_value import BaseValue
from specula.scalar_values import StringValue, IntValue
from specula.loop_control import LoopControl
from specula.data_objects.intensity import Intensity
from specula.processing_objects.dynamic_pyr_pupdata_calibrator import DynamicPyrPupdataCalibrator
from specula.scalar_values import FloatValue
from test.specula_testlib import cpu_and_gpu
from test.test_pyr_pupdata_calibrator import TestPyrPupdataCalibrator


class TestDynamicPyrPupdataCalibrator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_pyr_pupdata_calibrator')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

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
        thr1 = FloatValue(value=3.1415)
        thr1.generation_time = 42
        calibrator.inputs['in_thr1'].set(thr1)
        calibrator.check_ready(42)
        assert calibrator.thr1 == 3.1415

        # Float input
        thr2 = FloatValue(value=3.1416)
        thr2.generation_time = 42
        calibrator.inputs['in_thr2'].set(thr2)
        calibrator.check_ready(42)
        assert calibrator.thr2 == 3.1416

    @cpu_and_gpu
    def test_save(self, target_device_idx, xp):
        """Test the in_save dynamic input"""

        calibrator = DynamicPyrPupdataCalibrator(
            data_dir=self.tmp_dir,
            dt=1,
            auto_detect_obstruction=True,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        shape = (128, 128)
        radius = 20
        image_data, _, _ = TestPyrPupdataCalibrator()._create_synthetic_pupils(xp, shape=shape, radius=radius)
        
        # Wrap in Intensity object
        in_i = Intensity(128, 128, target_device_idx=target_device_idx)
        in_i.i = image_data
        
        calibrator.inputs['in_i'].set(in_i)
        pup_name = StringValue('pupils.fits')
        calibrator.inputs['in_output_tag'].set(pup_name)
        trigger = IntValue(1)
        calibrator.inputs['in_save'].set(trigger)

        loop = LoopControl()
        loop.add(calibrator, idx=0)
        loop.start(run_time=1, dt=1)
        in_i.generation_time = in_i.seconds_to_t(0)
        pup_name.generation_time = pup_name.seconds_to_t(0)
        trigger.generation_time = trigger.seconds_to_t(0)
        loop.iter()

        fname = os.path.join(self.tmp_dir, 'pupils.fits')
        assert os.path.exists(fname)


