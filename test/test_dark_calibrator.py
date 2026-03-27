import specula
from specula.loop_control import LoopControl
specula.init(0)

import unittest
import numpy as np
from specula.base_value import BaseValue
from specula.processing_objects.dynamic_dark_calibrator import DynamicDarkCalibrator
from test.specula_testlib import cpu_and_gpu

class TestDynamicDarkCalibrator(unittest.TestCase):

    @cpu_and_gpu
    def test_invalid_nframes_raises(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            DynamicDarkCalibrator(
                data_dir=".",
                nframes=0,
                target_device_idx=target_device_idx
            )

    @cpu_and_gpu
    def test_valid_initialization(self, target_device_idx, xp):
        calib = DynamicDarkCalibrator(
            data_dir=".",
            nframes=1,
            target_device_idx=target_device_idx
        )

        self.assertIsNotNone(calib.darkframe)
        self.assertEqual(calib.nframes, 1)

    @cpu_and_gpu
    def test_darkframe_output_properties(self, target_device_idx, xp):
        calib = DynamicDarkCalibrator(
            data_dir=".",
            nframes=1,
            target_device_idx=target_device_idx
        )

        # Create dummy input pixels
        in_pixels = specula.data_objects.pixels.Pixels(
            dimx=5, dimy=6, bits=12, signed=True,
            target_device_idx=target_device_idx
        )
        calib.inputs['in_pixels'].set(in_pixels)

        calib.setup()

        self.assertEqual(calib.darkframe.size, (6, 5))
        self.assertEqual(calib.darkframe.bpp, 12)
        self.assertTrue(calib.darkframe.signed)

    @cpu_and_gpu
    def test_darkcalibrator_trigger_inputs(self, target_device_idx, xp):
        calib = DynamicDarkCalibrator(
            data_dir=".",
            nframes=2,
            target_device_idx=target_device_idx
        )

        # Create dummy input pixels
        in_pixels = specula.data_objects.pixels.Pixels(
            dimx=5, dimy=6, bits=12, signed=True,
            target_device_idx=target_device_idx
        )
        data = xp.ones((6, 5), dtype=in_pixels.dtype) * 100
        in_pixels.pixels = data

        # Trigger with no frames integrated should do nothing
        trigger = BaseValue(value=1, target_device_idx=target_device_idx)
        trigger.generation_time = trigger.seconds_to_t(0)

        calib.inputs['in_pixels'].set(in_pixels)
        calib.inputs['in_trigger'].set(trigger)

        loop = LoopControl()
        loop.add(calib, idx=0)
        loop.start(run_time=2, dt=1)
        in_pixels.generation_time = in_pixels.seconds_to_t(0)
        loop.iter()
        in_pixels.generation_time = in_pixels.seconds_to_t(1)
        loop.iter()

        self.assertTrue(np.all(calib.darkframe.pixels == data))
