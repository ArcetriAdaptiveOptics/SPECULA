import specula
specula.init(0)

import unittest
import numpy as np

from specula.base_value import BaseValue
from specula.loop_control import LoopControl
from specula.data_objects.pixels import Pixels
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

    @cpu_and_gpu
    def test_interactive_inputs(self, target_device_idx, xp):
        """Test that interactive inputs are processed"""

        calibrator = DynamicDarkCalibrator(
            data_dir="/tmp",
            nframes=10,
            target_device_idx=target_device_idx
        )

        dummy_pixels = Pixels(10,10)
        calibrator.inputs['in_pixels'].set(dummy_pixels)

        # Float input
        nframes = BaseValue(value=10.0)
        nframes.generation_time = 42
        calibrator.inputs['in_nframes'].set(nframes)
        calibrator.check_ready(42)
        assert calibrator.nframes == 10

        # String input converted to int
        nframes = BaseValue(value='10')
        nframes.generation_time = 42
        calibrator.inputs['in_nframes'].set(nframes)
        calibrator.check_ready(42)
        assert calibrator.nframes == 10

    @cpu_and_gpu
    def test_darkframe_size(self, target_device_idx, xp):
        """Test that dark frame has the same dimensions as the input pixels after setup"""

        calibrator = DynamicDarkCalibrator(
            data_dir="/tmp",
            nframes=10,
            target_device_idx=target_device_idx
        )
        pixshape = (10, 20)
        dummy_pixels = Pixels(pixshape[1], pixshape[0])
        calibrator.inputs['in_pixels'].set(dummy_pixels)

        calibrator.setup()

        assert calibrator.darkframe.pixels.shape == pixshape

    @cpu_and_gpu
    def test_output_pixel_size(self, target_device_idx, xp):
        """Test that output pixels have the same dimensions as the input pixels after setup"""

        calibrator = DynamicDarkCalibrator(
            data_dir="/tmp",
            nframes=10,
            target_device_idx=target_device_idx
        )
        pixshape = (10, 20)
        dummy_pixels = Pixels(pixshape[1], pixshape[0])
        calibrator.inputs['in_pixels'].set(dummy_pixels)

        calibrator.setup()

        assert calibrator.outputs['out_subtracted_pixels'].pixels.shape == pixshape

    @cpu_and_gpu
    def test_reset_inputs(self, target_device_idx, xp):
        """Test that the reset commands zeroes out the dark frame"""

        calibrator = DynamicDarkCalibrator(
            data_dir="/tmp",
            nframes=10,
            target_device_idx=target_device_idx
        )
        dummy_pixels = Pixels(10,10)
        calibrator.inputs['in_pixels'].set(dummy_pixels)

        calibrator.darkframe = Pixels(10, 10)
        calibrator.darkframe.pixels += 1

        reset = BaseValue(value=10.0)
        reset.generation_time = 42
        calibrator.inputs['in_reset'].set(reset)
        calibrator.check_ready(42)

        assert calibrator.darkframe.pixels.sum() == 0
