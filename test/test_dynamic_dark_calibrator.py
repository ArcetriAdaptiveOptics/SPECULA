import specula
specula.init(0)

import unittest
from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.processing_objects.dynamic_dark_calibrator import DynamicDarkCalibrator
from test.specula_testlib import cpu_and_gpu

class TestDynamicDarkCalibrator(unittest.TestCase):

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
