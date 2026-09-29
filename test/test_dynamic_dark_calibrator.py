import specula
specula.init(0)

import os
import shutil
import unittest
from unittest import mock
import numpy as np

from specula.loop_control import LoopControl
from specula.data_objects.pixels import Pixels
from specula.processing_objects.dynamic_dark_calibrator import DynamicDarkCalibrator
from specula.scalar_values import IntValue, StringValue
from test.specula_testlib import cpu_and_gpu
from specula import cpuArray


class TestDynamicDarkCalibrator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_dark_calibrator')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

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
        trigger = IntValue(value=1)
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

        # Integer input
        nframes = IntValue(value=10)
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

        reset = IntValue(value=10)
        reset.generation_time = 42
        calibrator.inputs['in_reset'].set(reset)
        calibrator.check_ready(42)

        assert calibrator.darkframe.pixels.sum() == 0


    @cpu_and_gpu
    def test_negative_number_of_frames(self, target_device_idx, xp):
        """Test that negative number of frames log an error and do not change the current nframes value"""

        calibrator = DynamicDarkCalibrator(
            data_dir="/tmp",
            nframes=10,
            target_device_idx=target_device_idx
        )

        dummy_pixels = Pixels(10,10)
        calibrator.inputs['in_pixels'].set(dummy_pixels)

        # Negative integer input
        nframes = IntValue(value=-5)
        calibrator.inputs['in_nframes'].set(nframes)
        with self.assertLogs(calibrator.logger.logger, level='ERROR') as log:
            nframes.generation_time = 42
            calibrator.inputs['in_nframes'].set(nframes)
            calibrator.check_ready(42)
    
    @cpu_and_gpu
    def test_load(self, target_device_idx, xp):
        """Test the in_load dynamic input"""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            target_device_idx=target_device_idx
        )

        in_pixels = Pixels(10,10, target_device_idx=target_device_idx)
        dark = Pixels(10,10, target_device_idx=target_device_idx)
        dark.pixels = xp.arange(100, dtype=xp.int16).reshape((10,10))
        dark.save(os.path.join(self.tmp_dir, 'dark.fits'))
        darkname = StringValue('dark.fits')
        calibrator.inputs['in_pixels'].set(in_pixels)
        calibrator.inputs['in_load'].set(darkname)

        loop = LoopControl()
        loop.add(calibrator, idx=0)
        loop.start(run_time=1, dt=1)
        in_pixels.generation_time = in_pixels.seconds_to_t(0)
        darkname.generation_time = darkname.seconds_to_t(0)
        loop.iter()

        print(f'{dark.pixels=}')
        print(f'{calibrator.darkframe.pixels=}')
        np.testing.assert_array_equal(cpuArray(dark.pixels),
                                      cpuArray(calibrator.darkframe.pixels))

    @cpu_and_gpu
    def test_save(self, target_device_idx, xp):
        """Test the in_save dynamic input"""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        in_pixels = Pixels(10,10, target_device_idx=target_device_idx)
        in_pixels.pixels = xp.arange(100, dtype=xp.int16).reshape((10,10))
        calibrator.inputs['in_pixels'].set(in_pixels)
        darkname = StringValue('dark.fits')
        calibrator.inputs['in_save'].set(darkname)
        dark_nframes = IntValue(1)
        calibrator.inputs['in_nframes'].set(dark_nframes)
        dark_trigger = IntValue(1)
        calibrator.inputs['in_trigger'].set(dark_trigger)

        loop = LoopControl()
        loop.add(calibrator, idx=0)
        loop.start(run_time=1, dt=1)
        in_pixels.generation_time = in_pixels.seconds_to_t(0)
        darkname.generation_time = darkname.seconds_to_t(0)
        dark_nframes.generation_time = dark_nframes.seconds_to_t(0)
        dark_trigger.generation_time = dark_trigger.seconds_to_t(0)
        loop.iter()

        test_dark = Pixels.restore(os.path.join(self.tmp_dir, darkname.value))

        print(f'{test_dark.pixels=}')
        print(f'{in_pixels.pixels=}')
        np.testing.assert_array_equal(cpuArray(test_dark.pixels),
                                      cpuArray(in_pixels.pixels))

    @cpu_and_gpu
    def test_dark_frame_tag_loads_at_setup(self, target_device_idx, xp):
        """Test that dark_frame_tag loads the dark frame in setup(), in place,
        with and without the '.fits' extension."""

        dark = Pixels(10, 10, target_device_idx=target_device_idx)
        dark.pixels = xp.arange(100, dtype=dark.pixels.dtype).reshape((10, 10))
        dark.save(os.path.join(self.tmp_dir, 'dark.fits'))

        for tag in ('dark.fits', 'dark'):
            with self.subTest(tag=tag):
                calibrator = DynamicDarkCalibrator(
                    data_dir=self.tmp_dir,
                    nframes=10,
                    dark_frame_tag=tag,
                    target_device_idx=target_device_idx
                )
                darkframe_before_setup = calibrator.darkframe

                in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
                in_pixels.pixels = xp.full((10, 10), 200, dtype=in_pixels.dtype)
                calibrator.inputs['in_pixels'].set(in_pixels)

                loop = LoopControl()
                loop.add(calibrator, idx=0)
                loop.start(run_time=1, dt=1)
                in_pixels.generation_time = in_pixels.seconds_to_t(0)
                loop.iter()

                np.testing.assert_array_equal(cpuArray(dark.pixels),
                                              cpuArray(calibrator.darkframe.pixels))
                self.assertIs(calibrator.outputs['out_darkframe'], darkframe_before_setup)
                expected = cpuArray(in_pixels.pixels) - cpuArray(dark.pixels)
                np.testing.assert_array_equal(
                    cpuArray(calibrator.outputs['out_subtracted_pixels'].pixels), expected)

    @cpu_and_gpu
    def test_dark_frame_tag_missing_file_raises(self, target_device_idx, xp):
        """Test that a missing dark_frame_tag file raises during setup()"""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            dark_frame_tag='missing_dark.fits',
            target_device_idx=target_device_idx
        )
        in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        calibrator.inputs['in_pixels'].set(in_pixels)

        with self.assertRaises(FileNotFoundError):
            calibrator.setup()

    @cpu_and_gpu
    def test_dark_frame_tag_shape_mismatch_raises(self, target_device_idx, xp):
        """Test that a dark_frame_tag file with a different shape raises ValueError in setup()"""

        wrong_shape_dark = Pixels(5, 5, target_device_idx=target_device_idx)
        wrong_shape_dark.save(os.path.join(self.tmp_dir, 'wrong_shape.fits'))

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            dark_frame_tag='wrong_shape.fits',
            target_device_idx=target_device_idx
        )
        in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        calibrator.inputs['in_pixels'].set(in_pixels)

        with self.assertRaises(ValueError):
            calibrator.setup()

    @cpu_and_gpu
    def test_in_load_keeps_darkframe_object_identity(self, target_device_idx, xp):
        """Regression test: in_load must update the darkframe object in place,
        so out_darkframe keeps referencing the same object (it used to be
        replaced, leaving the output stale)."""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            target_device_idx=target_device_idx
        )

        in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        dark = Pixels(10, 10, target_device_idx=target_device_idx)
        dark.pixels = xp.arange(100, dtype=xp.int16).reshape((10, 10))
        dark.save(os.path.join(self.tmp_dir, 'dark.fits'))
        darkname = StringValue('dark.fits')

        calibrator.inputs['in_pixels'].set(in_pixels)
        calibrator.inputs['in_load'].set(darkname)

        loop = LoopControl()
        loop.add(calibrator, idx=0)
        loop.start(run_time=1, dt=1)
        darkframe_before = calibrator.outputs['out_darkframe']

        in_pixels.generation_time = in_pixels.seconds_to_t(0)
        darkname.generation_time = darkname.seconds_to_t(0)
        loop.iter()

        self.assertIs(calibrator.outputs['out_darkframe'], darkframe_before)
        np.testing.assert_array_equal(cpuArray(dark.pixels), cpuArray(calibrator.darkframe.pixels))

    @cpu_and_gpu
    def test_in_load_missing_file_logs_error_and_keeps_dark(self, target_device_idx, xp):
        """A failed runtime in_load must be logged and must not corrupt the
        current dark frame (previously the except clause itself crashed)."""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            target_device_idx=target_device_idx
        )
        in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        calibrator.inputs['in_pixels'].set(in_pixels)
        calibrator.setup()

        calibrator.darkframe.pixels[:] = 7
        previous = calibrator.darkframe.pixels.copy()
        darkframe_before = calibrator.outputs['out_darkframe']

        missing_name = StringValue('missing_dark.fits')
        calibrator.inputs['in_load'].set(missing_name)

        with self.assertLogs(calibrator.logger.logger, level='ERROR') as log:
            missing_name.generation_time = 1
            calibrator.check_ready(1)

        self.assertTrue(any('FileNotFoundError' in msg for msg in log.output))
        np.testing.assert_array_equal(cpuArray(calibrator.darkframe.pixels), cpuArray(previous))
        self.assertIs(calibrator.outputs['out_darkframe'], darkframe_before)

    @cpu_and_gpu
    def test_in_save_failure_logs_error(self, target_device_idx, xp):
        """A failed runtime in_save must be logged with the exception class name and not raise"""

        calibrator = DynamicDarkCalibrator(
            data_dir=self.tmp_dir,
            nframes=10,
            target_device_idx=target_device_idx
        )
        in_pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        calibrator.inputs['in_pixels'].set(in_pixels)

        darkname = StringValue('dark.fits')
        darkname.generation_time = 1
        calibrator.inputs['in_save'].set(darkname)
        calibrator.check_ready(1)

        with mock.patch.object(calibrator, 'save', side_effect=RuntimeError('boom')):
            with self.assertLogs(calibrator.logger.logger, level='ERROR') as log:
                calibrator.post_trigger()

        self.assertTrue(any('RuntimeError' in msg for msg in log.output))

