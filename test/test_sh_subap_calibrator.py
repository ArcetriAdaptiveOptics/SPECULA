import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import specula
specula.init(0)  # Default target device

from specula.data_objects.intensity import Intensity
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_subap_calibrator import ShSubapCalibrator

from test.specula_testlib import cpu_and_gpu


class TestShSubapCalibrator(unittest.TestCase):

    @cpu_and_gpu
    def test_init_requires_tag_or_template(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            ShSubapCalibrator(
                subap_on_diameter=4,
                data_dir=".",
                energy_th=0.5,
                target_device_idx=target_device_idx
            )

    @cpu_and_gpu
    def test_setup_requires_one_input(self, target_device_idx, xp):
        calib = ShSubapCalibrator(
            subap_on_diameter=4,
            data_dir=".",
            energy_th=0.5,
            output_tag="test",
            target_device_idx=target_device_idx
        )
        # Case 1: no inputs
        with self.assertRaises(ValueError):
            calib.setup()

        # Case 2: both inputs set
        calib.inputs["in_i"].set(Intensity(8, 8, target_device_idx=target_device_idx))
        calib.inputs["in_pixels"].set(Pixels(8, 8, target_device_idx=target_device_idx))
        with self.assertRaises(ValueError):
            calib.setup()

    @cpu_and_gpu
    def test_trigger_code_with_intensity(self, target_device_idx, xp):
        calib = ShSubapCalibrator(
            subap_on_diameter=2,
            data_dir=".",
            energy_th=0.0,  # no threshold to simplify
            output_tag="test",
            target_device_idx=target_device_idx
        )
        # Give Intensity input
        ii = Intensity(8, 8, target_device_idx=target_device_idx)
        ii.i = xp.ones((8,8))
        calib.inputs["in_i"].set(ii)
        calib.setup()

        calib.trigger_code()
        self.assertIsInstance(calib.subaps, SubapData)

    @cpu_and_gpu
    def test_trigger_code_with_pixels(self, target_device_idx, xp):
        calib = ShSubapCalibrator(
            subap_on_diameter=2,
            data_dir=".",
            energy_th=0.0,
            output_tag="test",
            target_device_idx=target_device_idx
        )
        # Give Pixels input
        pix = Pixels(8, 8, target_device_idx=target_device_idx)
        pix.pixels = xp.ones((8,8))
        calib.inputs["in_pixels"].set(pix)

        calib.setup()

        calib.trigger_code()
        self.assertIsInstance(calib.subaps, SubapData)

    @cpu_and_gpu
    def test_finalize_saves_file(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as tmpdir:
            calib = ShSubapCalibrator(
                subap_on_diameter=2,
                data_dir=tmpdir,
                energy_th=0.0,
                output_tag="output_file",
                target_device_idx=target_device_idx
            )
            pix = Pixels(8, 8, target_device_idx=target_device_idx)
            pix.pixels = xp.ones((8,8))
            calib.inputs["in_pixels"].set(pix)

            calib.setup()
            calib.trigger_code()

            # Patch save method of SubapData
            with patch.object(SubapData, "save", MagicMock()) as mock_save:
                calib.finalize()
                mock_save.assert_called_once()
                file_path = os.path.join(tmpdir, "output_file.fits")
                args, kwargs = mock_save.call_args
                self.assertEqual(args[0], file_path)
