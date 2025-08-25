import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil

from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.processing_objects.multi_im_calibrator import MultiImCalibrator

from test.specula_testlib import cpu_and_gpu

class TestMultiImRecCalibrator(unittest.TestCase):

    def setUp(self):
        """Create unique temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_existing_im_file_is_detected(self):
        """Test that ImCalibrator detects existing IM files"""
        im_tag = 'test_im'
        im_filename = f'{im_tag}.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, full_im_tag=im_tag)
 
    def test_existing_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        im_tag = 'test_im_overwrite'
        im_filename = f'{im_tag}.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)

    @cpu_and_gpu
    def test_triggered_by_slopes(self, target_device_idx, xp):
        """Test that calibrator """
        im_tag = 'test_im_trigger'

        slopes1 = Slopes(2, target_device_idx=target_device_idx)
        slopes2 = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)
        calibrator = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd, cmd])
        calibrator.setup()

        slopes1.generation_time = 1
        slopes2.generation_time = 1
        cmd.generation_time = 1

        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that output was created and updated
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].generation_time, 1)

        # Advance both
        slopes1.generation_time = 3
        slopes2.generation_time = 3
        cmd.generation_time = 3

        calibrator.check_ready(t=3)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that trigger was executed
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].generation_time, 3)

