import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import uuid
import shutil

from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue
from specula.processing_objects.im_calibrator import ImCalibrator
from specula.processing_objects.rec_calibrator import RecCalibrator

from test.specula_testlib import cpu_and_gpu

class TestImRecCalibrator(unittest.TestCase):

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

        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag)
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)

        with self.assertRaises(FileExistsError):
            calibrator.setup()

    def test_existing_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        im_tag = 'test_im_overwrite'
        im_filename = f'{im_tag}.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        slopes = Slopes(2)
        cmd = BaseValue(value=2)
        calibrator = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)

        # Should not raise
        calibrator.setup()

    def test_existing_rec_file_is_detected(self):
        """Test that RecCalibrator detects existing REC files"""
        rec_tag = 'test_rec'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        # Create mock interaction matrix
        intmat = BaseValue(value=specula.np.array([[1, 2], [3, 4]]))
        rec_calibrator = RecCalibrator(nmodes=2, data_dir=self.test_dir, rec_tag=rec_tag)
        rec_calibrator.inputs['in_intmat'].set(intmat)

        with self.assertRaises(FileExistsError):
            rec_calibrator.setup()

    @cpu_and_gpu
    def test_triggered_by_slopes_only(self, target_device_idx, xp):
        """Test that calibrator only triggers when slopes are updated"""
        im_tag = 'test_im_trigger'

        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)
        calibrator = ImCalibrator(nmodes=10, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)
        calibrator.inputs['in_slopes'].set(slopes)
        calibrator.inputs['in_commands'].set(cmd)
        calibrator.setup()

        slopes.generation_time = 1
        cmd.generation_time = 1

        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that output was created and updated
        if len(calibrator.outputs['out_im']) > 0:
            self.assertEqual(calibrator.outputs['out_im'][0].generation_time, 1)

        # Do not advance slopes.generation_time
        slopes.generation_time = 1
        cmd.generation_time = 2

        calibrator.check_ready(t=2)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that trigger was not executed (output time unchanged)
        if len(calibrator.outputs['out_im']) > 0:
            self.assertEqual(calibrator.outputs['out_im'][0].generation_time, 1)

        # Advance both
        slopes.generation_time = 3
        cmd.generation_time = 3

        calibrator.check_ready(t=3)
        calibrator.trigger()
        calibrator.post_trigger()

        # Check that trigger was executed
        if len(calibrator.outputs['out_im']) > 0:
            self.assertEqual(calibrator.outputs['out_im'][0].generation_time, 3)

    def test_existing_rec_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        rec_tag = 'test_rec_overwrite'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        intmat = BaseValue(value=specula.np.array([[1, 2], [3, 4]]))
        calibrator = RecCalibrator(nmodes=2, data_dir=self.test_dir, rec_tag=rec_tag, overwrite=True)
        calibrator.inputs['in_intmat'].set(intmat)

        # Should not raise
        calibrator.setup()

    @cpu_and_gpu
    def test_reconstructor_generation(self, target_device_idx, xp):
        """Test that reconstructor is generated from interaction matrix"""
        rec_tag = 'test_rec_gen'

        # Create mock interaction matrix (slopes x modes)
        n_slopes = 6
        n_modes = 3
        mock_im = xp.random.random((n_slopes, n_modes)).astype(xp.float32)
        intmat = BaseValue(value=mock_im, target_device_idx=target_device_idx)

        # Set generation time BEFORE setup
        intmat.generation_time = 1

        calibrator = RecCalibrator(nmodes=n_modes, data_dir=self.test_dir, rec_tag=rec_tag, overwrite=True)
        calibrator.inputs['in_intmat'].set(intmat)
        calibrator.setup()

        # Check ready and trigger
        calibrator.check_ready(t=1)
        calibrator.trigger()
        calibrator.finalize()

        # Check that file was created
        rec_path = os.path.join(self.test_dir, f'{rec_tag}.fits')
        self.assertTrue(os.path.exists(rec_path))