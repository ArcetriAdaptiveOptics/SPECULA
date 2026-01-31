import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil

from specula.processing_objects.sn_calibrator import SnCalibrator


class TestSnCalibrator(unittest.TestCase):

    def setUp(self):
        """Create unique temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_existing_sn_file_is_detected(self):
        """Test that ImCalibrator detects existing IM files"""
        sn_tag = 'test_im'
        sn_filename = f'{sn_tag}.fits' 
        sn_path = os.path.join(self.test_dir, sn_filename)

        # Create empty file
        with open(sn_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = SnCalibrator(data_dir=self.test_dir, output_tag=sn_tag)
 
    def test_existing_sn_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        sn_tag = 'test_sn_overwrite'
        sn_filename = f'{sn_tag}.fits'
        sn_path = os.path.join(self.test_dir, sn_filename)

        # Create empty file
        with open(sn_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = SnCalibrator(data_dir=self.test_dir, output_tag=sn_tag, overwrite=True)
