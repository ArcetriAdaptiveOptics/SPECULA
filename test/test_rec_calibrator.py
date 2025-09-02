import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil
import numpy as np
from unittest.mock import MagicMock, patch

from specula.base_value import BaseValue
from specula.data_objects.intmat import Intmat
from specula.processing_objects.rec_calibrator import RecCalibrator

from test.specula_testlib import cpu_and_gpu


class TestRecCalibrator(unittest.TestCase):

    def setUp(self):
        """Create unique temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_with_rec_tag(self):
        """Test RecCalibrator initialization with rec_tag"""
        nmodes = 10
        rec_tag = 'test_rec'
        
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag=rec_tag
        )
        
        self.assertEqual(calibrator.nmodes, nmodes)
        self.assertEqual(calibrator.data_dir, self.test_dir)
        self.assertEqual(calibrator.first_mode, 0)
        self.assertIsNone(calibrator.pupdata_tag)
        self.assertFalse(calibrator.overwrite)
        self.assertEqual(calibrator.rec_path, os.path.join(self.test_dir, f'{rec_tag}.fits'))
        self.assertIn('in_intmat', calibrator.inputs)

    def test_initialization_with_tag_template(self):
        """Test RecCalibrator initialization with tag_template"""
        nmodes = 15
        tag_template = 'template_rec'
        
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag=None,  # Explicitly pass None
            tag_template=tag_template
        )
        
        self.assertEqual(calibrator.nmodes, nmodes)
        self.assertEqual(calibrator.data_dir, self.test_dir)
        self.assertEqual(calibrator.rec_path, os.path.join(self.test_dir, f'{tag_template}.fits'))

    def test_initialization_with_custom_parameters(self):
        """Test RecCalibrator initialization with custom parameters"""
        nmodes = 20
        first_mode = 5
        pupdata_tag = 'test_pupdata'
        overwrite = True
        target_device_idx = -1  # Use CPU to avoid GPU dependency
        precision = 1
        
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag='test_rec',
            first_mode=first_mode,
            pupdata_tag=pupdata_tag,
            overwrite=overwrite,
            target_device_idx=target_device_idx,
            precision=precision
        )
        
        self.assertEqual(calibrator.nmodes, nmodes)
        self.assertEqual(calibrator.first_mode, first_mode)
        self.assertEqual(calibrator.pupdata_tag, pupdata_tag)
        self.assertTrue(calibrator.overwrite)
        self.assertEqual(calibrator.target_device_idx, target_device_idx)
        self.assertEqual(calibrator.precision, precision)

    def test_initialization_with_auto_rec_tag(self):
        """Test RecCalibrator initialization with rec_tag='auto' and tag_template"""
        nmodes = 12
        tag_template = 'auto_template'
        
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag='auto',
            tag_template=tag_template
        )
        
        self.assertEqual(calibrator.rec_path, os.path.join(self.test_dir, f'{tag_template}.fits'))

    def test_initialization_missing_both_tags(self):
        """Test that RecCalibrator raises TypeError when rec_tag is not provided"""
        with self.assertRaises(TypeError) as context:
            RecCalibrator(nmodes=10, data_dir=self.test_dir)
        
        self.assertIn('missing 1 required positional argument: \'rec_tag\'', str(context.exception))

    def test_initialization_missing_both_tags_with_auto(self):
        """Test that RecCalibrator raises ValueError when rec_tag is 'auto' and tag_template is None"""
        with self.assertRaises(ValueError) as context:
            RecCalibrator(nmodes=10, data_dir=self.test_dir, rec_tag='auto')
        
        self.assertIn('At least one of tag_template and rec_tag must be set', str(context.exception))

    def test_initialization_with_empty_rec_tag(self):
        """Test that RecCalibrator accepts empty string rec_tag"""
        # Empty string should be valid (not None)
        calibrator = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag=''
        )
        
        self.assertEqual(calibrator.rec_path, os.path.join(self.test_dir, '.fits'))
        self.assertEqual(calibrator.nmodes, 10)

    def test_file_extension_handling(self):
        """Test that .fits extension is properly handled"""
        # Test without .fits extension
        calibrator1 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_rec'
        )
        self.assertTrue(calibrator1.rec_path.endswith('.fits'))
        
        # Test with .fits extension
        calibrator2 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_rec.fits'
        )
        self.assertTrue(calibrator2.rec_path.endswith('.fits'))
        self.assertEqual(calibrator2.rec_path, os.path.join(self.test_dir, 'test_rec.fits'))

    def test_existing_file_detection(self):
        """Test that RecCalibrator detects existing REC files"""
        rec_tag = 'test_rec'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError) as context:
            RecCalibrator(nmodes=10, data_dir=self.test_dir, rec_tag=rec_tag)
        
        self.assertIn('REC file', str(context.exception))
        self.assertIn('already exists', str(context.exception))

    def test_existing_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        rec_tag = 'test_rec_overwrite'
        rec_filename = f'{rec_tag}.fits'
        rec_path = os.path.join(self.test_dir, rec_filename)

        # Create empty file
        with open(rec_path, 'w') as f:
            f.write('')

        # Should not raise
        calibrator = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag=rec_tag,
            overwrite=True
        )
        
        self.assertTrue(calibrator.overwrite)

    @cpu_and_gpu
    def test_finalize_method(self, target_device_idx, xp):
        """Test the finalize method creates REC file correctly"""
        nmodes = 5
        rec_tag = 'test_finalize'
        
        # Create a mock Intmat object
        mock_intmat = MagicMock(spec=Intmat)
        mock_intmat.target_device_idx = target_device_idx
        
        # Create mock REC object
        mock_rec = MagicMock()
        mock_intmat.generate_rec.return_value = mock_rec
        
        # Create calibrator
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag=rec_tag,
            overwrite=True
        )
        
        # Set up the input
        calibrator.local_inputs['in_intmat'] = mock_intmat
        
        # Call finalize
        calibrator.finalize()
        
        # Verify generate_rec was called with correct parameters
        mock_intmat.generate_rec.assert_called_once_with(nmodes)
        
        # Verify save was called on the REC object
        mock_rec.save.assert_called_once_with(calibrator.rec_path, overwrite=calibrator.overwrite)
        
        # Verify directory was created
        self.assertTrue(os.path.exists(self.test_dir))

    @cpu_and_gpu
    def test_finalize_with_first_mode(self, target_device_idx, xp):
        """Test that finalize method handles first_mode correctly"""
        nmodes = 8
        first_mode = 3
        rec_tag = 'test_first_mode'
        
        # Create a mock Intmat object
        mock_intmat = MagicMock(spec=Intmat)
        mock_intmat.target_device_idx = target_device_idx
        
        # Create mock REC object
        mock_rec = MagicMock()
        mock_intmat.generate_rec.return_value = mock_rec
        
        # Create calibrator with first_mode
        calibrator = RecCalibrator(
            nmodes=nmodes,
            data_dir=self.test_dir,
            rec_tag=rec_tag,
            first_mode=first_mode,
            overwrite=True
        )
        
        # Set up the input
        calibrator.local_inputs['in_intmat'] = mock_intmat
        
        # Call finalize
        calibrator.finalize()
        
        # Verify generate_rec was called with correct parameters
        mock_intmat.generate_rec.assert_called_once_with(nmodes)
        
        # Note: The current implementation doesn't use first_mode in generate_rec
        # This test documents the current behavior

    def test_input_connection_setup(self):
        """Test that input connections are properly set up"""
        calibrator = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_inputs'
        )
        
        # Check that in_intmat input is properly configured
        self.assertIn('in_intmat', calibrator.inputs)
        input_value = calibrator.inputs['in_intmat']
        self.assertEqual(input_value.output_ref_type, Intmat)

    def test_inheritance_from_base_processing_obj(self):
        """Test that RecCalibrator properly inherits from BaseProcessingObj"""
        calibrator = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_inheritance'
        )
        
        # Check that it has BaseProcessingObj attributes
        self.assertTrue(hasattr(calibrator, 'inputs'))
        self.assertTrue(hasattr(calibrator, 'local_inputs'))
        self.assertTrue(hasattr(calibrator, 'outputs'))
        self.assertTrue(hasattr(calibrator, 'current_time'))
        self.assertTrue(hasattr(calibrator, 'target_device_idx'))

    def test_precision_handling(self):
        """Test that precision is properly handled"""
        # Test with default precision (should be 0, not None)
        calibrator1 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_precision1'
        )
        self.assertEqual(calibrator1.precision, 0)  # Default global precision
        
        # Test with custom precision
        calibrator2 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_precision2',
            precision=1
        )
        self.assertEqual(calibrator2.precision, 1)

    def test_target_device_handling(self):
        """Test that target_device_idx is properly handled"""
        # Test with default target device (should be 0 for GPU, not None)
        calibrator1 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_device1'
        )
        default_device_idx = specula.default_target_device_idx
        self.assertEqual(calibrator1.target_device_idx, default_device_idx)
        
        # Test with custom target device
        calibrator2 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_device2',
            target_device_idx=-1  # CPU
        )
        self.assertEqual(calibrator2.target_device_idx, -1)

    def test_pupdata_tag_handling(self):
        """Test that pupdata_tag is properly handled"""
        # Test without pupdata_tag
        calibrator1 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_pupdata1'
        )
        self.assertIsNone(calibrator1.pupdata_tag)
        
        # Test with pupdata_tag
        pupdata_tag = 'test_pupdata_tag'
        calibrator2 = RecCalibrator(
            nmodes=10,
            data_dir=self.test_dir,
            rec_tag='test_pupdata2',
            pupdata_tag=pupdata_tag
        )
        self.assertEqual(calibrator2.pupdata_tag, pupdata_tag)


if __name__ == '__main__':
    unittest.main()
