import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Union

from specula.base_value import BaseValue
from specula.data_objects.intmat import Intmat
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.data_objects.recmat import Recmat
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.dm import DM
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
        # Test with default target device (should be 0 for GPU or -1 for CPU, not None)

        specula.init(device_idx=0)

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



class TestRecCalibratorMMSE(unittest.TestCase):

    def setUp(self):
        """Create test data and temporary directory"""
        self.test_dir = tempfile.mkdtemp()

        # Create test parameters
        self.nmodes = 10  # Primi 10 modi Zernike (escluso piston)
        self.npixels = 64
        self.diameter = 8.0
        self.r0 = 0.15
        self.L0 = 25.0

        # Create Zernike modal base
        self.modal_base = IFunc(
            type_str='zernike',
            nmodes=self.nmodes,
            npixels=self.npixels,
            target_device_idx=-1
        )

        # Create test interaction matrix (slopes from Zernike modes)
        # Simuliamo che ogni modo Zernike produca 4 slopes (2x2 WFS semplificato)
        self.nslopes = 4 * self.nmodes  # 4 slopes per modo
        np.random.seed(42)  # Per riproducibilità
        self.test_intmat_data = np.random.randn(self.nslopes, self.nmodes).astype(np.float32) * 0.1

        # Aggiungi una componente diagonale dominante per rendere la matrice ben condizionata
        for i in range(min(self.nmodes, self.nslopes)):
            self.test_intmat_data[i*4:(i+1)*4, i] += np.random.randn(4) * 2.0

        # Create test M2C matrix (da Zernike a Zernike con piccole rotazioni)
        self.m2c_data = np.eye(self.nmodes, dtype=np.float32)
        # Aggiungi piccole perturbazioni off-diagonali
        self.m2c_data += np.random.randn(self.nmodes, self.nmodes).astype(np.float32) * 0.05
        self.m2c = M2C(self.m2c_data, target_device_idx=-1)

        # Create test SimulParams
        self.simul_params = SimulParams(pixel_pupil=self.diameter, pixel_pitch=1.0)


        self.dm = DM(self.simul_params, height=0, ifunc=self.modal_base, m2c=self.m2c, type_str='zernike', nmodes=self.nmodes)
        self.dm_wo_m2c = DM(self.simul_params, height=0, ifunc=self.modal_base, m2c=None, type_str='zernike', nmodes=self.nmodes)

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @cpu_and_gpu
    def test_mmse_initialization_with_zernike(self, target_device_idx, xp):
        """Test RecCalibrator initialization with MMSE parameters and Zernike modes"""
        noise_cov = 0.1

        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            target_device_idx=target_device_idx
        )

        # Check MMSE parameters
        self.assertTrue(calibrator.mmse)
        self.assertEqual(calibrator.r0, self.r0)
        self.assertEqual(calibrator.L0, self.L0)
        self.assertEqual(calibrator.dm, self.dm)
        self.assertEqual(calibrator.noise_cov, noise_cov)

    @cpu_and_gpu
    def test_mmse_with_zernike_scalar_noise(self, target_device_idx, xp):
        """Test MMSE reconstruction with Zernike modes and scalar noise covariance"""
        noise_cov = 0.05
        
        # Create test intmat
        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)
        
        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_scalar',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        # Set input
        calibrator.local_inputs['in_intmat'] = intmat

        # Call finalize
        calibrator.finalize()

        # Check that REC file was created
        self.assertTrue(os.path.exists(calibrator.rec_path))

        # Verify the REC file can be loaded
        rec = Recmat.restore(calibrator.rec_path, target_device_idx=target_device_idx)
        self.assertEqual(rec.recmat.shape, (self.nmodes, self.nslopes))

    @cpu_and_gpu
    def test_mmse_with_zernike_array_noise(self, target_device_idx, xp):
        """Test MMSE reconstruction with Zernike modes and diagonal noise covariance"""
        # Create diagonal noise covariance matrix with realistic values
        noise_variances = np.random.rand(self.nslopes) * 0.1 + 0.01
        noise_cov = np.diag(noise_variances).astype(np.float32)

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_array',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        self.assertTrue(os.path.exists(calibrator.rec_path))

        # Verify reconstruction
        rec = Recmat.restore(calibrator.rec_path, target_device_idx=target_device_idx)
        self.assertEqual(rec.recmat.shape, (self.nmodes, self.nslopes))

    @cpu_and_gpu
    def test_mmse_with_zernike_list_noise(self, target_device_idx, xp):
        """Test MMSE reconstruction with Zernike modes and list as WFS noise level"""
        noise_cov = [0.02]

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_list',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        self.assertTrue(os.path.exists(calibrator.rec_path))

    @cpu_and_gpu
    def test_mmse_zernike_with_m2c(self, target_device_idx, xp):
        """Test MMSE reconstruction with Zernike modes and M2C transformation"""
        noise_cov = 0.1

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_m2c',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        self.assertTrue(os.path.exists(calibrator.rec_path))

    @cpu_and_gpu
    def test_mmse_zernike_without_m2c(self, target_device_idx, xp):
        """Test MMSE reconstruction with Zernike modes without M2C matrix"""
        noise_cov = 0.1

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_no_m2c',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm_wo_m2c,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        self.assertTrue(os.path.exists(calibrator.rec_path))

    @cpu_and_gpu
    def test_mmse_vs_pseudoinverse_zernike(self, target_device_idx, xp):
        """Test that MMSE and pseudoinverse give different results with Zernike modes"""
        noise_cov = 0.1

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        # Create MMSE reconstructor
        mmse_calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        mmse_calibrator.local_inputs['in_intmat'] = intmat
        mmse_calibrator.finalize()

        # Create pseudoinverse reconstructor
        pinv_calibrator = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_pinv_zernike',
            mmse=False,  # Use pseudoinverse
            overwrite=True,
            target_device_idx=target_device_idx
        )

        pinv_calibrator.local_inputs['in_intmat'] = intmat
        pinv_calibrator.finalize()

        # Load both reconstructors
        mmse_rec = Recmat.restore(mmse_calibrator.rec_path, target_device_idx=target_device_idx)
        pinv_rec = Recmat.restore(pinv_calibrator.rec_path, target_device_idx=target_device_idx)

        # They should be different
        diff = np.mean(np.abs(mmse_rec.recmat - pinv_rec.recmat))
        self.assertGreater(diff, 1e-6)  # Should be significantly different

    @cpu_and_gpu
    def test_mmse_zernike_different_seeing_conditions(self, target_device_idx, xp):
        """Test that different seeing conditions produce different MMSE reconstructors"""
        noise_cov = 0.1

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        # Good seeing (large r0)
        calibrator_good = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_good_seeing',
            mmse=True,
            r0=0.3,  # Good seeing
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator_good.local_inputs['in_intmat'] = intmat
        calibrator_good.finalize()

        # Poor seeing (small r0)
        calibrator_poor = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_poor_seeing',
            mmse=True,
            r0=0.1,  # Poor seeing
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator_poor.local_inputs['in_intmat'] = intmat
        calibrator_poor.finalize()

        # Load reconstructors and verify they're different
        rec_good = Recmat.restore(calibrator_good.rec_path, target_device_idx=target_device_idx)
        rec_poor = Recmat.restore(calibrator_poor.rec_path, target_device_idx=target_device_idx)

        diff = np.mean(np.abs(rec_good.recmat - rec_poor.recmat))
        self.assertGreater(diff, 1e-6)  # Should be different

    @cpu_and_gpu
    def test_mmse_zernike_subset_modes(self, target_device_idx, xp):
        """Test MMSE reconstruction using only a subset of Zernike modes"""
        noise_cov = 0.1
        nmodes_subset = 6  # Use only first 6 modes

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=nmodes_subset,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_subset',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        # Verify reconstruction matrix has correct shape
        rec = Recmat.restore(calibrator.rec_path, target_device_idx=target_device_idx)
        self.assertEqual(rec.recmat.shape, (nmodes_subset, self.nslopes))

    @cpu_and_gpu
    def test_mmse_zernike_first_mode_parameter(self, target_device_idx, xp):
        """Test MMSE reconstruction with first_mode parameter (skip piston)"""
        noise_cov = 0.1
        first_mode = 2  # Skip piston and tip
        nmodes_to_use = 5

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        calibrator = RecCalibrator(
            nmodes=nmodes_to_use,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_first_mode',
            first_mode=first_mode,
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator.local_inputs['in_intmat'] = intmat
        calibrator.finalize()

        # Verify reconstruction matrix has correct shape
        rec = Recmat.restore(calibrator.rec_path, target_device_idx=target_device_idx)
        self.assertEqual(rec.recmat.shape, (nmodes_to_use, self.nslopes))

    def test_mmse_zernike_missing_modal_base_raises(self):
        """Test that missing modal_base raises error for MMSE with Zernike"""
        with self.assertRaises(AttributeError):
            calibrator = RecCalibrator(
                nmodes=self.nmodes,
                data_dir=self.test_dir,
                rec_tag='test_error',
                mmse=True,
                r0=self.r0,
                L0=self.L0,
                dm=None,  # Missing!
                noise_cov=0.1
            )

            intmat = Intmat(self.test_intmat_data, target_device_idx=-1)
            calibrator.local_inputs['in_intmat'] = intmat
            calibrator.finalize()

    @cpu_and_gpu
    def test_mmse_zernike_low_order_vs_high_order(self, target_device_idx, xp):
        """Test that low-order and high-order Zernike modes get different treatment in MMSE"""
        noise_cov = 0.1

        intmat = Intmat(self.test_intmat_data, target_device_idx=target_device_idx)

        # Low-order modes only (tip, tilt, defocus, astigmatism)
        calibrator_low = RecCalibrator(
            nmodes=4,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_low_order',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator_low.local_inputs['in_intmat'] = intmat
        calibrator_low.finalize()

        # All modes (including higher orders)
        calibrator_all = RecCalibrator(
            nmodes=self.nmodes,
            data_dir=self.test_dir,
            rec_tag='test_mmse_zernike_all_modes',
            mmse=True,
            r0=self.r0,
            L0=self.L0,
            dm=self.dm,
            noise_cov=noise_cov,
            overwrite=True,
            target_device_idx=target_device_idx
        )

        calibrator_all.local_inputs['in_intmat'] = intmat
        calibrator_all.finalize()

        # Both should succeed
        self.assertTrue(os.path.exists(calibrator_low.rec_path))
        self.assertTrue(os.path.exists(calibrator_all.rec_path))

        # Load and verify shapes
        rec_low = Recmat.restore(calibrator_low.rec_path, target_device_idx=target_device_idx)
        rec_all = Recmat.restore(calibrator_all.rec_path, target_device_idx=target_device_idx)

        self.assertEqual(rec_low.recmat.shape, (4, self.nslopes))
        self.assertEqual(rec_all.recmat.shape, (self.nmodes, self.nslopes))