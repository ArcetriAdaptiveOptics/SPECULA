import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
import shutil
import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.processing_objects.multi_im_calibrator import MultiImCalibrator

from test.specula_testlib import cpu_and_gpu


class TestMultiImCalibrator(unittest.TestCase):

    def setUp(self):
        """Create unique temporary directory for each test"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization_with_valid_parameters(self):
        """Test that MultiImCalibrator initializes correctly with valid parameters"""
        calibrator = MultiImCalibrator(
            nmodes=10, 
            n_inputs=3, 
            data_dir=self.test_dir, 
            overwrite=True
        )
        
        self.assertEqual(calibrator.nmodes, 10)
        self.assertEqual(calibrator.n_inputs, 3)
        self.assertEqual(calibrator.data_dir, self.test_dir)
        self.assertTrue(calibrator.overwrite)
        self.assertEqual(len(calibrator.outputs['out_intmat_list']), 3)
        self.assertIsInstance(calibrator.outputs['out_intmat_full'], Intmat)

    def test_initialization_with_tags(self):
        """Test that MultiImCalibrator initializes correctly with custom tags"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            im_tag='custom_im',
            full_im_tag='custom_full_im',
            overwrite=True
        )
        
        self.assertEqual(calibrator.im_filename, 'custom_im')
        self.assertEqual(calibrator.full_im_filename, 'custom_full_im')

    def test_initialization_with_tag_templates(self):
        """Test that MultiImCalibrator initializes correctly with tag templates"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            im_tag='auto',
            im_tag_template='template_im_{}',
            full_im_tag='auto',
            full_im_tag_template='template_full_im',
            overwrite=True
        )
        
        self.assertEqual(calibrator.im_filename, 'template_im_{}')
        self.assertEqual(calibrator.full_im_filename, 'template_full_im')

    def test_tag_filename_validation(self):
        """Test that tag_filename method validates parameters correctly"""
        # Test auto tag without template
        with self.assertRaises(ValueError):
            MultiImCalibrator(
                nmodes=5, 
                n_inputs=2, 
                data_dir=self.test_dir, 
                im_tag='auto',
                overwrite=True
            )

    def test_existing_im_file_is_detected(self):
        """Test that MultiImCalibrator detects existing IM files"""
        im_tag = 'test_im'
        im_filename = f'{im_tag}0.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, im_tag=im_tag)

    def test_existing_full_im_file_is_detected(self):
        """Test that MultiImCalibrator detects existing full IM files"""
        full_im_tag = 'test_full_im'
        full_im_filename = f'{full_im_tag}.fits'
        full_im_path = os.path.join(self.test_dir, full_im_filename)

        # Create empty file
        with open(full_im_path, 'w') as f:
            f.write('')

        with self.assertRaises(FileExistsError):
            _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, full_im_tag=full_im_tag)

    def test_existing_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        im_tag = 'test_im_overwrite'
        im_filename = f'{im_tag}0.fits'
        im_path = os.path.join(self.test_dir, im_filename)

        # Create empty file
        with open(im_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, im_tag=im_tag, overwrite=True)

    def test_existing_full_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing full IM files"""
        full_im_tag = 'test_full_im_overwrite'
        full_im_filename = f'{full_im_tag}.fits'
        full_im_path = os.path.join(self.test_dir, full_im_filename)

        # Create empty file
        with open(full_im_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir, full_im_tag=full_im_tag, overwrite=True)

    def test_im_path_generation(self):
        """Test that im_path method generates correct paths"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            im_tag='test_im',
            overwrite=True
        )
        
        expected_path = os.path.join(self.test_dir, 'test_im0.fits')
        self.assertEqual(calibrator.im_path(0), expected_path)
        
        # Test with None filename
        calibrator_no_im = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True
        )
        self.assertIsNone(calibrator_no_im.im_path(0))

    def test_full_im_path_generation(self):
        """Test that full_im_path method generates correct paths"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            full_im_tag='test_full_im',
            overwrite=True
        )
        
        expected_path = os.path.join(self.test_dir, 'test_full_im.fits')
        self.assertEqual(calibrator.full_im_path(), expected_path)
        
        # Test with None filename
        calibrator_no_full_im = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True
        )
        self.assertIsNone(calibrator_no_full_im.full_im_path())

    @cpu_and_gpu
    def test_setup_validation_success(self, target_device_idx, xp):
        """Test that setup validates inputs correctly when they match expected counts"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(3, target_device_idx=target_device_idx)
        slopes2 = Slopes(3, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        
        # Should not raise
        calibrator.setup()

    @cpu_and_gpu
    def test_setup_validation_slopes_mismatch(self, target_device_idx, xp):
        """Test that setup raises error when slopes count doesn't match n_inputs"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=3, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(3, target_device_idx=target_device_idx)
        slopes2 = Slopes(3, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        
        with self.assertRaises(ValueError) as context:
            calibrator.setup()
        self.assertIn("Number of input slopes (2) does not match expected n_inputs (3)", str(context.exception))

    @cpu_and_gpu
    def test_setup_validation_commands_mismatch(self, target_device_idx, xp):
        """Test that setup raises error when commands count doesn't match n_inputs"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=3, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(3, target_device_idx=target_device_idx)
        slopes2 = Slopes(3, target_device_idx=target_device_idx)
        slopes3 = Slopes(3, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2, slopes3])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        
        with self.assertRaises(ValueError) as context:
            calibrator.setup()
        self.assertIn("Number of input commands (2) does not match expected n_inputs (3)", str(context.exception))

    @cpu_and_gpu
    def test_trigger_code_initialization(self, target_device_idx, xp):
        """Test that trigger_code initializes nslopes correctly on first iteration"""
        calibrator = MultiImCalibrator(
            nmodes=5, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(3, target_device_idx=target_device_idx)
        slopes2 = Slopes(4, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        calibrator.setup()
        
        # Check initial state - nslopes starts at 0, not None
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].nslopes, 0)
        self.assertEqual(calibrator.outputs['out_intmat_list'][1].nslopes, 0)
        
        # Trigger first iteration
        calibrator.trigger_code()
        
        # Check that nslopes was set
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].nslopes, 3)
        self.assertEqual(calibrator.outputs['out_intmat_list'][1].nslopes, 4)

    @cpu_and_gpu
    def test_trigger_code_mode_processing(self, target_device_idx, xp):
        """Test that trigger_code processes modes correctly"""
        calibrator = MultiImCalibrator(
            nmodes=3, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(2, target_device_idx=target_device_idx)
        slopes2 = Slopes(2, target_device_idx=target_device_idx)
        
        # Set specific slope values
        slopes1.slopes = xp.array([1.0, 2.0])
        slopes2.slopes = xp.array([3.0, 4.0])
        
        # Create commands with specific mode activations
        cmd1 = BaseValue(value=xp.array([0.0, 5.0, 0.0]), target_device_idx=target_device_idx)  # Mode 1
        cmd2 = BaseValue(value=xp.array([0.0, 0.0, 10.0]), target_device_idx=target_device_idx)  # Mode 2
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        calibrator.setup()
        
        # Trigger processing
        calibrator.trigger_code()
        
        # Check that mode 1 was processed for intmat 0
        expected_mode1 = slopes1.slopes / 5.0
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][0].modes[1]), 
            cpuArray(expected_mode1)
        )
        
        # Check that mode 2 was processed for intmat 1
        expected_mode2 = slopes2.slopes / 10.0
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][1].modes[2]), 
            cpuArray(expected_mode2)
        )
        
        # Check command counts
        self.assertEqual(calibrator.count_commands[0][1], 1)  # Mode 1 for input 0
        self.assertEqual(calibrator.count_commands[1][2], 1)  # Mode 2 for input 1

    @cpu_and_gpu
    def test_trigger_code_multiple_iterations(self, target_device_idx, xp):
        """Test that trigger_code accumulates results over multiple iterations"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=1, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        slopes.slopes = xp.array([1.0, 2.0])
        
        cmd = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)  # Mode 0
        
        calibrator.inputs['in_slopes_list'].set([slopes])
        calibrator.inputs['in_commands_list'].set([cmd])
        calibrator.setup()
        
        # First iteration
        calibrator.trigger_code()
        
        # Check first result
        expected_first = slopes.slopes / 5.0
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][0].modes[0]), 
            cpuArray(expected_first)
        )
        self.assertEqual(calibrator.count_commands[0][0], 1)
        
        # Second iteration with same slopes
        calibrator.trigger_code()
        
        # Check accumulated result
        expected_accumulated = (slopes.slopes / 5.0) * 2
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][0].modes[0]), 
            cpuArray(expected_accumulated)
        )
        self.assertEqual(calibrator.count_commands[0][0], 2)

    @cpu_and_gpu
    def test_trigger_code_invalid_mode_ignored(self, target_device_idx, xp):
        """Test that trigger_code ignores commands with invalid mode indices"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=1, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        slopes.slopes = xp.array([1.0, 2.0])
        
        # Command with mode index >= nmodes
        cmd = BaseValue(value=xp.array([0.0, 0.0, 5.0]), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes])
        calibrator.inputs['in_commands_list'].set([cmd])
        calibrator.setup()
        
        # Should not raise error, just ignore invalid mode
        calibrator.trigger_code()
        
        # Check that no processing occurred
        self.assertEqual(calibrator.count_commands[0][0], 0)
        self.assertEqual(calibrator.count_commands[0][1], 0)

    @cpu_and_gpu
    def test_finalize_normalization(self, target_device_idx, xp):
        """Test that finalize normalizes interaction matrices by command counts"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=1, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        slopes.slopes = xp.array([1.0, 2.0])
        
        cmd = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)  # Mode 0
        
        calibrator.inputs['in_slopes_list'].set([slopes])
        calibrator.inputs['in_commands_list'].set([cmd])
        calibrator.setup()
        
        # Process multiple times
        for _ in range(3):
            calibrator.trigger_code()
        
        # Check accumulated result before finalize
        expected_accumulated = (slopes.slopes / 5.0) * 3
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][0].modes[0]), 
            cpuArray(expected_accumulated)
        )
        self.assertEqual(calibrator.count_commands[0][0], 3)
        
        # Finalize
        calibrator.finalize()
        
        # Check normalized result
        expected_normalized = slopes.slopes / 5.0
        np.testing.assert_array_almost_equal(
            cpuArray(calibrator.outputs['out_intmat_list'][0].modes[0]), 
            cpuArray(expected_normalized)
        )

    @cpu_and_gpu
    def test_finalize_file_saving(self, target_device_idx, xp):
        """Test that finalize saves files when paths are specified"""
        im_tag = 'test_save_im'
        full_im_tag = 'test_save_full_im'
        
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            im_tag=im_tag,
            full_im_tag=full_im_tag,
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(2, target_device_idx=target_device_idx)
        slopes2 = Slopes(2, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.array([0.0, 10.0]), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        calibrator.setup()
        
        # Process some data
        calibrator.trigger_code()
        
        # Finalize and save
        calibrator.finalize()
        
        # Check that files were created
        im_path0 = os.path.join(self.test_dir, f'{im_tag}0.fits')
        im_path1 = os.path.join(self.test_dir, f'{im_tag}1.fits')
        full_im_path = os.path.join(self.test_dir, f'{full_im_tag}.fits')
        
        self.assertTrue(os.path.exists(im_path0))
        self.assertTrue(os.path.exists(im_path1))
        self.assertTrue(os.path.exists(full_im_path))

    @cpu_and_gpu
    def test_finalize_no_file_saving(self, target_device_idx, xp):
        """Test that finalize doesn't save files when paths are not specified"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=1, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes])
        calibrator.inputs['in_commands_list'].set([cmd])
        calibrator.setup()
        
        # Process some data
        calibrator.trigger_code()
        
        # Finalize (should not save files)
        calibrator.finalize()
        
        # Check that no files were created
        files_created = [f for f in os.listdir(self.test_dir) if f.endswith('.fits')]
        self.assertEqual(len(files_created), 0)

    @cpu_and_gpu
    def test_finalize_full_im_generation(self, target_device_idx, xp):
        """Test that finalize generates full interaction matrix correctly"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            full_im_tag='test_full_im',
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(2, target_device_idx=target_device_idx)
        slopes2 = Slopes(2, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)
        cmd2 = BaseValue(value=xp.array([0.0, 10.0]), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        calibrator.setup()
        
        # Process some data
        calibrator.trigger_code()
        
        # Finalize
        calibrator.finalize()
        
        # Check that full IM was generated
        full_im = calibrator.outputs['out_intmat_full'].intmat
        self.assertIsNotNone(full_im)
        self.assertEqual(full_im.shape, (4, 2))  # 2 slopes * 2 inputs, 2 modes

    @cpu_and_gpu
    def test_finalize_empty_intmat_list(self, target_device_idx, xp):
        """Test that finalize handles empty intmat list correctly"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=0, 
            data_dir=self.test_dir, 
            full_im_tag='test_empty',
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        # Finalize with empty list
        calibrator.finalize()
        
        # Check that full IM is empty array
        full_im = calibrator.outputs['out_intmat_full'].intmat
        self.assertEqual(len(full_im), 0)

    @cpu_and_gpu
    def test_generation_time_updates(self, target_device_idx, xp):
        """Test that generation_time is updated correctly throughout processing"""
        calibrator = MultiImCalibrator(
            nmodes=2, 
            n_inputs=1, 
            data_dir=self.test_dir, 
            full_im_tag = 'test_full_im',
            overwrite=True,

            target_device_idx=target_device_idx
        )
        
        slopes = Slopes(2, target_device_idx=target_device_idx)
        cmd = BaseValue(value=xp.array([5.0, 0.0]), target_device_idx=target_device_idx)
        
        calibrator.inputs['in_slopes_list'].set([slopes])
        calibrator.inputs['in_commands_list'].set([cmd])
        calibrator.setup()
        
        # Set current time
        slopes.generation_time = 100
        cmd.generation_time = 100
        
        # Trigger processing
        calibrator.check_ready(100)
        calibrator.prepare_trigger(100)
        calibrator.trigger_code()
        calibrator.post_trigger()
        
        # Check that generation time was updated
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].generation_time, 100)
        
        # Update time and finalize
        calibrator.current_time = 200
        calibrator.finalize()
        
        # Check that generation time was updated again
        self.assertEqual(calibrator.outputs['out_intmat_list'][0].generation_time, 200)
        self.assertEqual(calibrator.outputs['out_intmat_full'].generation_time, 200)

    @cpu_and_gpu
    def test_count_commands_initialization(self, target_device_idx, xp):
        """Test that count_commands is properly initialized"""
        calibrator = MultiImCalibrator(
            nmodes=3, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        # Check initialization
        self.assertEqual(len(calibrator.count_commands), 2)  # n_inputs
        self.assertEqual(len(calibrator.count_commands[0]), 3)  # nmodes
        self.assertEqual(len(calibrator.count_commands[1]), 3)  # nmodes
        
        # Check all counts start at 0
        for input_counts in calibrator.count_commands:
            for count in input_counts:
                self.assertEqual(count, 0)

    @cpu_and_gpu
    def test_count_commands_tracking(self, target_device_idx, xp):
        """Test that count_commands properly tracks command usage"""
        calibrator = MultiImCalibrator(
            nmodes=3, 
            n_inputs=2, 
            data_dir=self.test_dir, 
            overwrite=True,
            target_device_idx=target_device_idx
        )
        
        slopes1 = Slopes(2, target_device_idx=target_device_idx)
        slopes2 = Slopes(2, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.array([5.0, 0.0, 0.0]), target_device_idx=target_device_idx)  # Mode 0
        cmd2 = BaseValue(value=xp.array([0.0, 0.0, 10.0]), target_device_idx=target_device_idx)  # Mode 2
        
        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1, cmd2])
        calibrator.setup()
        
        # Process multiple times
        for _ in range(3):
            calibrator.trigger_code()
        
        # Check command counts
        self.assertEqual(calibrator.count_commands[0][0], 3)  # Mode 0, input 0
        self.assertEqual(calibrator.count_commands[0][1], 0)  # Mode 1, input 0
        self.assertEqual(calibrator.count_commands[0][2], 0)  # Mode 2, input 0
        self.assertEqual(calibrator.count_commands[1][0], 0)  # Mode 0, input 1
        self.assertEqual(calibrator.count_commands[1][1], 0)  # Mode 1, input 1
        self.assertEqual(calibrator.count_commands[1][2], 3)  # Mode 2, input 1


if __name__ == '__main__':
    unittest.main()
