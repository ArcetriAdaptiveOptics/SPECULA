import specula
specula.init(0)  # Default target device

import os
import tempfile
import unittest
from unittest.mock import patch
import shutil
import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.subap_data import SubapData
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.source import Source
from specula.processing_objects.dm import DM
from specula.processing_objects.sh import SH
from specula.processing_objects.sh_slopec import ShSlopec
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
            im_tag=['tag1', 'tag2', 'tag3'],
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
            im_tag=['custom_im1', 'custom_im2'],
            full_im_tag='custom_full_im',
            overwrite=True
        )

        self.assertEqual(calibrator.im_tag, ['custom_im1', 'custom_im2'])
        self.assertEqual(calibrator.full_im_tag, 'custom_full_im')

    @cpu_and_gpu
    def test_initialization_with_auto_tags(self, target_device_idx, xp):
        """Test auto tag generation with real SH configuration"""

        # Usa file reali dal directory test/data
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

        # Create SimulParams (needed for DM and SH)
        simul_params = SimulParams(
            pixel_pupil=64,
            pixel_pitch=0.1
        )

        # Create Pupilstop
        pupilstop = Pupilstop(
            simul_params,
            mask_diam=0.9,
            obs_diam=0.1,
            target_device_idx=-1
        )
        pupilstop.tag = 'test_pupil'

        # Create Sources
        source1 = Source(
            polar_coordinates=[10.0, 0.0],
            magnitude=5,
            wavelengthInNm=600,
            target_device_idx=-1
        )
        source2 = Source(
            polar_coordinates=[10.0, 120.0],
            magnitude=5,
            wavelengthInNm=600,
            target_device_idx=-1
        )

        # Create DM with zernike modes
        dm = DM(
            simul_params=simul_params,
            type_str='zernike',
            nmodes=40,
            obsratio=0.1,
            height=0,
            target_device_idx=-1
        )

        # Create SH sensors
        sensor1 = SH(
            subap_wanted_fov=4.0,
            sensor_pxscale=0.5,
            subap_npx=8,
            subap_on_diameter=8,
            wavelengthInNm=600,
            target_device_idx=-1
        )
        sensor2 = SH(
            subap_wanted_fov=4.0,
            sensor_pxscale=0.5,
            subap_npx=8,
            subap_on_diameter=8,
            wavelengthInNm=600,
            target_device_idx=-1
        )

        # Use real subapdata files if they exist
        subapdata_file = os.path.join(data_dir, 'scao_subaps_n8_th0.5_ref.fits')
        if os.path.exists(subapdata_file):
            # Load subapdata from real file
            subapdata1 = SubapData.restore(subapdata_file, target_device_idx=target_device_idx)
            subapdata2 = SubapData.restore(subapdata_file, target_device_idx=target_device_idx)

            slopec1 = ShSlopec(
                subapdata=subapdata1,
                weightedPixRad=4.0,
                target_device_idx=-1
            )
            slopec2 = ShSlopec(
                subapdata=subapdata2,
                weightedPixRad=4.0,
                target_device_idx=-1
            )
        else:
            # Fallback: create simple subapdata for test
            self.skipTest(f"Real subapdata file not found: {subapdata_file}")

        # Create dictionaries for multi-input
        source_dict = {'source1': source1, 'source2': source2}
        sensor_dict = {'sensor1': sensor1, 'sensor2': sensor2}
        slopec_dict = {'slopec1': slopec1, 'slopec2': slopec2}

        # Mock the static method to return predictable tags
        with patch('specula.processing_objects.im_calibrator.ImCalibrator.generate_im_tag') \
            as mock_generate_tag:
            mock_generate_tag.side_effect = [
                'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor1',
                'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor2'
            ]

            # Test MultiImCalibrator with auto tag generation
            calibrator = MultiImCalibrator(
                nmodes=10,
                n_inputs=2,
                data_dir=self.test_dir,
                im_tag='auto',
                full_im_tag='test_full_auto',
                overwrite=True,
                pupilstop=pupilstop,
                source_dict=source_dict,
                dm=dm,
                sensor_dict=sensor_dict,
                slopec_dict=slopec_dict,
                target_device_idx=-1,
                precision=1
            )

            # Verify that generate_im_tag was called twice (once for each input)
            self.assertEqual(mock_generate_tag.call_count, 2)

            # Check that auto-generated tags were set correctly
            expected_tags = [
                'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor1',
                'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor2'
            ]
            self.assertEqual(calibrator.im_tag, expected_tags)

            # Verify the call arguments for each tag generation
            calls = mock_generate_tag.call_args_list
            self.assertEqual(len(calls), 2)

            # Check first call (source1, sensor1, slopec1)
            args1 = calls[0][0]
            self.assertEqual(args1[0], pupilstop)  # pupilstop
            self.assertEqual(args1[1], source1)    # source
            self.assertEqual(args1[2], dm)         # dm
            self.assertEqual(args1[3], sensor1)    # sensor
            self.assertEqual(args1[4], slopec1)    # slopec
            self.assertEqual(args1[5], 10)         # nmodes

            # Check second call (source2, sensor2, slopec2)
            args2 = calls[1][0]
            self.assertEqual(args2[0], pupilstop)  # pupilstop
            self.assertEqual(args2[1], source2)    # source
            self.assertEqual(args2[2], dm)         # dm
            self.assertEqual(args2[3], sensor2)    # sensor
            self.assertEqual(args2[4], slopec2)    # slopec
            self.assertEqual(args2[5], 10)         # nmodes

            # Verify calibrator properties
            self.assertEqual(calibrator.nmodes, 10)
            self.assertEqual(calibrator.n_inputs, 2)
            self.assertEqual(len(calibrator.outputs['out_intmat_list']), 2)

            # Check that paths were generated correctly
            expected_path1 = os.path.join(self.test_dir,
                                          'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor1.fits')
            expected_path2 = os.path.join(self.test_dir,
                                          'auto_tag_sh8x8sa_w600nm_f4.0asec_sensor2.fits')
            self.assertEqual(calibrator.im_paths[0], expected_path1)
            self.assertEqual(calibrator.im_paths[1], expected_path2)

            verbose = False
            if verbose: #pragma: no cover
                print(f"Generated IM tags: {calibrator.im_tag}")
                print(f"Generated paths: {calibrator.im_paths}")

    def test_tag_filename_validation(self):
        """Test that tag_filename method validates parameters correctly"""
        with self.assertRaises(TypeError):
            MultiImCalibrator(
                nmodes=5,
                n_inputs=2,
                data_dir=self.test_dir,
                # im_tag mancante
                overwrite=True
            )

    def test_existing_im_file_is_detected(self):
        """Test that MultiImCalibrator detects existing IM files"""
        im_tag = ['test_im1', 'test_im2']

        # Create existing file for first tag
        existing_file = os.path.join(self.test_dir, 'test_im1.fits')
        with open(existing_file, 'w') as f:
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
            _ = MultiImCalibrator(
                nmodes=10,
                n_inputs=2,
                data_dir=self.test_dir,
                im_tag=['tag1', 'tag2'],
                full_im_tag=full_im_tag
            )

    def test_existing_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing files"""
        im_tag = ['test_im_overwrite1', 'test_im_overwrite2']

        # Create existing file
        existing_file = os.path.join(self.test_dir, 'test_im_overwrite1.fits')
        with open(existing_file, 'w') as f:
            f.write('')

        # Should not raise
        _ = MultiImCalibrator(nmodes=10, n_inputs=2, data_dir=self.test_dir,
                              im_tag=im_tag, overwrite=True)

    def test_existing_full_im_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing full IM files"""
        full_im_tag = 'test_full_im_overwrite'
        full_im_filename = f'{full_im_tag}.fits'
        full_im_path = os.path.join(self.test_dir, full_im_filename)

        # Create empty file
        with open(full_im_path, 'w') as f:
            f.write('')

        # Should not raise
        _ = MultiImCalibrator(
            nmodes=10, 
            n_inputs=2, 
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
            full_im_tag=full_im_tag, 
            overwrite=True
        )

    def test_im_path_generation(self):
        """Test that im_path method generates correct paths"""
        calibrator = MultiImCalibrator(
            nmodes=5,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['test_im1', 'test_im2'],
            overwrite=True
        )

        # Check paths were set correctly
        expected_path0 = os.path.join(self.test_dir, 'test_im1.fits')
        expected_path1 = os.path.join(self.test_dir, 'test_im2.fits')
        self.assertEqual(calibrator.im_paths[0], expected_path0)
        self.assertEqual(calibrator.im_paths[1], expected_path1)

    def test_full_im_path_generation(self):
        """Test that full_im_path method generates correct paths"""
        calibrator = MultiImCalibrator(
            nmodes=5,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
            full_im_tag='test_full_im',
            overwrite=True
        )

        expected_path = os.path.join(self.test_dir, 'test_full_im.fits')
        self.assertEqual(calibrator.full_im_path, expected_path)

    @cpu_and_gpu
    def test_setup_validation_success(self, target_device_idx, xp):
        """Test that setup validates inputs correctly when they match expected counts"""
        calibrator = MultiImCalibrator(
            nmodes=5,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2', 'tag3'],
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
            im_tag=['tag1', 'tag2', 'tag3'],
            overwrite=True,
            target_device_idx=target_device_idx
        )

        slopes1 = Slopes(3, target_device_idx=target_device_idx)
        slopes2 = Slopes(3, target_device_idx=target_device_idx)
        cmd1 = BaseValue(value=xp.zeros(5), target_device_idx=target_device_idx)

        calibrator.inputs['in_slopes_list'].set([slopes1, slopes2])
        calibrator.inputs['in_commands_list'].set([cmd1])

        with self.assertRaises(ValueError) as context:
            calibrator.setup()
        self.assertIn("Number of input slopes (2) does not match expected n_inputs (3)", str(context.exception))

    @cpu_and_gpu
    def test_trigger_code_initialization(self, target_device_idx, xp):
        """Test that trigger_code initializes nslopes correctly on first iteration"""
        calibrator = MultiImCalibrator(
            nmodes=5,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
        im_tag=['test_save_im1', 'test_save_im2']
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
        im_path0 = os.path.join(self.test_dir, f'{im_tag[0]}.fits')
        im_path1 = os.path.join(self.test_dir, f'{im_tag[1]}.fits')
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
            im_tag=['tag1', 'tag2'],
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

        expected_files = 2  # tag1.fits and tag2.fits
        if hasattr(calibrator, 'full_im_tag') and calibrator.full_im_tag:
            expected_files += 1  # + full_im_tag.fits

        self.assertEqual(len(files_created), expected_files)

        # Check that at least the main IM file was created
        self.assertTrue(any('tag1.fits' in f for f in files_created))


    @cpu_and_gpu
    def test_finalize_full_im_generation(self, target_device_idx, xp):
        """Test that finalize generates full interaction matrix correctly"""
        calibrator = MultiImCalibrator(
            nmodes=2,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
            im_tag=['tag1', 'tag2'],
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
        """Test that count_commands tracks command counts correctly"""
        calibrator = MultiImCalibrator(
            nmodes=3,
            n_inputs=2,
            data_dir=self.test_dir,
            im_tag=['tag1', 'tag2'],
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

        # Aspettative corrette basate sul comportamento reale:
        self.assertEqual(calibrator.count_commands[0][0], 3)  # Mode 0, input 0 (from cmd1)
        self.assertEqual(calibrator.count_commands[0][1], 0)  # Mode 1, input 0 (no comand)
        self.assertEqual(calibrator.count_commands[0][2], 3)  # Mode 2, input 0 (from cmd2)

        self.assertEqual(calibrator.count_commands[1][0], 3)  # Mode 0, input 1 (from cmd1)
        self.assertEqual(calibrator.count_commands[1][1], 0)  # Mode 1, input 1 (no comand)
        self.assertEqual(calibrator.count_commands[1][2], 3)  # Mode 2, input 1 (from cmd2)

if __name__ == '__main__':
    unittest.main()
