import specula
specula.init(0)  # Default target device

import os
import unittest
import tempfile
import shutil
import numpy as np

from specula import cpuArray
from test.specula_testlib import cpu_and_gpu
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat
from specula.processing_objects.psd_calibrator import PSDCalibrator

class TestPSDCalibrator(unittest.TestCase):

    def setUp(self):
        """Set up a temporary workspace and mock data objects."""
        self.test_dir = tempfile.mkdtemp()
        self.n_slopes = 12
        self.n_modes = 8
        self.mock_rec = Recmat(recmat=np.random.rand(self.n_modes, self.n_slopes))

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    @cpu_and_gpu
    def test_recmat_loading(self, target_device_idx, xp):
        """Verify that the calibrator correctly loads and stores the recmat."""
        calibrator = PSDCalibrator(data_dir=self.test_dir, recmat=self.mock_rec,
                                        target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(calibrator.rec), cpuArray(self.mock_rec.recmat))

    @cpu_and_gpu
    def test_save_and_overwrite_logic(self, target_device_idx, xp):
        """Check that files are saved and overwrite = true works."""
        calib = PSDCalibrator(data_dir=self.test_dir, recmat=self.mock_rec, output_tag='test_save',
                                   overwrite=True, target_device_idx=target_device_idx)
        s = Slopes(self.n_slopes)
        calib.inputs['in_values'].set(s)
        s.generation_time = 1

        calib.setup()
        calib.check_ready(1)
        calib.trigger_code()

        alias_path = os.path.join(calib._data_dir,calib._filename+'.fits')
        with open(alias_path, 'w') as f:
            f.write('')

        with self.assertRaises(ValueError): # value error raised in finalize() as we are computing a PSD from 1 sample
            calib.finalize()

    # @cpu_and_gpu
    # def test_overwrite_raise(self, target_device_idx, xp):
    #     """Check that overwrite flags are respected."""
    #     tag = 'test_overwrite'
    #     calib = PSDCalibrator(data_dir=self.test_dir, recmat=self.mock_rec, output_tag=tag,
    #                                overwrite=False, target_device_idx=target_device_idx)
    #     s = Slopes(self.n_slopes)
    #     calib.inputs['in_slopes'].set(s)
    #     s.generation_time = 1

    #     calib.setup()
    #     calib.check_ready(1)
    #     calib.trigger_code()

    #     alias_path = os.path.join(self.test_dir,tag+'.fits')
    #     with open(alias_path, 'w') as f:
    #         f.write('')
            
    #     with self.assertRaises(FileExistsError):
    #         with self.assertRaises(ValueError): # value error raised in finalize() as we are computing a PSD from 1 sample
    #             calib.finalize()


    # TODO uncomment when LoopControl is used for triggering in tests
    # @cpu_and_gpu
    # def test_trigger_on_slope_update(self, target_device_idx, xp):
    #     """Ensure trigger logic correctly tracks slope updates"""
    #     calib = PSDCalibrator(data_dir=self.test_dir, 
    #                                recmat=self.mock_rec,
    #                                target_device_idx=target_device_idx)
    #     s = Slopes(self.n_slopes)
    #     calib.inputs['in_slopes'].set(s)
        
    #     # Manually verify trigger increases the internal iteration counter
    #     initial_count = calib._n_iter
    #     n_iterations = 10
    #     for i in range(n_iterations):
    #         s = Slopes(self.n_slopes)
    #         s.slopes = np.random.rand(self.n_slopes)
    #         s.generation_time = i if i%2==0 else 0
    #         calib.inputs['in_slopes'].set(s)
    #         calib.setup()
    #         calib.check_ready(i)
    #         calib.trigger_code()
    #         self.assertEqual(calib._n_iter, initial_count + i//2 + 1)
    #     self.assertEqual(len(calib.slopes_list), n_iterations//2)

    @cpu_and_gpu
    def test_finalize_shape_integrity(self, target_device_idx, xp):
        """Verify the output modal PSD shape matches the reconstruction matrix."""
        calib = PSDCalibrator(data_dir=self.test_dir, recmat=self.mock_rec, 
                                   overwrite=True, target_device_idx=target_device_idx)
        
        n_iterations = 5
        for i in range(n_iterations):
            s = Slopes(self.n_slopes)
            s.slopes = np.random.rand(self.n_slopes)
            s.generation_time = i
            calib.inputs['in_values'].set(s)
            calib.setup()
            calib.check_ready(i)
            calib.trigger_code()

        with self.assertRaises(ValueError): # value error raised in finalize() as we are computing a PSD from 5 samples 
            calib.finalize()
        slopes_thist = calib.to_xp(calib.values_list)
        projected = calib.rec @ slopes_thist.T
        self.assertEqual(projected.shape[0], self.n_modes)
        self.assertEqual(projected.shape[1], n_iterations)