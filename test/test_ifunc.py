
import specula
specula.init(0)  # Default target device

import os
import numpy as np
import unittest

from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.lib.make_mask import make_mask

from test.specula_testlib import cpu_and_gpu

class TestIFunv(unittest.TestCase):

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        self.mask = np.array([[0, 1, 0], [0, 1, 0]], dtype=int)

        self.inv_data = np.array([[-0.94444444,  0.44444444],
                             [-0.11111111,  0.11111111],
                             [ 0.72222222, -0.22222222]])

        self.inv_filename = os.path.join(self.datadir, 'ifunc_inv.fits')
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    def tearDown(self):
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_ifunc_inv_data(self, target_device_idx, xp):
        '''Test that the inversion in IFunc is correct'''
        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv = ifunc.inverse()
        assert isinstance(inv, IFuncInv)

        np.testing.assert_array_almost_equal(cpuArray(self.inv_data), cpuArray(inv.ifunc_inv))

    @cpu_and_gpu
    def test_ifunc_inv_idx(self, target_device_idx, xp):
        '''Test that the mask indexes in the inverted ifunc are the same as in IFunc'''

        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv = ifunc.inverse()

        idx1 = cpuArray(ifunc.idx_inf_func[0]), cpuArray(ifunc.idx_inf_func[1])
        idx2 = cpuArray(inv.idx_inf_func[0]), cpuArray(inv.idx_inf_func[1])
        np.testing.assert_array_equal(idx1[0], idx2[0])
        np.testing.assert_array_equal(idx1[1], idx2[1])

    @cpu_and_gpu
    def test_ifunc_inv_restore(self, target_device_idx, xp):
        '''Test that data saved and restored is the same as data obtained from IFunc.inverse()'''

        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

        inv = IFuncInv(self.inv_data, mask=self.mask, target_device_idx=target_device_idx)
        inv.save(self.inv_filename)

        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv1 = ifunc.inverse()
        inv2 = IFuncInv.restore(self.inv_filename)

        np.testing.assert_array_almost_equal(cpuArray(inv1.ifunc_inv), cpuArray(inv2.ifunc_inv))
        idx1 = cpuArray(inv1.idx_inf_func[0]), cpuArray(inv1.idx_inf_func[1])
        idx2 = cpuArray(inv2.idx_inf_func[0]), cpuArray(inv2.idx_inf_func[1])
        np.testing.assert_array_equal(idx1[0], idx2[0])
        np.testing.assert_array_equal(idx1[1], idx2[1])

    @cpu_and_gpu
    def test_ifunc_2d_to_3d(self, target_device_idx, xp):
        '''Test for ifunc_2d_to_3d method'''
        mask = make_mask(64)
        ifunc = IFunc(type_str='zernike', mask=mask, nmodes=3, npixels=64, target_device_idx=target_device_idx)
        data = ifunc.influence_function
        result_3d = ifunc.ifunc_2d_to_3d(normalize=False)
        result_3d_cpu = cpuArray(result_3d)

        # Check that values are placed only where mask > 0
        mask_idx = np.where(mask > 0)

        # Result should have the same dtype as the ifunc
        self.assertEqual(result_3d.dtype, ifunc.dtype)

        # Expected shape: (npixels, npixels, nmodes)
        expected_shape = (mask.shape[0], mask.shape[1], data.shape[0])
        self.assertEqual(result_3d.shape, expected_shape)

        # For each mode, check values at masked positions
        for mode in range(data.shape[0]):
            # Get values at masked positions for this mode
            values_at_mask = result_3d_cpu[mask_idx[0], mask_idx[1], mode]
            expected_values = data[mode, :]  # All pixels for this mode
            np.testing.assert_array_equal(cpuArray(values_at_mask), cpuArray(expected_values))

        # Check that values are zero where mask == 0
        mask_zero_idx = np.where(mask == 0)
        for mode in range(data.shape[0]):
            values_at_zero = result_3d_cpu[mask_zero_idx[0], mask_zero_idx[1], mode]
            expected_zeros = np.zeros(len(mask_zero_idx[0]))
            np.testing.assert_array_equal(cpuArray(values_at_zero), expected_zeros)

        # Test with normalization
        result_norm = ifunc.ifunc_2d_to_3d(normalize=True)

        # Calculate expected RMS values for each mode
        expected_rms = np.sqrt(np.mean(cpuArray(data)**2, axis=1))

        # Check that normalized result has correct scaling
        mask_idx = np.where(mask > 0)

        for mode in range(data.shape[0]):
            values_no_norm = cpuArray(result_3d[mask_idx[0], mask_idx[1], mode])
            values_norm = cpuArray(result_norm[mask_idx[0], mask_idx[1], mode])

            # Normalized values should equal non-normalized values divided by RMS
            expected_normalized = values_no_norm / expected_rms[mode]
            np.testing.assert_array_almost_equal(values_norm, expected_normalized)