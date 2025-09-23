import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.utils import local_mean_rebin
from test.specula_testlib import cpu_and_gpu

class TestLocalMeanRebin(unittest.TestCase):

    @cpu_and_gpu
    def test_local_mean_rebin_basic(self, target_device_idx, xp):
        """Basic functionality test"""
        arr = xp.arange(16, dtype=float).reshape(4, 4)
        mask = xp.ones_like(arr, dtype=bool)
        block_size = 2
        result = local_mean_rebin(arr, mask, xp, block_size=block_size)
        # Each 2x2 block should have the mean of its values
        expected = xp.array([
            [2.5, 2.5, 4.5, 4.5],
            [2.5, 2.5, 4.5, 4.5],
            [10.5, 10.5, 12.5, 12.5],
            [10.5, 10.5, 12.5, 12.5]
        ])
        np.testing.assert_allclose(cpuArray(result), cpuArray(expected))

    @cpu_and_gpu
    def test_local_mean_rebin_with_invalid_pixels(self, target_device_idx, xp):
        """Test with some invalid pixels"""
        arr = xp.arange(16, dtype=float).reshape(4, 4)
        mask = xp.ones_like(arr, dtype=bool)
        mask[0, 0] = False  # Make one pixel invalid
        block_size = 2
        result = local_mean_rebin(arr, mask, xp, block_size=block_size)
        # The mean for the top-left block should ignore arr[0,0]
        expected_mean = cpuArray(xp.mean(xp.array([arr[0,1], arr[1,0], arr[1,1]])))
        self.assertAlmostEqual(result[0,0], expected_mean)
        self.assertAlmostEqual(result[0,1], expected_mean)
        self.assertAlmostEqual(result[1,0], expected_mean)
        self.assertAlmostEqual(result[1,1], expected_mean)

    @cpu_and_gpu
    def test_local_mean_rebin_empty_block(self, target_device_idx, xp):
        """Test with a block that has all invalid pixels"""
        arr = xp.arange(16, dtype=float).reshape(4, 4)
        mask = xp.zeros_like(arr, dtype=bool)
        # All pixels invalid: result should be filled with global mean (which is nan)
        block_size = 2
        result = local_mean_rebin(arr, mask, xp, block_size=block_size)
        # Since mask is all False, global_mean is nan, so result should be all nan
        self.assertTrue(np.all(np.isnan(cpuArray(result))))