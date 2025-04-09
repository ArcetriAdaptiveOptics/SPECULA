

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.utils import unravel_index_2d

from test.specula_testlib import cpu_and_gpu

class TestUtils(unittest.TestCase):
   
    @cpu_and_gpu
    def test_unravel_index_square_shape(self, target_device_idx, xp):
        
        idxs = xp.array([1,2,3])
        shape = (3,3)
        y, x = unravel_index_2d(idxs, shape, xp) 
        ytest, xtest = xp.unravel_index(idxs, shape)
        np.testing.assert_array_almost_equal(cpuArray(x), cpuArray(xtest))
        np.testing.assert_array_almost_equal(cpuArray(y), cpuArray(ytest))

    @cpu_and_gpu
    def test_unravel_index_rectangular_shape(self, target_device_idx, xp):
       
        idxs = xp.array([2,6,13])
        shape = (4,8)
        y, x = unravel_index_2d(idxs, shape, xp) 
        ytest, xtest = xp.unravel_index(idxs, shape)
        np.testing.assert_array_almost_equal(cpuArray(x), cpuArray(xtest))
        np.testing.assert_array_almost_equal(cpuArray(y), cpuArray(ytest))

    @cpu_and_gpu
    def test_unravel_index_wrong_shape(self, target_device_idx, xp):

        with self.assertRaises(ValueError):
            _ = unravel_index_2d([1,2,3], (1,2,3), xp)