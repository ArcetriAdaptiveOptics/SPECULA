import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.lib.toccd import toccd
from test.specula_testlib import cpu_and_gpu

class TestToccd(unittest.TestCase):

    @cpu_and_gpu
    def test_toccd_shape_inputs(self, target_device_idx, xp):
        
        a = xp.arange(9).reshape((3,3)).astype(xp.float32)

        a1 = toccd(a, newshape=(2, 2), xp=xp)  # tuple
        a2 = toccd(a, newshape=xp.array((2, 2)), xp=xp)   # numpy/cupy array
        a3 = toccd(a, newshape=[2, 2], xp=xp)  # list

        np.testing.assert_array_equal(cpuArray(a1), cpuArray(a2))
        np.testing.assert_array_equal(cpuArray(a1), cpuArray(a3))

