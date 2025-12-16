import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray, cp
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

    @cpu_and_gpu
    def test_toccd_identity(self, target_device_idx, xp):
        """If input shape equals newshape, should return the same array."""
        arr = xp.ones((5, 5), dtype=float)
        out = toccd(arr, newshape=(5, 5), xp=xp)
        xp.testing.assert_array_equal(out, arr)


    @cpu_and_gpu
    def test_toccd_resizes_array(self, target_device_idx, xp):
        """Test that output has correct shape after resizing."""
        arr = xp.arange(16, dtype=float).reshape((4, 4))
        newshape = (2, 2)
        out = toccd(arr, newshape=newshape, xp=xp)
        assert out.shape == newshape


    @cpu_and_gpu
    def test_toccd_preserves_total_sum(self, target_device_idx, xp):
        """Test that total sum is preserved when set_total=None."""
        arr = xp.arange(16, dtype=float).reshape((4, 4))
        total = arr.sum()
        out = toccd(arr, newshape=(2, 2), xp=xp)
        assert xp.isclose(out.sum(), total, rtol=1e-6)


    @cpu_and_gpu
    def test_toccd_applies_set_total(self, target_device_idx, xp):
        """Test that total sum matches set_total if provided."""
        arr = xp.ones((4, 4), dtype=float)
        set_total = 10.0
        out = toccd(arr, newshape=(2, 2), set_total=set_total, xp=xp)
        assert xp.isclose(out.sum(), set_total, rtol=1e-6)


    @cpu_and_gpu
    def test_toccd_invalid_input_shape(self, target_device_idx, xp):
        """Should raise ValueError if input array is not 2D."""
        arr = xp.ones((4,), dtype=float)
        with self.assertRaises(ValueError):
            toccd(arr, newshape=(2, 2), xp=xp)


    @cpu_and_gpu
    def test_toccd_invalid_output_shape(self, target_device_idx, xp):
        """Should raise ValueError if newshape is not 2D."""
        arr = xp.ones((4, 4), dtype=float)
        with self.assertRaises(ValueError):
            toccd(arr, newshape=(2,), xp=xp)


    @cpu_and_gpu
    def test_toccd_dtype_float32(self, target_device_idx, xp):
        """Check that float32 input produces float32 output."""
        arr = xp.ones((4, 4), dtype=xp.float32)
        out = toccd(arr, newshape=(2, 2), xp=xp)
        assert out.dtype == xp.float32


    @cpu_and_gpu
    def test_toccd_dtype_float64(self, target_device_idx, xp):
        """Check that float64 input produces float64 output."""
        arr = xp.ones((4, 4), dtype=xp.float64)
        out = toccd(arr, newshape=(2, 2), xp=xp)
        assert out.dtype == xp.float64

    @cpu_and_gpu
    def test_toccd_gpu_delegation(self, target_device_idx, xp):
        """If xp is cp, should call GPU version."""
        arr = xp.ones((4, 4), dtype=float)
        if xp == cp:
            out = toccd(arr, newshape=(2, 2), xp=xp)
            assert out.shape == (2, 2)
            assert xp.isclose(out.sum(), arr.sum(), rtol=1e-6)

