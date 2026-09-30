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
    def test_toccd_does_not_renormalize(self, target_device_idx, xp):
        """Test that the array is not renormalized when set_total <= 0."""
        arr = xp.arange(16, dtype=float).reshape((4, 4))
        total = arr.sum()
        out = toccd(arr, newshape=(2, 2), set_total = 0, xp=xp)
        assert xp.isclose(out.sum(), total/4, rtol=1e-6)


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
    def test_toccd_fp64_truncation_bug(self, target_device_idx, xp):
        """
        Verify that rebinning avoids float truncation errors 
        when dx_in creates inexact floats (like 1/5 = 0.2) in FP64.
        """
        # Let's use 5x5 and 3x3. LCM = 15. dx_in = 3. 1/3 = 0.3333...
        # We use progressive numbers to ensure that spatial order matters!
        arr_cpu = np.arange(25, dtype=np.float64).reshape((5, 5))

        # We calculate the "truth" using the CPU (numpy) since we know
        # it should be correct and not affected by GPU-specific issues.
        expected = toccd(arr_cpu, newshape=(3, 3), xp=np)
        
        # If we are testing the GPU, perform the calculation there and compare
        if xp == cp:
            arr_gpu = cp.array(arr_cpu)
            out_gpu = toccd(arr_gpu, newshape=(3, 3), xp=cp)
    
            # This is important to catch truncation errors that might not affect
            # the total sum but do affect the distribution of values.
            cp.testing.assert_allclose(out_gpu, cp.array(expected))

    @cpu_and_gpu
    def test_toccd_gpu_delegation(self, target_device_idx, xp):
        """If xp is cp, should call GPU version."""
        arr = xp.ones((4, 4), dtype=float)
        if xp == cp:
            out = toccd(arr, newshape=(2, 2), xp=xp)
            assert out.shape == (2, 2)
            assert xp.isclose(out.sum(), arr.sum(), rtol=1e-6)


    @staticmethod
    def _lcm_reference(a, newshape):
        '''oaalib's toccd() in float64: upsample to the L.C.M. and average'''
        from math import lcm
        (iny, inx), (outy, outx) = a.shape, newshape
        ly, lx = lcm(iny, outy), lcm(inx, outx)
        up = np.repeat(np.repeat(a.astype(np.float64), ly // iny, axis=0), lx // inx, axis=1)
        out = up.reshape(outy, ly // outy, outx, lx // outx).mean(axis=(1, 3))
        return out * a.sum() / out.sum()

    @cpu_and_gpu
    def test_toccd_matches_lcm_reference(self, target_device_idx, xp):
        """Non-square arrays, non-integer and prime size ratios, up and downsampling"""
        rng = np.random.default_rng(1)
        cases = [((80, 40), (20, 10)), ((40, 80), (10, 20)), ((60, 30), (40, 20)),
                 ((30, 60), (20, 40)), ((101, 101), (7, 7)), ((40, 40), (14, 14)),
                 ((7, 9), (20, 16))]
        for dtype, rtol in [(np.float32, 1e-5), (np.float64, 1e-12)]:
            for in_shape, out_shape in cases:
                with self.subTest(dtype=dtype, in_shape=in_shape, out_shape=out_shape):
                    a = rng.random(in_shape).astype(dtype)
                    out = toccd(xp.asarray(a), out_shape, xp=xp)
                    assert out.shape == out_shape
                    assert out.dtype == dtype
                    np.testing.assert_allclose(cpuArray(out), self._lcm_reference(a, out_shape),
                                               rtol=rtol)

    @cpu_and_gpu
    def test_toccd_strided_input(self, target_device_idx, xp):
        """A non-contiguous view gives the same result as its contiguous copy"""
        a = xp.asarray(np.random.default_rng(2).random((100, 100)), dtype=xp.float32)
        view = a[10:70, 10:70]
        np.testing.assert_allclose(cpuArray(toccd(view, (20, 20), xp=xp)),
                                   cpuArray(toccd(view.copy(), (20, 20), xp=xp)), rtol=1e-6)

    @cpu_and_gpu
    def test_toccd_out(self, target_device_idx, xp):
        """The result is written in out (also a view), with the same values"""
        a = xp.asarray(np.random.default_rng(3).random((60, 45)), dtype=xp.float32)
        for set_total in [None, 0, 10.0]:
            with self.subTest(set_total=set_total):
                expected = cpuArray(toccd(a, (14, 9), set_total=set_total, xp=xp))
                out = xp.empty((14, 9), dtype=xp.float32)
                assert toccd(a, (14, 9), set_total=set_total, xp=xp, out=out) is out
                np.testing.assert_allclose(cpuArray(out), expected, rtol=1e-6)

                frame = xp.zeros((20, 15), dtype=xp.float32)
                toccd(a, (14, 9), set_total=set_total, xp=xp, out=frame[3:17, 2:11])
                np.testing.assert_allclose(cpuArray(frame[3:17, 2:11]), expected, rtol=1e-6)
                frame[3:17, 2:11] = 0
                assert not frame.any()

        # Same shape: copied to out
        out = xp.empty((60, 45), dtype=xp.float32)
        toccd(a, (60, 45), xp=xp, out=out)
        np.testing.assert_array_equal(cpuArray(out), cpuArray(a))

        with self.assertRaises(ValueError):
            toccd(a, (14, 9), xp=xp, out=xp.empty((9, 14), dtype=xp.float32))
        if xp is cp:
            with self.assertRaises(TypeError):
                toccd(a, (14, 9), xp=xp, out=xp.empty((14, 9), dtype=xp.float64))
