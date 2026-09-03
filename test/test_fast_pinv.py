'''
Tests for specula.lib.fast_pinv.

The two rank-deficient-matrix tests below run on CPU (numpy) only. This is
not a limitation of fast_pinv itself: cupy's own linalg.pinv/svd already
fail on this exact input even called directly, with no trick involved --
for the 3x5 float32 matrix used here (rank 2), cupy genuinely computes its
SVD in float32 and resolves the smallest singular value as ~2.6e-7 (the
expected float32 noise floor), instead of numpy's ~2.8e-16. That numpy
figure is not a fairer answer: numpy.linalg.svd/pinv silently compute in
float64 internally regardless of input dtype (see
numpy.linalg.linalg._commonType, "in lite version, use higher precision"),
only casting the result back down -- confirmed across reference BLAS,
OpenBLAS and numpy 2.x. Forcing a genuine float32 SVD on CPU (e.g. via
scipy with an explicit lapack_driver) reproduces the same ~1e-7 residual
*and* the same six-orders-of-magnitude pinv blowup that cupy shows. So
this is a real float32-precision fragility of pinv on near-singular
input, not a GPU-specific limitation -- numpy just normally hides it.
'''
import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula import cpuArray
from specula.lib.fast_pinv import fast_pinv

from test.specula_testlib import cpu_and_gpu


class TestFastPinv(unittest.TestCase):

    @cpu_and_gpu
    def test_wide_matrix_matches_direct_pinv(self, target_device_idx, xp):
        '''Typical influence-function shape: far more columns (pixels) than rows (modes).'''
        a = xp.asarray([[1., 2., 3., 4., 5.],
                         [2., -1., 0., 3., 1.]], dtype=xp.float32)

        expected = xp.linalg.pinv(a)
        result = fast_pinv(a, xp)

        self.assertEqual(result.shape, (5, 2))
        np.testing.assert_array_almost_equal(cpuArray(expected), cpuArray(result))

    @cpu_and_gpu
    def test_tall_matrix_matches_direct_pinv(self, target_device_idx, xp):
        '''The opposite (unusual for influence functions) shape: more rows than columns.'''
        a = xp.asarray([[1., 2.],
                         [2., -1.],
                         [0., 3.],
                         [4., 1.],
                         [5., 2.]], dtype=xp.float32)

        expected = xp.linalg.pinv(a)
        result = fast_pinv(a, xp)

        self.assertEqual(result.shape, (2, 5))
        np.testing.assert_array_almost_equal(cpuArray(expected), cpuArray(result))

    @cpu_and_gpu
    def test_square_matrix_matches_direct_pinv(self, target_device_idx, xp):
        a = xp.asarray([[4., 1., 0.],
                         [1., 3., 1.],
                         [0., 1., 2.]], dtype=xp.float32)

        expected = xp.linalg.pinv(a)
        result = fast_pinv(a, xp)

        np.testing.assert_array_almost_equal(cpuArray(expected), cpuArray(result))

    def test_rank_deficient_wide_matrix_cpu(self):
        '''Row 3 is a linear combination of rows 1 and 2: A A^T is singular,
        so this exercises the pinv-of-a-singular-Gram-matrix path.

        CPU (numpy) only -- see the module docstring for why this is
        not also run on GPU.
        '''
        a = np.asarray([[1., 2., 3., 4., 5.],
                         [2., -1., 0., 3., 1.],
                         [3., 1., 3., 7., 6.]], dtype=np.float32)  # row0 + row1

        expected = np.linalg.pinv(a)
        result = fast_pinv(a, np)

        np.testing.assert_array_almost_equal(expected, result, decimal=6)

    def test_pinv_is_a_valid_pseudoinverse_rank_deficient_cpu(self):
        '''Independently of the reference np.linalg.pinv, check the defining
        Moore-Penrose property A @ A+ @ A == A on a rank-deficient case.

        CPU (numpy) only -- see the module docstring for why this is
        not also run on GPU.
        '''
        a = np.asarray([[1., 2., 3., 4., 5.],
                         [2., -1., 0., 3., 1.],
                         [3., 1., 3., 7., 6.]], dtype=np.float32)

        result = fast_pinv(a, np)
        reconstructed = a @ result @ a
        np.testing.assert_array_almost_equal(a, reconstructed, decimal=5)
