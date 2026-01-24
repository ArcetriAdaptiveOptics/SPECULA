import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula import cpuArray
from specula.data_objects.ssr_filter_data import SsrFilterData
from test.specula_testlib import cpu_and_gpu


class TestSsrFilterData(unittest.TestCase):
    """Test suite for SsrFilterData"""

    @cpu_and_gpu
    def test_init_basic(self, target_device_idx, xp):
        """Test basic initialization with single filter"""
        A = xp.array([[0.9]])
        B = xp.array([[0.1]])
        C = xp.array([[1.0]])
        D = xp.array([[0.0]])

        ssr_data = SsrFilterData(A, B, C, D,
                                 target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 1)
        self.assertEqual(ssr_data.total_states, 1)

    @cpu_and_gpu
    def test_init_with_n_modes_expansion(self, target_device_idx, xp):
        """Test n_modes expansion"""
        A = xp.array([[0.9]])
        B = xp.array([[0.1]])
        C = xp.array([[1.0]])
        D = xp.array([[0.0]])

        n_modes = [3, 2]
        ssr_data = SsrFilterData([A, A], [B, B], [C, C], [D, D], 
                                n_modes=n_modes,
                                target_device_idx=target_device_idx)

        # Should have 5 filters total (3 + 2)
        self.assertEqual(ssr_data.nfilter, 5)

    @cpu_and_gpu
    def test_dimension_validation(self, target_device_idx, xp):
        """Test that dimension validation catches errors"""
        A = xp.array([[0.9]])
        B = xp.array([[0.1, 0.2]])  # Wrong shape - 2 inputs instead of 1
        C = xp.array([[1.0]])
        D = xp.array([[0.0]])

        with self.assertRaises(ValueError):
            SsrFilterData(A, B, C, D, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_from_gain(self, target_device_idx, xp):
        """Test from_gain factory method"""
        gains = [0.5, 1.0, 2.0]
        ssr_data = SsrFilterData.from_gain(gains, target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)

        # Test that it implements y = gain * u (pure feedthrough)
        # D is now a diagonal matrix, not individual matrices
        D_cpu = cpuArray(ssr_data.D)
        for i in range(3):
            np.testing.assert_almost_equal(D_cpu[i, i], gains[i])

    @cpu_and_gpu
    def test_from_integrator(self, target_device_idx, xp):
        """Test from_integrator factory method"""
        gains = [0.5, 1.0]
        ssr_data = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 2)

        # Check block-diagonal structure
        A_cpu = cpuArray(ssr_data.A)
        B_cpu = cpuArray(ssr_data.B)
        C_cpu = cpuArray(ssr_data.C)
        D_cpu = cpuArray(ssr_data.D)

        # A should be identity block-diagonal (ff=1.0)
        np.testing.assert_almost_equal(A_cpu[0, 0], 1.0)
        np.testing.assert_almost_equal(A_cpu[1, 1], 1.0)

        # B should have gains on diagonal of each filter's column
        np.testing.assert_almost_equal(B_cpu[0, 0], gains[0])
        np.testing.assert_almost_equal(B_cpu[1, 1], gains[1])

        # C should pick out each state
        np.testing.assert_almost_equal(C_cpu[0, 0], 1.0)
        np.testing.assert_almost_equal(C_cpu[1, 1], 1.0)

        # D should be zero
        np.testing.assert_almost_equal(D_cpu[0, 0], 0.0)
        np.testing.assert_almost_equal(D_cpu[1, 1], 0.0)

    @cpu_and_gpu
    def test_stability_check(self, target_device_idx, xp):
        """Test stability checking via eigenvalues"""
        # Stable filter: eigenvalue < 1
        gains = [0.5]
        ff = [0.9]
        ssr_stable = SsrFilterData.from_integrator(gains, ff=ff,
                                                   target_device_idx=target_device_idx)
        self.assertTrue(ssr_stable.is_stable())

        # Unstable filter: eigenvalue > 1
        ff_unstable = [1.1]
        ssr_unstable = SsrFilterData.from_integrator(gains, ff=ff_unstable,
                                                    target_device_idx=target_device_idx)
        self.assertFalse(ssr_unstable.is_stable())

    @cpu_and_gpu
    def test_save_restore(self, target_device_idx, xp):
        """Test save and restore functionality"""
        import tempfile
        import os

        # Create test filter
        gains = [0.5, 1.0]
        original = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            tmp_name = tmp.name

        try:
            original.save(tmp_name)

            # Restore
            restored = SsrFilterData.restore(tmp_name, target_device_idx=target_device_idx)

            # Compare block-diagonal matrices
            self.assertEqual(restored.nfilter, original.nfilter)
            self.assertEqual(restored.total_states, original.total_states)

            np.testing.assert_array_almost_equal(cpuArray(restored.A), 
                                                cpuArray(original.A))
            np.testing.assert_array_almost_equal(cpuArray(restored.B), 
                                                cpuArray(original.B))
            np.testing.assert_array_almost_equal(cpuArray(restored.C), 
                                                cpuArray(original.C))
            np.testing.assert_array_almost_equal(cpuArray(restored.D), 
                                                cpuArray(original.D))
        finally:
            os.unlink(tmp_name)
