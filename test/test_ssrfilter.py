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
        self.assertEqual(ssr_data.get_state_size(0), 1)
        self.assertEqual(ssr_data.get_input_size(0), 1)
        self.assertEqual(ssr_data.get_output_size(0), 1)

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
        for i in range(3):
            np.testing.assert_almost_equal(cpuArray(ssr_data.D[i])[0, 0], gains[i])
            np.testing.assert_almost_equal(cpuArray(ssr_data.C[i])[0, 0], 0.0)

    @cpu_and_gpu
    def test_from_integrator(self, target_device_idx, xp):
        """Test from_integrator factory method"""
        gains = [0.5, 1.0]
        dt = 0.001
        ssr_data = SsrFilterData.from_integrator(gains, dt=dt, 
                                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 2)

        # Test integrator structure: x[k+1] = x[k] + dt*gain*u[k], y[k] = x[k]
        for i in range(2):
            np.testing.assert_almost_equal(cpuArray(ssr_data.A[i])[0, 0], 1.0)
            np.testing.assert_almost_equal(cpuArray(ssr_data.B[i])[0, 0], dt * gains[i])
            np.testing.assert_almost_equal(cpuArray(ssr_data.C[i])[0, 0], 1.0)
            np.testing.assert_almost_equal(cpuArray(ssr_data.D[i])[0, 0], 0.0)

    @cpu_and_gpu
    def test_stability_check(self, target_device_idx, xp):
        """Test stability checking via eigenvalues"""
        # Stable filter: eigenvalue < 1
        A_stable = xp.array([[0.9]])
        B = xp.array([[0.1]])
        C = xp.array([[1.0]])
        D = xp.array([[0.0]])

        ssr_stable = SsrFilterData(A_stable, B, C, D, target_device_idx=target_device_idx)
        self.assertTrue(ssr_stable.is_stable(0))

        # Unstable filter: eigenvalue > 1
        A_unstable = xp.array([[1.1]])
        ssr_unstable = SsrFilterData(A_unstable, B, C, D, target_device_idx=target_device_idx)
        self.assertFalse(ssr_unstable.is_stable(0))

    @cpu_and_gpu
    def test_save_restore(self, target_device_idx, xp):
        """Test save and restore functionality"""
        import tempfile
        import os

        # Create test filter
        gains = [0.5, 1.0]
        dt = 0.001
        original = SsrFilterData.from_integrator(gains, dt=dt, 
                                                target_device_idx=target_device_idx)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            tmp_name = tmp.name

        try:
            original.save(tmp_name)

            # Restore
            restored = SsrFilterData.restore(tmp_name, target_device_idx=target_device_idx)

            # Compare
            self.assertEqual(restored.nfilter, original.nfilter)
            for i in range(original.nfilter):
                np.testing.assert_array_almost_equal(cpuArray(restored.A[i]), 
                                                    cpuArray(original.A[i]))
                np.testing.assert_array_almost_equal(cpuArray(restored.B[i]), 
                                                    cpuArray(original.B[i]))
                np.testing.assert_array_almost_equal(cpuArray(restored.C[i]), 
                                                    cpuArray(original.C[i]))
                np.testing.assert_array_almost_equal(cpuArray(restored.D[i]), 
                                                    cpuArray(original.D[i]))
        finally:
            os.unlink(tmp_name)
