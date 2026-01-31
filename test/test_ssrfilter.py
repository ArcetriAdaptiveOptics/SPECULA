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

    @cpu_and_gpu
    def test_ensure_matrix_list_scalar(self, target_device_idx, xp):
        """Test _ensure_matrix_list with scalar input"""
        # Single scalar should become [[[scalar]]]
        ssr_data = SsrFilterData(1.0, 0.5, 1.0, 0.0,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 1)
        self.assertEqual(ssr_data.total_states, 1)

        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_almost_equal(A_cpu[0, 0], 1.0)

    @cpu_and_gpu
    def test_ensure_matrix_list_1d_list(self, target_device_idx, xp):
        """Test _ensure_matrix_list with 1D list (multiple filters)"""
        # List of scalars [1, 2, 3] -> 3 filters with 1x1 matrices
        A = [0.9, 0.8, 0.7]
        B = [0.1, 0.2, 0.3]
        C = [1.0, 1.0, 1.0]
        D = [0.0, 0.0, 0.0]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)
        self.assertEqual(ssr_data.total_states, 3)

        # Check block-diagonal A
        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_almost_equal(A_cpu[0, 0], 0.9)
        np.testing.assert_almost_equal(A_cpu[1, 1], 0.8)
        np.testing.assert_almost_equal(A_cpu[2, 2], 0.7)

    @cpu_and_gpu
    def test_ensure_matrix_list_2d_array(self, target_device_idx, xp):
        """Test _ensure_matrix_list with 2D numpy array (single filter)"""
        # Single 2D array [[a, b], [c, d]] -> 1 filter with 2x2 state matrix
        A = np.array([[0.9, 0.1], [0.0, 0.8]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.0]])
        D = np.array([[0.0]])

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 1)
        self.assertEqual(ssr_data.total_states, 2)

        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_array_almost_equal(A_cpu, A)

    @cpu_and_gpu
    def test_ensure_matrix_list_nested_lists(self, target_device_idx, xp):
        """Test _ensure_matrix_list with list of lists (multiple filters)"""
        # List of 2D matrices (as nested lists)
        A = [[[0.9]], [[0.8]]]
        B = [[[0.1]], [[0.2]]]
        C = [[[1.0]], [[1.0]]]
        D = [[[0.0]], [[0.0]]]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 2)
        self.assertEqual(ssr_data.total_states, 2)

        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_almost_equal(A_cpu[0, 0], 0.9)
        np.testing.assert_almost_equal(A_cpu[1, 1], 0.8)

    @cpu_and_gpu
    def test_ensure_matrix_list_numpy_arrays(self, target_device_idx, xp):
        """Test _ensure_matrix_list with list of numpy arrays"""
        # List of 2D numpy arrays (multiple filters)
        A = [np.array([[0.9]]), np.array([[0.8]]), np.array([[0.7]])]
        B = [np.array([[0.1]]), np.array([[0.2]]), np.array([[0.3]])]
        C = [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]
        D = [np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]])]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)
        self.assertEqual(ssr_data.total_states, 3)

    @cpu_and_gpu
    def test_yaml_style_input_single_2d_matrix(self, target_device_idx, xp):
        """Test YAML-style input: [[1,2],[3,4]] for single 2x2 matrix"""
        # This is how YAML represents a single 2D matrix
        A = [[0.9, 0.1], [0.0, 0.8]]  # Single 2x2 matrix
        B = [[1.0], [0.5]]              # 2x1 matrix
        C = [[1.0, 0.5]]                # 1x2 matrix
        D = [[0.0]]                     # 1x1 matrix

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 1)
        self.assertEqual(ssr_data.total_states, 2)

        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_array_almost_equal(A_cpu, np.array(A))

    @cpu_and_gpu
    def test_yaml_style_input_multiple_scalars(self, target_device_idx, xp):
        """Test YAML-style input: [1, 2, 3] for multiple filters"""
        # This is how YAML represents a list of scalars
        A = [0.9, 0.8, 0.7]  # 3 filters with scalar A
        B = [0.1, 0.2, 0.3]
        C = [1.0, 1.0, 1.0]
        D = [0.0, 0.0, 0.0]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)
        self.assertEqual(ssr_data.total_states, 3)

    @cpu_and_gpu
    def test_mixed_state_sizes(self, target_device_idx, xp):
        """Test filters with different state dimensions"""
        # Filter 1: 1 state, Filter 2: 2 states
        A1 = np.array([[0.9]])
        A2 = np.array([[0.8, 0.1], [0.0, 0.7]])

        B1 = np.array([[0.1]])
        B2 = np.array([[1.0], [0.5]])

        C1 = np.array([[1.0]])
        C2 = np.array([[1.0, 0.5]])

        D1 = np.array([[0.0]])
        D2 = np.array([[0.0]])

        ssr_data = SsrFilterData([A1, A2], [B1, B2], [C1, C2], [D1, D2],
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 2)
        self.assertEqual(ssr_data.total_states, 3)  # 1 + 2

        # Check A is block-diagonal with correct sizes
        A_cpu = cpuArray(ssr_data.A)
        self.assertEqual(A_cpu.shape, (3, 3))

        # First filter's A
        np.testing.assert_almost_equal(A_cpu[0, 0], 0.9)

        # Second filter's A
        np.testing.assert_almost_equal(A_cpu[1, 1], 0.8)
        np.testing.assert_almost_equal(A_cpu[1, 2], 0.1)
        np.testing.assert_almost_equal(A_cpu[2, 2], 0.7)

    @cpu_and_gpu
    def test_invalid_input_empty_list(self, target_device_idx, xp):
        """Test that empty list raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SsrFilterData([], [0.1], [1.0], [0.0],
                         target_device_idx=target_device_idx)

        self.assertIn("Empty matrix list", str(context.exception))

    @cpu_and_gpu
    def test_invalid_input_wrong_dimensions(self, target_device_idx, xp):
        """Test that mismatched dimensions raise ValueError"""
        # A has 2 filters, B has 1 filter
        A = [0.9, 0.8]
        B = [0.1]
        C = [1.0, 1.0]
        D = [0.0, 0.0]

        with self.assertRaises(ValueError) as context:
            SsrFilterData(A, B, C, D,
                         target_device_idx=target_device_idx)

        self.assertIn("must have same length", str(context.exception))

    @cpu_and_gpu
    def test_equivalence_different_input_formats(self, target_device_idx, xp):
        """Test that different input formats produce same result"""
        # All these should create the same filter

        # Format 1: List of scalars
        ssr1 = SsrFilterData([0.9], [0.1], [1.0], [0.0],
                            target_device_idx=target_device_idx)

        # Format 2: List of 2D arrays
        ssr2 = SsrFilterData([np.array([[0.9]])],
                            [np.array([[0.1]])],
                            [np.array([[1.0]])],
                            [np.array([[0.0]])],
                            target_device_idx=target_device_idx)

        # Format 3: List of nested lists
        ssr3 = SsrFilterData([[[0.9]]], [[[0.1]]], [[[1.0]]], [[[0.0]]],
                            target_device_idx=target_device_idx)

        # Format 4: Single 2D array
        ssr4 = SsrFilterData(np.array([[0.9]]),
                            np.array([[0.1]]),
                            np.array([[1.0]]),
                            np.array([[0.0]]),
                            target_device_idx=target_device_idx)

        # All should produce identical results
        for ssr in [ssr2, ssr3, ssr4]:
            np.testing.assert_array_almost_equal(cpuArray(ssr1.A), cpuArray(ssr.A))
            np.testing.assert_array_almost_equal(cpuArray(ssr1.B), cpuArray(ssr.B))
            np.testing.assert_array_almost_equal(cpuArray(ssr1.C), cpuArray(ssr.C))
            np.testing.assert_array_almost_equal(cpuArray(ssr1.D), cpuArray(ssr.D))

    @cpu_and_gpu
    def test_ensure_matrix_list_0d_arrays(self, target_device_idx, xp):
        """Test _ensure_matrix_list with list of 0D arrays (scalar numpy arrays)"""
        # List of 0D arrays (common after deserialization)
        A = [np.array(0.9), np.array(0.8), np.array(0.7)]
        B = [np.array(0.1), np.array(0.2), np.array(0.3)]
        C = [np.array(1.0), np.array(1.0), np.array(1.0)]
        D = [np.array(0.0), np.array(0.0), np.array(0.0)]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)
        self.assertEqual(ssr_data.total_states, 3)

        # Check block-diagonal A
        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_almost_equal(A_cpu[0, 0], 0.9)
        np.testing.assert_almost_equal(A_cpu[1, 1], 0.8)
        np.testing.assert_almost_equal(A_cpu[2, 2], 0.7)

    @cpu_and_gpu
    def test_ensure_matrix_list_mixed_scalars_and_0d(self, target_device_idx, xp):
        """Test _ensure_matrix_list with mix of Python scalars and 0D arrays"""
        # Mix of regular scalars and numpy 0D arrays (edge case from deserialization)
        A = [0.9, np.array(0.8), 0.7]  # Mix of types
        B = [np.array(0.1), 0.2, np.array(0.3)]
        C = [1.0, 1.0, 1.0]
        D = [0.0, 0.0, 0.0]

        ssr_data = SsrFilterData(A, B, C, D,
                                target_device_idx=target_device_idx)

        self.assertEqual(ssr_data.nfilter, 3)
        self.assertEqual(ssr_data.total_states, 3)

        A_cpu = cpuArray(ssr_data.A)
        np.testing.assert_almost_equal(A_cpu[0, 0], 0.9)
        np.testing.assert_almost_equal(A_cpu[1, 1], 0.8)
        np.testing.assert_almost_equal(A_cpu[2, 2], 0.7)

    @cpu_and_gpu
    def test_invalid_input_empty_row(self, target_device_idx, xp):
        """Test that empty row in nested list raises ValueError"""
        with self.assertRaises(ValueError) as context:
            SsrFilterData([[]], [[0.1]], [[1.0]], [[0.0]],
                         target_device_idx=target_device_idx)

        self.assertIn("Empty row in matrix list", str(context.exception))

    @cpu_and_gpu
    def test_invalid_input_1d_array_list(self, target_device_idx, xp):
        """Test that list of 1D arrays raises ValueError (ambiguous)"""
        # List of 1D arrays is ambiguous
        A = [np.array([0.9, 0.1]), np.array([0.8, 0.2])]
        B = [np.array([0.1]), np.array([0.2])]
        C = [np.array([1.0, 0.5]), np.array([1.0, 0.5])]
        D = [np.array([0.0]), np.array([0.0])]

        with self.assertRaises(ValueError) as context:
            SsrFilterData(A, B, C, D,
                         target_device_idx=target_device_idx)

        self.assertIn("ambiguous", str(context.exception).lower())
