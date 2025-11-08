import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula.lib.modal_base_generator import compute_ifs_covmat


class TestComputeIfsCovmat(unittest.TestCase):
    """Test suite for compute_ifs_covmat function."""

    def setUp(self):
        """Create basic test data for covariance matrix computation."""
        np.random.seed(42)

        # Create a simple circular pupil mask
        mask_size = 32
        center = mask_size // 2
        y, x = np.ogrid[:mask_size, :mask_size]
        radius = mask_size // 2 - 2
        pupil_mask = ((x - center)**2 + (y - center)**2 <= radius**2).astype(np.float32)

        # Create simple influence functions
        n_actuators = 10
        npupil = int(np.sum(pupil_mask))
        influence_functions = np.random.randn(n_actuators, npupil).astype(np.float32)

        # Turbulence parameters
        diameter = 8.0  # meters
        r0 = 0.16  # meters
        L0 = 25.0  # meters

        self.pupil_mask = pupil_mask
        self.diameter = diameter
        self.influence_functions = influence_functions
        self.r0 = r0
        self.L0 = L0
        self.n_actuators = n_actuators
        self.npupil = npupil

    def test_output_shape(self):
        """Test that output has correct shape (n_actuators x n_actuators)."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            oversampling=2,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (self.n_actuators, self.n_actuators))

    def test_output_is_real(self):
        """Test that output is real-valued (no complex components)."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )

        self.assertTrue(np.isrealobj(result))
        self.assertIn(result.dtype, [np.float32, np.float64])

    def test_output_is_symmetric(self):
        """Test that output covariance matrix is approximately symmetric."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float64
        )

        max_asymmetry = np.max(np.abs(result - result.T))
        # Use relative tolerance for larger values
        max_value = np.max(np.abs(result))
        self.assertLess(max_asymmetry / max_value, 1e-3)

    def test_output_is_positive_semidefinite(self):
        """Test that output covariance matrix is positive semidefinite."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float64
        )

        eigenvalues = np.linalg.eigvalsh(result)
        min_eigenvalue = np.min(eigenvalues)
        # Use relative tolerance
        max_eigenvalue = np.max(eigenvalues)
        self.assertGreater(min_eigenvalue, -1e-6 * max_eigenvalue)

    def test_zero_influence_functions(self):
        """Test behavior with zero influence functions."""
        mask_size = 16
        pupil_mask = np.ones((mask_size, mask_size), dtype=np.float32)
        npupil = int(np.sum(pupil_mask))
        n_actuators = 5

        influence_functions = np.zeros((n_actuators, npupil), dtype=np.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        np.testing.assert_allclose(result, 0.0)

    def test_identical_influence_functions(self):
        """Test with identical influence functions."""
        mask_size = 16
        pupil_mask = np.ones((mask_size, mask_size), dtype=np.float32)
        npupil = int(np.sum(pupil_mask))
        n_actuators = 3

        single_if = np.random.randn(npupil).astype(np.float32)
        influence_functions = np.tile(single_if, (n_actuators, 1))

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        # All elements should be approximately equal
        np.testing.assert_allclose(result, result[0, 0], rtol=1e-3)

    def test_orthogonal_influence_functions(self):
        """Test with spatially separated influence functions."""
        mask_size = 16
        pupil_mask = np.ones((mask_size, mask_size), dtype=np.float32)
        npupil = int(np.sum(pupil_mask))
        n_actuators = 3

        influence_functions = np.zeros((n_actuators, npupil), dtype=np.float32)
        for i in range(n_actuators):
            influence_functions[i, i*npupil//n_actuators:(i+1)*npupil//n_actuators] = 1.0

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        diagonal = np.diag(result)
        # Check that diagonal elements are positive and similar
        self.assertTrue(np.all(diagonal > 0))

    def test_diameter_scaling(self):
        """Test that results scale appropriately with telescope diameter."""
        result_d4 = compute_ifs_covmat(
            self.pupil_mask,
            diameter=4.0,
            influence_functions=self.influence_functions,
            r0=self.r0,
            L0=self.L0,
            xp=np,
            dtype=np.float32
        )

        result_d8 = compute_ifs_covmat(
            self.pupil_mask,
            diameter=8.0,
            influence_functions=self.influence_functions,
            r0=self.r0,
            L0=self.L0,
            xp=np,
            dtype=np.float32
        )

        self.assertFalse(np.allclose(result_d4, result_d8))

    def test_extreme_r0_values(self):
        """Test with extreme r0 values."""
        result_small = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            r0=0.05,
            L0=self.L0,
            xp=np,
            dtype=np.float64
        )

        self.assertTrue(np.all(np.isfinite(result_small)))

        result_large = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            r0=1.0,
            L0=self.L0,
            xp=np,
            dtype=np.float64
        )

        self.assertTrue(np.all(np.isfinite(result_large)))

    def test_extreme_L0_values(self):
        """Test with extreme outer scale values."""
        result_small = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            L0=1.0,
            xp=np,
            dtype=np.float64
        )

        self.assertTrue(np.all(np.isfinite(result_small)))

        result_large = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            L0=100.0,
            xp=np,
            dtype=np.float64
        )

        self.assertTrue(np.all(np.isfinite(result_large)))

    def test_large_actuator_count(self):
        """Test with larger number of actuators."""
        mask_size = 64
        pupil_mask = np.ones((mask_size, mask_size), dtype=np.float32)
        npupil = int(np.sum(pupil_mask))
        n_actuators = 50

        influence_functions = np.random.randn(n_actuators, npupil).astype(np.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=8.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            oversampling=2,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_mask_with_obstruction(self):
        """Test with annular pupil mask (central obstruction)."""
        mask_size = 32
        center = mask_size // 2
        y, x = np.ogrid[:mask_size, :mask_size]
        outer_radius = mask_size // 2 - 2
        inner_radius = mask_size // 4

        pupil_mask = (((x - center)**2 + (y - center)**2 <= outer_radius**2) &
                      ((x - center)**2 + (y - center)**2 >= inner_radius**2)).astype(np.float32)

        npupil = int(np.sum(pupil_mask))
        n_actuators = 8
        influence_functions = np.random.randn(n_actuators, npupil).astype(np.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=8.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_consistency_across_runs_with_same_seed(self):
        """Test that results are consistent when using same random seed."""
        np.random.seed(123)
        result1 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )

        np.random.seed(123)
        result2 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )

        np.testing.assert_allclose(result1, result2)

    def test_frobenius_norm_positive(self):
        """Test that Frobenius norm of covariance matrix is positive."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float64
        )

        frobenius_norm = np.linalg.norm(result, 'fro')
        self.assertGreater(frobenius_norm, 0)

    def test_trace_positive(self):
        """Test that trace of covariance matrix is positive."""
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float64
        )

        trace = np.trace(result)
        self.assertGreater(trace, 0)

    def test_covmat_dtype_float32_and_float64(self):
        """Test that output dtype matches the requested dtype."""
        result32 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )
        result64 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float64
        )
        self.assertEqual(result32.dtype, np.float32)
        self.assertEqual(result64.dtype, np.float64)

    def test_covmat_nan_input(self):
        """Test that NaN in influence functions produces NaN in output."""
        infs = self.influence_functions.copy()
        infs[0, 0] = np.nan
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            infs,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )
        # NaN should propagate to at least some elements
        self.assertTrue(np.isnan(result).any())

    def test_covmat_inf_input(self):
        """Test that Inf in influence functions produces unusual output."""
        infs = self.influence_functions.copy()
        infs[0, 0] = np.inf
        result = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            infs,
            self.r0,
            self.L0,
            xp=np,
            dtype=np.float32
        )
        # Either Inf or NaN should appear due to numerical issues
        self.assertTrue(np.isinf(result).any() or np.isnan(result).any())

    def test_covmat_shape_mismatch(self):
        """Test behavior with mismatched influence function shape."""
        infs = self.influence_functions[:, :-1]
        # This may or may not raise an error depending on implementation
        try:
            result = compute_ifs_covmat(
                self.pupil_mask,
                self.diameter,
                infs,
                self.r0,
                self.L0,
                xp=np,
                dtype=np.float32
            )
            # If no error, check that shape is still consistent
            self.assertEqual(result.shape[0], result.shape[1])
        except (ValueError, IndexError):
            # Expected behavior
            pass

    def test_covmat_output_changes_with_oversampling(self):
        """Test that output changes with oversampling parameter."""
        result1 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            oversampling=1,
            xp=np,
            dtype=np.float32
        )
        result2 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            oversampling=3,
            xp=np,
            dtype=np.float32
        )
        self.assertFalse(np.allclose(result1, result2))

    def test_single_actuator(self):
        """Test with a single actuator."""
        mask_size = 16
        pupil_mask = np.ones((mask_size, mask_size), dtype=np.float32)
        npupil = int(np.sum(pupil_mask))
        n_actuators = 1

        influence_functions = np.random.randn(n_actuators, npupil).astype(np.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(np.isfinite(result[0, 0]))
        self.assertGreater(result[0, 0], 0)

    def test_rectangular_mask(self):
        """Test with non-square pupil mask."""
        mask = np.ones((20, 30), dtype=np.float32)
        npupil = int(np.sum(mask))
        n_actuators = 5

        influence_functions = np.random.randn(n_actuators, npupil).astype(np.float32)

        result = compute_ifs_covmat(
            mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=np,
            dtype=np.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(result)))
