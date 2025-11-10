import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula.lib.modal_base_generator import compute_ifs_covmat
from specula.data_objects.ifunc import IFunc
from specula import cpuArray

from test.specula_testlib import cpu_and_gpu


class TestComputeIfsCovmat(unittest.TestCase):
    """Test suite for compute_ifs_covmat function."""

    def setUp(self):
        """Create basic test data for covariance matrix computation."""
        # Use numpy for setUp since it's called before test methods
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

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        """Test that output has correct shape (n_actuators x n_actuators)."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            oversampling=2,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (self.n_actuators, self.n_actuators))

    @cpu_and_gpu
    def test_output_is_real(self, target_device_idx, xp):
        """Test that output is real-valued (no complex components)."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertTrue(xp.isrealobj(cpuArray(result)))
        self.assertIn(result.dtype, [xp.float32, xp.float64])

    @cpu_and_gpu
    def test_output_is_symmetric(self, target_device_idx, xp):
        """Test that output covariance matrix is approximately symmetric."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float64
        )

        max_asymmetry = xp.max(xp.abs(result - result.T))
        # Use relative tolerance for larger values
        max_value = xp.max(xp.abs(result))
        self.assertLess(float(max_asymmetry / max_value), 1e-3)

    @cpu_and_gpu
    def test_output_is_positive_semidefinite(self, target_device_idx, xp):
        """Test that output covariance matrix is positive semidefinite."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float64
        )

        eigenvalues = xp.linalg.eigvalsh(result)
        min_eigenvalue = xp.min(eigenvalues)
        # Use relative tolerance
        max_eigenvalue = xp.max(eigenvalues)
        self.assertGreater(float(min_eigenvalue), -1e-6 * float(max_eigenvalue))

    @cpu_and_gpu
    def test_zero_influence_functions(self, target_device_idx, xp):
        """Test behavior with zero influence functions."""
        mask_size = 16
        pupil_mask = xp.ones((mask_size, mask_size), dtype=xp.float32)
        npupil = int(xp.sum(pupil_mask))
        n_actuators = 5

        influence_functions = xp.zeros((n_actuators, npupil), dtype=xp.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        np.testing.assert_allclose(cpuArray(result), 0.0)

    @cpu_and_gpu
    def test_identical_influence_functions(self, target_device_idx, xp):
        """Test with identical influence functions."""
        mask_size = 16
        pupil_mask = xp.ones((mask_size, mask_size), dtype=xp.float32)
        npupil = int(xp.sum(pupil_mask))
        n_actuators = 3

        single_if = xp.random.randn(npupil).astype(xp.float32)
        influence_functions = xp.tile(single_if, (n_actuators, 1))

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        # All elements should be approximately equal
        np.testing.assert_allclose(cpuArray(result), float(cpuArray(result)[0, 0]), rtol=1e-3)

    @cpu_and_gpu
    def test_orthogonal_influence_functions(self, target_device_idx, xp):
        """Test with spatially separated influence functions."""
        mask_size = 16
        pupil_mask = xp.ones((mask_size, mask_size), dtype=xp.float32)
        npupil = int(xp.sum(pupil_mask))
        n_actuators = 3

        influence_functions = xp.zeros((n_actuators, npupil), dtype=xp.float32)
        for i in range(n_actuators):
            influence_functions[i, i*npupil//n_actuators:(i+1)*npupil//n_actuators] = 1.0

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        diagonal = xp.diag(result)
        # Check that diagonal elements are positive and similar
        self.assertTrue(np.all(cpuArray(diagonal) > 0))

    @cpu_and_gpu
    def test_diameter_scaling(self, target_device_idx, xp):
        """Test that results scale appropriately with telescope diameter."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result_d4 = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=self.r0,
            L0=self.L0,
            xp=xp,
            dtype=xp.float32
        )

        result_d8 = compute_ifs_covmat(
            pupil_mask,
            diameter=8.0,
            influence_functions=influence_functions,
            r0=self.r0,
            L0=self.L0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertFalse(np.allclose(cpuArray(result_d4), cpuArray(result_d8)))

    @cpu_and_gpu
    def test_extreme_r0_values(self, target_device_idx, xp):
        """Test with extreme r0 values."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result_small = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            r0=0.05,
            L0=self.L0,
            xp=xp,
            dtype=xp.float64
        )

        self.assertTrue(xp.all(xp.isfinite(result_small)))

        result_large = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            r0=1.0,
            L0=self.L0,
            xp=xp,
            dtype=xp.float64
        )

        self.assertTrue(np.all(np.isfinite(cpuArray(result_large))))

    @cpu_and_gpu
    def test_extreme_L0_values(self, target_device_idx, xp):
        """Test with extreme outer scale values."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result_small = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            L0=1.0,
            xp=xp,
            dtype=xp.float64
        )

        self.assertTrue(np.all(np.isfinite(cpuArray(result_small))))

        result_large = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            L0=100.0,
            xp=xp,
            dtype=xp.float64
        )

        self.assertTrue(np.all(np.isfinite(cpuArray(result_large))))

    @cpu_and_gpu
    def test_large_actuator_count(self, target_device_idx, xp):
        """Test with larger number of actuators."""
        mask_size = 64
        pupil_mask = xp.ones((mask_size, mask_size), dtype=xp.float32)
        npupil = int(xp.sum(pupil_mask))
        n_actuators = 50

        influence_functions = xp.random.randn(n_actuators, npupil).astype(xp.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=8.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            oversampling=2,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(cpuArray(result))))

    @cpu_and_gpu
    def test_mask_with_obstruction(self, target_device_idx, xp):
        """Test with annular pupil mask (central obstruction)."""
        mask_size = 32
        center = mask_size // 2
        y, x = xp.ogrid[:mask_size, :mask_size]
        outer_radius = mask_size // 2 - 2
        inner_radius = mask_size // 4

        pupil_mask = (((x - center)**2 + (y - center)**2 <= outer_radius**2) &
                      ((x - center)**2 + (y - center)**2 >= inner_radius**2)).astype(xp.float32)

        npupil = int(xp.sum(pupil_mask))
        n_actuators = 8
        influence_functions = xp.random.randn(n_actuators, npupil).astype(xp.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=8.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(cpuArray(result))))

    @cpu_and_gpu
    def test_consistency_across_runs_with_same_seed(self, target_device_idx, xp):
        """Test that results are consistent when using same random seed."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        xp.random.seed(123)
        result1 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )

        xp.random.seed(123)
        result2 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )

        np.testing.assert_allclose(cpuArray(result1), cpuArray(result2))

    @cpu_and_gpu
    def test_frobenius_norm_positive(self, target_device_idx, xp):
        """Test that Frobenius norm of covariance matrix is positive."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float64
        )

        frobenius_norm = xp.linalg.norm(result, 'fro')
        self.assertGreater(float(frobenius_norm), 0)

    @cpu_and_gpu
    def test_trace_positive(self, target_device_idx, xp):
        """Test that trace of covariance matrix is positive."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float64
        )

        trace = xp.trace(result)
        self.assertGreater(float(trace), 0)

    @cpu_and_gpu
    def test_covmat_dtype_float32_and_float64(self, target_device_idx, xp):
        """Test that output dtype matches the requested dtype."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result32 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )
        result64 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float64
        )
        self.assertEqual(result32.dtype, xp.float32)
        self.assertEqual(result64.dtype, xp.float64)

    @cpu_and_gpu
    def test_covmat_nan_input(self, target_device_idx, xp):
        """Test that NaN in influence functions produces NaN in output."""
        pupil_mask = xp.asarray(self.pupil_mask)
        infs = xp.asarray(self.influence_functions).copy()
        infs[0, 0] = xp.nan
        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            infs,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )
        # NaN should propagate to at least some elements
        self.assertTrue(np.isnan(cpuArray(result)).any())

    @cpu_and_gpu
    def test_covmat_inf_input(self, target_device_idx, xp):
        """Test that Inf in influence functions produces unusual output."""
        pupil_mask = xp.asarray(self.pupil_mask)
        infs = xp.asarray(self.influence_functions).copy()
        infs[0, 0] = xp.inf if xp.__name__ == 'cupy' else np.inf
        result = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            infs,
            self.r0,
            self.L0,
            xp=xp,
            dtype=xp.float32
        )
        # Either Inf or NaN should appear due to numerical issues
        self.assertTrue(np.isinf(cpuArray(result)).any() or np.isnan(cpuArray(result)).any())

    @cpu_and_gpu
    def test_covmat_shape_mismatch(self, target_device_idx, xp):
        """Test behavior with mismatched influence function shape."""
        pupil_mask = xp.asarray(self.pupil_mask)
        infs = xp.asarray(self.influence_functions)[:, :-1]
        # This may or may not raise an error depending on implementation
        try:
            result = compute_ifs_covmat(
                pupil_mask,
                self.diameter,
                infs,
                self.r0,
                self.L0,
                xp=xp,
                dtype=xp.float32
            )
            # If no error, check that shape is still consistent
            self.assertEqual(result.shape[0], result.shape[1])
        except (ValueError, IndexError):
            # Expected behavior
            pass

    @cpu_and_gpu
    def test_covmat_output_changes_with_oversampling(self, target_device_idx, xp):
        """Test that output changes with oversampling parameter."""
        pupil_mask = xp.asarray(self.pupil_mask)
        influence_functions = xp.asarray(self.influence_functions)

        result1 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            oversampling=2,
            xp=xp,
            dtype=xp.float32
        )
        result2 = compute_ifs_covmat(
            pupil_mask,
            self.diameter,
            influence_functions,
            self.r0,
            self.L0,
            oversampling=4,
            xp=xp,
            dtype=xp.float32
        )
        self.assertFalse(np.allclose(cpuArray(result1), cpuArray(result2)))

    @cpu_and_gpu
    def test_single_actuator(self, target_device_idx, xp):
        """Test with a single actuator."""
        mask_size = 16
        pupil_mask = xp.ones((mask_size, mask_size), dtype=xp.float32)
        npupil = int(xp.sum(pupil_mask))
        n_actuators = 1

        influence_functions = xp.random.randn(n_actuators, npupil).astype(xp.float32)

        result = compute_ifs_covmat(
            pupil_mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (1, 1))
        self.assertTrue(np.isfinite(cpuArray(result)[0, 0]))
        self.assertGreater(float(cpuArray(result)[0, 0]), 0)

    @cpu_and_gpu
    def test_rectangular_mask(self, target_device_idx, xp):
        """Test with non-square pupil mask."""
        mask = xp.ones((20, 30), dtype=xp.float32)
        npupil = int(xp.sum(mask))
        n_actuators = 5

        influence_functions = xp.random.randn(n_actuators, npupil).astype(xp.float32)

        result = compute_ifs_covmat(
            mask,
            diameter=4.0,
            influence_functions=influence_functions,
            r0=0.16,
            L0=25.0,
            xp=xp,
            dtype=xp.float32
        )

        self.assertEqual(result.shape, (n_actuators, n_actuators))
        self.assertTrue(np.all(np.isfinite(cpuArray(result))))

    @cpu_and_gpu
    def test_zernike_variance_decay_with_radial_order(self, target_device_idx, xp):
        """
        Test that Zernike RMS (sqrt of variance) averaged over azimuthal orders 
        decays with radial order n.
        """
        # Parameters
        diameter = 8.0
        r0 = 0.16
        L0 = 1000.0
        mask_size = 128
        max_radial_order = 15
        oversampling = 2

        nmodes = (max_radial_order + 1) * (max_radial_order + 2) // 2

        # Generate Zernike influence functions
        ifunc = IFunc(
            type_str='zernike',
            nmodes=nmodes,
            npixels=mask_size,
            obsratio=0.0,
            diaratio=1.0,
            precision=1,
            target_device_idx=target_device_idx
        )
        pupil_mask = xp.asarray(ifunc.mask_inf_func, dtype=xp.float32)
        z_if_3d = ifunc.ifunc_2d_to_3d(normalize=False)

        # Flatten inside pupil
        idx = xp.where(pupil_mask.ravel() > 0.5)[0]
        npupil = idx.size
        influence_functions = xp.zeros((nmodes, npupil), dtype=xp.float32)

        for k in range(nmodes):
            mode_flat = z_if_3d[:, :, k].ravel()[idx]
            # Normalize to unit variance
            var = xp.var(mode_flat)
            if var > 1e-10:
                mode_flat = mode_flat / xp.sqrt(var)
            influence_functions[k, :] = mode_flat

        # Compute covariance
        cov = compute_ifs_covmat(
            pupil_mask,
            diameter,
            influence_functions,
            r0,
            L0,
            oversampling=oversampling,
            xp=xp,
            dtype=xp.float32
        )
        diag = xp.diag(cov)

        # Map Zernike index to radial order n
        def zernike_j_to_n(j):
            """Convert Noll index j (1-based) to radial order n."""
            if j == 1:
                return 0
            n = int(np.floor((-3 + np.sqrt(9 + 8*(j-1))) / 2))
            # Verify and adjust
            while (n + 1) * (n + 2) // 2 < j:
                n += 1
            while n * (n + 1) // 2 >= j:
                n -= 1
            return n

        # Group variances by radial order
        variances_by_n = {}
        for j in range(1, nmodes + 1):
            n = zernike_j_to_n(j)
            if n not in variances_by_n:
                variances_by_n[n] = []
            variances_by_n[n].append(float(cpuArray(diag[j - 1])))

        # Average over azimuthal orders
        radial_orders = sorted(variances_by_n.keys())
        mean_variances = []

        for n in radial_orders:
            vars_n = variances_by_n[n]
            mean_var = np.mean(vars_n)
            mean_variances.append(mean_var)

        # Fit power law starting from n=4
        fit_start = 4
        n_fit = np.array(radial_orders[fit_start:], dtype=float)
        var_fit = np.array(mean_variances[fit_start:], dtype=float)

        log_n = np.log(n_fit)
        log_var = np.log(var_fit)

        A = np.vstack([np.ones_like(log_n), log_n]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_var, rcond=None)
        intercept, slope = coeffs

        # Expected slope with unit-variance normalization
        theoretical_slope = -3.0

        var_pred = np.exp(intercept + slope * log_n)
        ss_res = np.sum((var_fit - var_pred)**2)
        ss_tot = np.sum((var_fit - np.mean(var_fit))**2)
        r_squared = 1 - ss_res / ss_tot

        # Assertions for slope ≈ -3
        self.assertTrue(all(v > 0 for v in mean_variances[1:]),
                       "All radial order variances should be positive")

        # Check slope is close to -3.0 (allow 20% tolerance)
        rel_error = abs(slope - theoretical_slope) / abs(theoretical_slope)
        self.assertLess(rel_error, 0.20,
            f"Power law exponent {slope:.3f} should be within 20% of {theoretical_slope:.3f}"
        )

        self.assertGreater(r_squared, 0.90,
            f"Power law fit should be good (R²={r_squared:.3f} > 0.90)"
        )

    def test_oversampling_too_low_raises_error(self):
        """Test that oversampling < 2 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            compute_ifs_covmat(
                self.pupil_mask,
                self.diameter,
                self.influence_functions,
                self.r0,
                self.L0,
                oversampling=1,  # Too low!
                xp=np,
                dtype=np.float32
            )

        self.assertIn("Oversampling factor must be at least 2", str(context.exception))

    def test_oversampling_zero_raises_error(self):
        """Test that oversampling = 0 raises ValueError."""
        with self.assertRaises(ValueError) as context:
            compute_ifs_covmat(
                self.pupil_mask,
                self.diameter,
                self.influence_functions,
                self.r0,
                self.L0,
                oversampling=0,
                xp=np,
                dtype=np.float32
            )

        self.assertIn("Oversampling factor must be at least 2", str(context.exception))

    def test_oversampling_negative_raises_error(self):
        """Test that negative oversampling raises ValueError."""
        with self.assertRaises(ValueError) as context:
            compute_ifs_covmat(
                self.pupil_mask,
                self.diameter,
                self.influence_functions,
                self.r0,
                self.L0,
                oversampling=-1,
                xp=np,
                dtype=np.float32
            )

        self.assertIn("Oversampling factor must be at least 2", str(context.exception))
