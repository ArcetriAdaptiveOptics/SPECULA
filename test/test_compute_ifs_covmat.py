import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula.lib.modal_base_generator import compute_ifs_covmat
from specula.data_objects.ifunc import IFunc


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
            oversampling=2,
            xp=np,
            dtype=np.float32
        )
        result2 = compute_ifs_covmat(
            self.pupil_mask,
            self.diameter,
            self.influence_functions,
            self.r0,
            self.L0,
            oversampling=4,
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

    def test_zernike_variance_decay_with_radial_order(self):
        """
        Test that Zernike RMS (sqrt of variance) averaged over azimuthal orders 
        decays with radial order n. When normalizing modes to unit variance,
        we expect the covariance diagonal to scale as n^(-3) approximately.
        """
        # Parameters
        diameter = 8.0
        r0 = 0.16
        L0 = 1000.0  # Large L0 for pure Kolmogorov
        mask_size = 128
        max_radial_order = 15
        oversampling = 2

        nmodes = (max_radial_order + 1) * (max_radial_order + 2) // 2

        vebose = False
        if vebose: # pragma: no cover
            print(f"\n{'='*70}")
            print(f"Testing Zernike variance decay with radial order")
            print(f"{'='*70}")
            print(f"Parameters: D={diameter}m, r0={r0}m, L0={L0}m")
            print(f"D/r0 = {diameter/r0:.1f}")
            print(f"Mask size: {mask_size}x{mask_size}, modes: {nmodes}")
            print(f"With unit-variance normalization, expected: σ²(n) ∝ n^(-3)")
            print(f"{'='*70}\n")

        # Generate Zernike influence functions
        ifunc = IFunc(
            type_str='zernike',
            nmodes=nmodes,
            npixels=mask_size,
            obsratio=0.0,
            diaratio=1.0,
            precision=1,
            target_device_idx=-1
        )
        pupil_mask = ifunc.mask_inf_func.astype(np.float32)
        z_if_3d = ifunc.ifunc_2d_to_3d(normalize=False)

        # Flatten inside pupil
        idx = np.where(pupil_mask.ravel() > 0.5)[0]
        npupil = idx.size
        influence_functions = np.zeros((nmodes, npupil), dtype=np.float32)

        for k in range(nmodes):
            mode_flat = z_if_3d[:, :, k].ravel()[idx]
            # ✅ Normalize to unit variance
            var = np.var(mode_flat)
            if var > 1e-10:
                mode_flat = mode_flat / np.sqrt(var)
            influence_functions[k, :] = mode_flat

        # Compute covariance
        cov = compute_ifs_covmat(
            pupil_mask,
            diameter,
            influence_functions,
            r0,
            L0,
            oversampling=oversampling,
            xp=np,
            dtype=np.float32
        )
        diag = np.diag(cov)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt

            plt.figure(figsize=(14, 6))

            plt.subplot(1, 2, 1)
            plt.semilogy(range(1, nmodes + 1), diag, 'o', markersize=3, alpha=0.6, label='Individual modes')
            plt.xlabel('Zernike mode index (Noll)')
            plt.ylabel('Variance σ² [rad²]')
            plt.title('Zernike Mode Variances')
            plt.grid(True, alpha=0.3)
            plt.legend()

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
            variances_by_n[n].append(diag[j - 1])

        # Average over azimuthal orders
        radial_orders = sorted(variances_by_n.keys())
        mean_variances = []
        std_variances = []

        if vebose: # pragma: no cover
            print(f"{'n':<4} {'# modes':<10} {'Mean σ²':<15} {'Std σ²':<15}")
            print("-" * 50)

        for n in radial_orders:
            vars_n = variances_by_n[n]
            mean_var = np.mean(vars_n)
            std_var = np.std(vars_n) if len(vars_n) > 1 else 0.0
            mean_variances.append(mean_var)
            std_variances.append(std_var)
            if vebose: # pragma: no cover
                print(f"{n:<4} {len(vars_n):<10} {mean_var:<15.6e} {std_var:<15.6e}")

        if vebose: # pragma: no cover
            print("-" * 50)

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

        if vebose: # pragma: no cover
            print(f"\nPower law fit (n >= {fit_start}): σ²(n) ∝ n^({slope:.3f})")
            print(f"Expected (unit-var norm):   σ²(n) ∝ n^({theoretical_slope:.3f})")
            print(f"Relative error: {abs(slope - theoretical_slope)/abs(theoretical_slope)*100:.1f}%")

        var_pred = np.exp(intercept + slope * log_n)
        ss_res = np.sum((var_fit - var_pred)**2)
        ss_tot = np.sum((var_fit - np.mean(var_fit))**2)
        r_squared = 1 - ss_res / ss_tot
        if vebose: # pragma: no cover
            print(f"R² of fit: {r_squared:.4f}")

        if plot_debug: # pragma: no cover
            plt.subplot(1, 2, 2)

            plt.errorbar(radial_orders, mean_variances, yerr=std_variances,
                        fmt='o', markersize=6, capsize=4, capthick=1.5,
                        label='Mean ± std', color='C0', zorder=3)

            plt.errorbar(radial_orders[fit_start:], mean_variances[fit_start:],
                        yerr=std_variances[fit_start:],
                        fmt='s', markersize=8, capsize=4, capthick=1.5,
                        label=f'Fitted (n≥{fit_start})', color='C3', zorder=4)

            n_plot = np.linspace(fit_start, max_radial_order, 100)
            var_theory = np.exp(intercept) * n_plot**slope
            var_expected = np.exp(intercept) * n_plot**theoretical_slope

            plt.plot(n_plot, var_theory, '--', linewidth=2.5,
                    label=f'Fit: n$^{{{slope:.2f}}}$', color='C1', zorder=2)
            plt.plot(n_plot, var_expected, ':', linewidth=2.5,
                    label=f'Expected: n$^{{{theoretical_slope:.1f}}}$', color='C2', zorder=1)

            plt.xlabel('Radial order n')
            plt.ylabel('Mean variance σ²(n) [rad²]')
            plt.title(f'Variance Decay (R²={r_squared:.3f})')
            plt.yscale('log')
            plt.xscale('log')
            plt.grid(True, alpha=0.3, which='both')
            plt.legend()

            plt.tight_layout()
            plt.show()

        print(f"\n{'='*70}\n")

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
