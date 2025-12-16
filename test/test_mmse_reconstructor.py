import specula
specula.init(-1, precision=0)  # CPU, single precision

import unittest
import numpy as np
from specula import cpuArray, np
from specula.lib.mmse_reconstructor import compute_mmse_reconstructor
from test.specula_testlib import cpu_and_gpu

class TestMMSEReconstructor(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        # Simple test case: 3 modes, 4 slopes
        self.n_modes = 3
        self.n_slopes = 4

        # Create interaction matrix: shape (n_slopes, n_modes)
        # Corrected: era (3,4) ora è (4,3)
        self.interaction_matrix = np.array([
            [1.0, 0.2, 0.1],
            [0.5, 1.0, 0.3],
            [0.1, 0.8, 1.0],
            [0.0, 0.3, 0.9]
        ], dtype=np.float32)

        # Atmospheric covariance (identity for simplicity)
        self.c_atm = np.eye(self.n_modes, dtype=np.float32)

        # Noise variances
        self.noise_variance = [0.1, 0.1]  # 2 WFS with same noise

        # Noise covariance matrix (diagonal)
        self.c_noise = np.diag([0.1, 0.1, 0.1, 0.1]).astype(np.float32)

    @cpu_and_gpu
    def test_mmse_basic_functionality(self, target_device_idx, xp):
        """Test basic MMSE reconstructor computation"""
        # Convert arrays to appropriate backend
        A = xp.asarray(self.interaction_matrix)
        c_atm = xp.asarray(self.c_atm)

        # Compute reconstructor
        W = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            noise_variance=self.noise_variance,
            verbose=False
        )

        # Check output shape: (n_modes, n_slopes)
        expected_shape = (self.n_modes, self.n_slopes)
        self.assertEqual(W.shape, expected_shape)

        # Check that reconstructor is not all zeros
        self.assertGreater(float(xp.sum(xp.abs(W))), 0.0)

    @cpu_and_gpu
    def test_mmse_with_noise_matrix(self, target_device_idx, xp):
        """Test MMSE reconstructor with explicit noise covariance matrix"""
        A = xp.asarray(self.interaction_matrix)
        c_atm = xp.asarray(self.c_atm)
        c_noise = xp.asarray(self.c_noise)

        W = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            c_noise=c_noise,
            verbose=False
        )

        self.assertEqual(W.shape, (self.n_modes, self.n_slopes))
        self.assertGreater(float(xp.sum(xp.abs(W))), 0.0)

    @cpu_and_gpu
    def test_mmse_perfect_case(self, target_device_idx, xp):
        """Test MMSE reconstructor in perfect conditions (low noise)"""
        # Square interaction matrix for perfect inversion
        A_square = xp.eye(self.n_modes, dtype=xp.float32)
        c_atm = xp.eye(self.n_modes, dtype=xp.float32)

        W = compute_mmse_reconstructor(
            A_square, c_atm, xp, xp.float32,
            noise_variance=[1e-6],  # Very low noise instead of 0.0
            verbose=False
        )

        # In perfect case with low noise, should approximate pseudoinverse
        A_pinv = xp.linalg.pinv(A_square)
        xp.testing.assert_allclose(W, A_pinv, rtol=1e-2, atol=1e-2)

    @cpu_and_gpu
    def test_mmse_reconstruction_accuracy(self, target_device_idx, xp):
        """Test that reconstructor can recover known modes"""
        A = xp.asarray(self.interaction_matrix)
        c_atm = xp.asarray(self.c_atm) * 10.0  # Strong prior
        
        W = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            noise_variance=[0.01, 0.01],  # Low noise
            verbose=False
        )

        # Test reconstruction with known input
        true_modes = xp.array([1.0, -0.5, 0.3], dtype=xp.float32)
        slopes = A @ true_modes
        reconstructed_modes = W @ slopes

        # Should reconstruct reasonably well with low noise
        error = float(xp.mean(xp.abs(true_modes - reconstructed_modes)))
        self.assertLess(error, 0.5)  # Reasonable reconstruction error

    @cpu_and_gpu
    def test_mmse_diagonal_matrices(self, target_device_idx, xp):
        """Test MMSE with diagonal covariance matrices (optimized path)"""
        A = xp.asarray(self.interaction_matrix)

        # Diagonal atmospheric covariance
        c_atm_diag = xp.diag(xp.array([1.0, 2.0, 0.5], dtype=xp.float32))

        W = compute_mmse_reconstructor(
            A, c_atm_diag, xp, xp.float32,
            noise_variance=[0.1, 0.1],
            verbose=False
        )

        self.assertEqual(W.shape, (self.n_modes, self.n_slopes))
        self.assertGreater(float(xp.sum(xp.abs(W))), 0.0)

    @cpu_and_gpu
    def test_mmse_inverted_matrices(self, target_device_idx, xp):
        """Test MMSE with pre-inverted covariance matrices"""
        A = xp.asarray(self.interaction_matrix)

        # Pre-inverted matrices
        c_atm_inv = xp.linalg.inv(xp.asarray(self.c_atm))
        c_noise_inv = xp.linalg.inv(xp.asarray(self.c_noise))

        W = compute_mmse_reconstructor(
            A, c_atm_inv, xp, xp.float32,
            c_noise=c_noise_inv,
            c_inverse=True,
            verbose=False
        )

        self.assertEqual(W.shape, (self.n_modes, self.n_slopes))

    def test_mmse_dimension_mismatch(self):
        """Test error handling for dimension mismatches"""
        A = np.asarray(self.interaction_matrix)  # Usa np invece di xp
        c_atm_wrong = np.eye(self.n_modes + 1, dtype=np.float32)  # Wrong size

        with self.assertRaises(ValueError):
            compute_mmse_reconstructor(
                A, c_atm_wrong, np, np.float32,
                noise_variance=self.noise_variance
            )

    @cpu_and_gpu
    def test_mmse_multiple_wfs(self, target_device_idx, xp):
        """Test MMSE with multiple WFS having different noise levels"""
        A = xp.asarray(self.interaction_matrix)
        c_atm = xp.asarray(self.c_atm)

        # Different noise for each WFS
        noise_variance_multi = [0.05, 0.2]  # First WFS better than second

        W = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            noise_variance=noise_variance_multi,
            verbose=False
        )

        self.assertEqual(W.shape, (self.n_modes, self.n_slopes))

    @cpu_and_gpu
    def test_mmse_singular_matrix_handling(self, target_device_idx, xp):
        """Test handling of singular matrices (uses pseudoinverse)"""
        # Create singular interaction matrix: shape (4, 3)
        A_singular = xp.array([
            [1.0, 2.0, 0.1],
            [2.0, 4.0, 0.3],  # Linearly dependent rows
            [3.0, 6.0, 0.5],  # Linearly dependent
            [0.1, 0.2, 0.7]
        ], dtype=xp.float32)

        c_atm = xp.eye(3, dtype=xp.float32)

        # Should not raise error, but use pseudoinverse internally
        W = compute_mmse_reconstructor(
            A_singular, c_atm, xp, xp.float32,
            noise_variance=[0.1, 0.1],
            verbose=False  # Rimuovi verbose per evitare output nei test
        )

        self.assertEqual(W.shape, (3, 4))

    @cpu_and_gpu
    def test_mmse_consistency_different_noise_inputs(self, target_device_idx, xp):
        """Test that different ways of specifying noise give consistent results"""
        A = xp.asarray(self.interaction_matrix)
        c_atm = xp.asarray(self.c_atm)

        # Method 1: Use noise_variance list
        W1 = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            noise_variance=[0.1, 0.1],
            verbose=False
        )

        # Method 2: Use explicit noise covariance matrix
        c_noise_explicit = xp.asarray(self.c_noise)
        W2 = compute_mmse_reconstructor(
            A, c_atm, xp, xp.float32,
            c_noise=c_noise_explicit,
            verbose=False
        )

        # Should give same result
        xp.testing.assert_allclose(W1, W2, rtol=1e-5, atol=1e-5)

    def test_mmse_verbose_output(self):
        """Test that verbose mode doesn't crash and produces output"""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            A = np.asarray(self.interaction_matrix)  # Usa np invece di xp
            c_atm = np.asarray(self.c_atm)

            W = compute_mmse_reconstructor(
                A, c_atm, np, np.float32,
                noise_variance=self.noise_variance,
                verbose=True
            )

            # Check that some output was produced
            output = captured_output.getvalue()
            self.assertGreater(len(output), 0)

        finally:
            sys.stdout = sys.__stdout__