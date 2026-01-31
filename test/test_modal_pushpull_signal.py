 
import unittest
import numpy as np
from unittest.mock import patch

import specula
specula.init(0)  # Default target device

from specula.lib.modal_pushpull_signal import modal_pushpull_signal


class TestModalPushPullSignal(unittest.TestCase):

    def setUp(self):
        # Default mock for ZernikeGenerator.degree
        self.degree_patch = patch("specula.lib.zernike_generator.ZernikeGenerator.degree", return_value=(2, None))
        self.mock_degree = self.degree_patch.start()

    def tearDown(self):
        self.degree_patch.stop()

    def test_basic_vect_amplitude_computation(self):
        """Test automatic vect_amplitude calculation with sqrt(radorder)."""
        n_modes = 3
        amplitude = 10.0
        result = modal_pushpull_signal(n_modes, amplitude=amplitude)

        # vect_amplitude = amplitude / sqrt(2) for all modes
        expected_amplitude = amplitude / np.sqrt(2)
        # Check first few elements in time history
        self.assertAlmostEqual(result[0, 0], expected_amplitude)
        self.assertAlmostEqual(result[1, 0], -expected_amplitude)

    def test_linear_vect_amplitude(self):
        """Test automatic vect_amplitude calculation with linear=True."""
        n_modes = 2
        amplitude = 5.0
        result = modal_pushpull_signal(n_modes, amplitude=amplitude, linear=True)

        # vect_amplitude = amplitude / radorder = 5 / 2
        expected_amplitude = amplitude / 2
        self.assertAlmostEqual(result[0, 0], expected_amplitude)
        self.assertAlmostEqual(result[1, 0], -expected_amplitude)

    def test_custom_vect_amplitude(self):
        """Test using a custom vect_amplitude."""
        vect_amplitude = np.array([1.0, 2.0])
        result = modal_pushpull_signal(2, vect_amplitude=vect_amplitude)
        # For first mode, check positive then negative values
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[1, 0], -1.0)
        # For second mode
        self.assertEqual(result[2, 1], 2.0)
        self.assertEqual(result[3, 1], -2.0)

    def test_min_amplitude_threshold(self):
        """Test that vect_amplitude gets capped by min_amplitude."""
        n_modes = 2
        amplitude = 10.0
        min_amp = 2.0
        result = modal_pushpull_signal(n_modes, amplitude=amplitude, min_amplitude=min_amp)
        # vect_amplitude should be capped at 2.0 for both modes
        self.assertAlmostEqual(result[0, 0], min_amp)
        self.assertAlmostEqual(result[1, 0], -min_amp)

    def test_only_push_behavior(self):
        """Test only_push=True generates positive signals only."""
        vect_amplitude = np.array([3.0, 4.0])
        result = modal_pushpull_signal(2, vect_amplitude=vect_amplitude, only_push=True, ncycles=2)
        # First mode should have 2 cycles of +3.0
        self.assertTrue(np.all(result[0:2, 0] == 3.0))
        # Second mode should have 2 cycles of +4.0
        self.assertTrue(np.all(result[4:6, 1] == 4.0))
        # No negative values at all
        self.assertTrue(np.all(result >= 0))

    def test_repeat_ncycles_behavior(self):
        """Test repeat_ncycles=True creates push then pull for each mode."""
        vect_amplitude = np.array([2.0])
        result = modal_pushpull_signal(1, vect_amplitude=vect_amplitude, ncycles=2, repeat_ncycles=True)
        # First 2 samples = +2.0, next 2 samples = -2.0
        expected = np.array([2.0, 2.0, -2.0, -2.0])
        np.testing.assert_array_equal(result[:, 0], expected)

    def test_nsamples_repetition(self):
        """Test nsamples > 1 repeats each row accordingly."""
        vect_amplitude = np.array([1.0])
        result = modal_pushpull_signal(1, vect_amplitude=vect_amplitude, nsamples=3)
        # Original pattern = [+1, -1]
        expected = np.array([[1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0]])
        np.testing.assert_array_equal(result, expected)

    def test_zero_modes(self):
        """Edge case: n_modes=0 should return empty array."""
        result = modal_pushpull_signal(0, amplitude=1.0)
        self.assertEqual(result.shape[0], 0)
        self.assertEqual(result.shape[1], 0)

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_first_mode_works_correctly(self, mock_degree):
        """
        Verify that modes before first_mode remain zero and signals start from first_mode onwards.
        """
        n_modes = 5
        amplitude = 1.0
        first_mode = 2

        # Call function with first_mode=2
        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            first_mode=first_mode,
            ncycles=1,
            constant=True,
            xp=np
        )

        # Result shape should be (2 * real_n_modes * ncycles, n_modes)
        expected_rows = 2 * (n_modes - first_mode) * 1
        expected_shape = (expected_rows, n_modes)
        self.assertEqual(result.shape, expected_shape)

        # Verify that columns before first_mode are all zeros
        self.assertTrue(np.all(result[:, :first_mode] == 0))

        # Verify that columns from first_mode onwards have non-zero values
        self.assertTrue(np.any(result[:, first_mode:] != 0))

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_constant_vect_amplitude_applied_correctly(self, mock_degree):
        """
        Verify that when constant=True, all vect_amplitude values are equal across modes.
        """
        n_modes = 4
        amplitude = 5.0

        # Call function with constant=True
        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            constant=True,
            ncycles=1,
            xp=np
        )

        # In constant mode, amplitudes for active modes should be ±amplitude
        unique_values = np.unique(np.abs(result[result != 0]))
        self.assertTrue(np.allclose(unique_values, amplitude))

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(2,))
    def test_constant_false_uses_sqrt_radorder(self, mock_degree):
        """
        Verify that when constant=False, vect_amplitude = amplitude / sqrt(radorder).
        """
        n_modes = 3
        amplitude = 4.0

        # Call function with constant=False
        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            constant=False,
            ncycles=1,
            xp=np
        )

        # Since radorder is mocked to 2, vect_amplitude = amplitude / sqrt(2)
        expected_amplitude = amplitude / np.sqrt(2)
        unique_values = np.unique(np.abs(result[result != 0]))
        self.assertTrue(np.allclose(unique_values, expected_amplitude))

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_repeat_ncycles_with_first_mode(self, mock_degree):
        """
        Verify correct behavior when repeat_ncycles=True and first_mode > 0:
        - Modes before first_mode are zero.
        - Remaining modes alternate +amplitude/-amplitude.
        - Shape matches expected 2*real_n_modes*ncycles rows.
        """
        n_modes = 4
        amplitude = 2.0
        first_mode = 1
        ncycles = 2

        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            first_mode=first_mode,
            repeat_ncycles=True,
            ncycles=ncycles,
            constant=True,
            xp=np
        )

        # Expected shape: 2 * real_n_modes * ncycles rows, n_modes cols
        real_n_modes = n_modes - first_mode
        expected_rows = 2 * real_n_modes * ncycles
        self.assertEqual(result.shape, (expected_rows, n_modes))

        # Verify columns before first_mode are always zero
        self.assertTrue(np.all(result[:, :first_mode] == 0))

        # Verify signal alternates +amplitude / -amplitude after first_mode
        for mode in range(first_mode, n_modes):
            col = result[:, mode]
            # The pattern: [ +A, +A, -A, -A ] for ncycles=2
            expected_pattern = np.concatenate(
                [np.full(ncycles, amplitude), np.full(ncycles, -amplitude)]
            )
            # Each mode should have the same expected block repeated
            repeated_blocks = np.tile(expected_pattern, 1)  # Single block per mode
            x1 = (mode-first_mode)*(ncycles*2)
            x2 = x1 + ncycles*2
            self.assertTrue(np.array_equal(col[x1:x2], repeated_blocks))

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_custom_pattern_three_elements(self, mock_degree):
        """
        Verify that a custom pattern [1, -1, 1] produces the correct sequence for each mode.
        """
        n_modes = 3
        amplitude = 2.0
        pattern = [1, -1, 1]
        ncycles = 2

        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            pattern=pattern,
            ncycles=ncycles,
            constant=True,
            xp=np
        )

        n_pokes = len(pattern)
        expected_rows = n_pokes * n_modes * ncycles
        self.assertEqual(result.shape, (expected_rows, n_modes))

        # Expected block for each mode: amplitude * pattern, repeated ncycles times
        expected_block = np.tile(np.array(pattern) * amplitude, ncycles)
        for mode in range(n_modes):
            x1 = mode * len(pattern)*ncycles
            x2 = (mode+1) * len(pattern)*ncycles
            np.testing.assert_array_equal(result[x1:x2, mode], expected_block)

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_custom_pattern_negative_first(self, mock_degree):
        """
        Verify that a custom pattern [-1, 1] applies correctly, starting with a negative amplitude.
        """
        n_modes = 2
        amplitude = 3.0
        pattern = [-1, 1]
        ncycles = 2

        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            pattern=pattern,
            ncycles=ncycles,
            constant=True,
            xp=np
        )

        n_pokes = len(pattern)
        expected_rows = n_pokes * n_modes * ncycles
        self.assertEqual(result.shape, (expected_rows, n_modes))

        expected_block = np.tile(np.array(pattern) * amplitude, ncycles)
        for mode in range(n_modes):
            x1 = mode * len(pattern)*ncycles
            x2 = (mode+1) * len(pattern)*ncycles
            np.testing.assert_array_equal(result[x1:x2, mode], expected_block)

    @patch("specula.lib.modal_pushpull_signal.ZernikeGenerator.degree", return_value=(1,))
    def test_only_push_overrides_pattern(self, mock_degree):
        """
        Verify that setting only_push=True forces the pattern to [1] regardless of the input pattern.
        """
        n_modes = 3
        amplitude = 4.0
        ncycles = 2
        pattern = [-1, 1]

        # Provide a fake negative pattern; should be ignored
        result = modal_pushpull_signal(
            n_modes=n_modes,
            amplitude=amplitude,
            only_push=True,
            pattern=pattern,
            ncycles=ncycles,
            constant=True,
            xp=np
        )

        expected_rows = n_modes * ncycles  # only_push -> 1 poke per mode per cycle
        self.assertEqual(result.shape, (expected_rows, n_modes))

        expected_pattern = [1]
        expected_block = np.tile(np.array(expected_pattern) * amplitude, ncycles)
        for mode in range(n_modes):
            x1 = mode * len(expected_pattern)*ncycles
            x2 = (mode+1) * len(expected_pattern)*ncycles
            np.testing.assert_array_equal(result[x1:x2, mode], expected_block)


