 
import unittest
import numpy as np
from unittest.mock import patch

from specula.lib.modal_pushpull_signal import modal_pushpull_signal  # Replace with your module name


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

