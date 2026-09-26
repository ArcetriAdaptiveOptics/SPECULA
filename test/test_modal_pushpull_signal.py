 
import unittest
import numpy as np
from unittest.mock import patch

import specula
specula.init(0)  # Default target device

from specula.lib.modal_pushpull_signal import modal_pushpull_amplitudes


class TestModalPushPullAmplitudes(unittest.TestCase):

    def setUp(self):
        # Default mock for ZernikeGenerator.degree
        self.degree_patch = patch("specula.lib.zernike_generator.ZernikeGenerator.degree", return_value=(2, None))
        self.mock_degree = self.degree_patch.start()

    def tearDown(self):
        self.degree_patch.stop()

    def test_amplitudes_length_and_default_sqrt(self):
        """Test that modal_pushpull_amplitudes returns a vector of length n_modes,
        using amplitude / sqrt(radorder) by default (radorder mocked to 2)."""
        n_modes = 5
        amplitude = 10.0
        result = modal_pushpull_amplitudes(n_modes, amplitude=amplitude)

        self.assertEqual(result.shape, (n_modes,))
        expected = amplitude / np.sqrt(2)
        np.testing.assert_allclose(result, expected)

    def test_amplitudes_leading_zeros_for_first_mode(self):
        """Test that the first `first_mode` entries are zero and the rest are non-zero."""
        n_modes = 5
        first_mode = 2
        amplitude = 10.0
        result = modal_pushpull_amplitudes(n_modes, first_mode=first_mode, amplitude=amplitude)

        self.assertEqual(result.shape, (n_modes,))
        np.testing.assert_array_equal(result[:first_mode], 0)
        expected = amplitude / np.sqrt(2)
        np.testing.assert_allclose(result[first_mode:], expected)

    def test_amplitudes_constant(self):
        """Test that constant=True yields the same amplitude for every mode."""
        n_modes = 4
        amplitude = 5.0
        result = modal_pushpull_amplitudes(n_modes, amplitude=amplitude, constant=True)

        self.assertEqual(result.shape, (n_modes,))
        np.testing.assert_allclose(result, amplitude)

    def test_amplitudes_linear(self):
        """Test that linear=True yields amplitude / radorder (radorder mocked to 2)."""
        n_modes = 3
        amplitude = 6.0
        result = modal_pushpull_amplitudes(n_modes, amplitude=amplitude, linear=True)

        self.assertEqual(result.shape, (n_modes,))
        np.testing.assert_allclose(result, amplitude / 2)

    def test_amplitudes_min_amplitude(self):
        """Test that amplitudes are capped at min_amplitude (radorder mocked to 2)."""
        n_modes = 2
        amplitude = 10.0
        min_amp = 2.0
        result = modal_pushpull_amplitudes(n_modes, amplitude=amplitude, min_amplitude=min_amp)

        np.testing.assert_allclose(result, min_amp)

    def test_amplitudes_explicit_vect_amplitude_with_first_mode(self):
        """Test that an explicit vect_amplitude is simply prepended with first_mode zeros."""
        n_modes = 4
        first_mode = 1
        vect_amplitude = np.array([1.0, 2.0, 3.0])
        result = modal_pushpull_amplitudes(n_modes, first_mode=first_mode, vect_amplitude=vect_amplitude)

        expected = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)
