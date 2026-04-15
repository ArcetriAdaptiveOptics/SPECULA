import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula.lib.calc_phasescreen import calc_phasescreen
from test.specula_testlib import cpu_and_gpu

class TestCalcPhasescreen(unittest.TestCase):

    @cpu_and_gpu
    def test_calc_phasescreen_not_finite_first(self, target_device_idx, xp):
        """Test that the non-finite elements are detected."""
        L0 = 25.0
        dimension = 128
        pixel_pitch = 0.01
        precision = 1
        seed = 42

        def mock_sqrt(x):
            nelements = len(x.flat)
            n_10percent = int(0.1 * nelements)+1
            y = x.copy()
            y.flat[:n_10percent] = np.inf
            return y

        original_sqrt = xp.sqrt
        xp.sqrt = mock_sqrt
        try:
            with self.assertRaises(ValueError):
                _ = calc_phasescreen(L0, dimension, pixel_pitch, xp, precision, seed=seed)
        finally:
            xp.sqrt = original_sqrt
        
    @cpu_and_gpu
    def test_calc_phasescreen_not_finite_second(self, target_device_idx, xp):
        """Test that the non-finite elements are detected."""
        L0 = 25.0
        dimension = 128
        pixel_pitch = 0.01
        precision = 1
        seed = 42

        original_sqrt = xp.sqrt

        def mock_sqrt(x):
            '''Return non-finite data on second call'''
            mock_sqrt.counter += 1
            if mock_sqrt.counter == 2:
                nelements = len(x.flat)
                n_10percent = int(0.1 * nelements)+1
                y = x.copy()
                y.flat[:n_10percent] = np.inf
            else:
                y = original_sqrt(x)
            return y
        mock_sqrt.counter = 0

        xp.sqrt = mock_sqrt
        try:
            with self.assertRaises(ValueError):
                _ = calc_phasescreen(L0, dimension, pixel_pitch, xp, precision, seed=seed)
        finally:
            xp.sqrt = original_sqrt
        
    @cpu_and_gpu
    def test_calc_phasescreen_not_finite_overall(self, target_device_idx, xp):
        """Test that the non-finite elements are detected."""
        L0 = 25.0
        dimension = 128
        pixel_pitch = 0.01
        precision = 1
        seed = 42

        phasescreen = calc_phasescreen(L0, dimension, pixel_pitch, xp, precision, seed=seed)

        assert xp.isfinite(phasescreen).all()
        



