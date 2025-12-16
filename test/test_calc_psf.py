import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula.lib.calc_psf import calc_psf, calc_psf_pixel_size, calc_psf_sampling
from test.specula_testlib import cpu_and_gpu


class TestCalcPsf(unittest.TestCase):

    @cpu_and_gpu
    def test_calc_psf(self, target_device_idx, xp):

        # Create test phase and amplitude
        phase = xp.zeros((10, 10))
        amp = xp.ones((10, 10))

        # Test basic PSF calculation
        result = calc_psf(phase, amp, normalize=True, xp=xp)
        self.assertEqual(result.shape, (10, 10))
        self.assertAlmostEqual(float(xp.sum(result)), 1.0, places=6)

        # Test with different output size
        result_big = calc_psf(phase, amp, imwidth=20, normalize=True, xp=xp)
        self.assertEqual(result_big.shape, (20, 20))

        # Test without centering
        result_nocenter = calc_psf(phase, amp, nocenter=True, xp=xp)
        self.assertEqual(result_nocenter.shape, (10, 10))

    @cpu_and_gpu
    def test_calc_psf_sampling(self, target_device_idx, xp):
        """Test PSF sampling calculation"""
        pixel_pupil = 20
        pixel_pitch = 0.05
        wavelength_nm = 500.0

        # Test normal case
        sampling = calc_psf_sampling(pixel_pupil, pixel_pitch, wavelength_nm, 10.0)
        self.assertIsInstance(sampling, float)
        self.assertGreater(sampling, 0)

        # Test case where requested pixel size is too large
        dim_pup_in_m = pixel_pupil * pixel_pitch
        max_pixel_size_mas = (wavelength_nm * 1e-9 / dim_pup_in_m * 3600 * 180 / np.pi) * 1000

        with self.assertRaises(ValueError):
            calc_psf_sampling(pixel_pupil, pixel_pitch, wavelength_nm, max_pixel_size_mas * 2)

    def test_calc_psf_pixel_size(self):

        result = calc_psf_pixel_size(500, 1.0, 2.0)

        # Compute expected manually
        expected = (500e-9 / 1.0 * 3600 * 180 / np.pi) * 1000 / 2.0
        self.assertAlmostEqual(result, expected)