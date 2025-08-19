import specula
specula.init(0)  # Default target device

import unittest
import numpy as np

from specula.lib.calc_psf_geometry import calc_psf_sampling
from test.specula_testlib import cpu_and_gpu


class TestPsfGeometry(unittest.TestCase):

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

