import specula
specula.init(0)

import unittest

from specula import np, cpuArray
from specula.lib.radial_profile import (
    compute_radial_profile,
    compute_fwhm_from_profile,
    compute_encircled_energy,
    get_encircled_energy_at_distance,
)


from test.specula_testlib import cpu_and_gpu

class TestRadialProfile(unittest.TestCase):

    @cpu_and_gpu
    def test_compute_radial_profile_keeps_outermost_bin(self, target_device_idx, xp):
        image = xp.ones((5, 5), dtype=xp.float64)
        profile, radial_distance, counts = compute_radial_profile(
            image,
            center_in_px_y=2,
            center_in_px_x=2,
            return_counts=True,
            xp=xp
        )

        self.assertEqual(len(profile), 3)
        self.assertEqual(len(radial_distance), 3)
        np.testing.assert_array_equal(cpuArray(counts), np.array([1, 8, 16]))
        np.testing.assert_allclose(cpuArray(profile), np.ones(3))

    @cpu_and_gpu
    def test_compute_fwhm_from_profile(self, target_device_idx, xp):
        fwhm_true = 1.7
        radial_distance = xp.linspace(0.0, 5.0, 2000)
        profile = xp.exp(-4.0 * xp.log(2.0) * (radial_distance / fwhm_true) ** 2)

        fwhm = compute_fwhm_from_profile(profile, radial_distance, xp=xp)

        self.assertAlmostEqual(float(fwhm), fwhm_true, places=3)

    @cpu_and_gpu
    def test_compute_encircled_energy_and_value_at_distance(self, target_device_idx, xp):
        image = xp.ones((5, 5), dtype=xp.float64)
        profile, radial_distance, counts = compute_radial_profile(
            image,
            center_in_px_y=2,
            center_in_px_x=2,
            return_counts=True,
            xp=xp,
        )

        ee = compute_encircled_energy(profile, counts, xp=xp)
        ee_at_1p5 = get_encircled_energy_at_distance(ee, radial_distance, 1.5, xp=xp)

        self.assertTrue(np.all(np.diff(ee) >= 0))
        self.assertAlmostEqual(float(ee[-1]), 1.0, places=12)
        expected_ee = np.interp(1.5, cpuArray(radial_distance), cpuArray(ee))
        self.assertAlmostEqual(float(ee_at_1p5), float(expected_ee), places=12)

    @cpu_and_gpu
    def test_compute_encircled_energy_without_counts_uses_radial_distance(self, target_device_idx, xp):
        profile = xp.ones(4, dtype=xp.float64)
        radial_distance = xp.array([0.0, 1.0, 2.0, 3.0], dtype=xp.float64)

        ee = compute_encircled_energy(profile, radial_distance=radial_distance, xp=xp)

        expected_weights = xp.array([0.25, 2.0, 4.0, 6.0], dtype=xp.float64)
        expected_ee = xp.cumsum(expected_weights) / xp.sum(expected_weights)

        np.testing.assert_allclose(cpuArray(ee), cpuArray(expected_ee))
