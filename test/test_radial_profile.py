import specula
specula.init(0)

import unittest

from specula import np
from specula.lib.radial_profile import (
    computeRadialProfile,
    computeFWHMFromProfile,
    computeEncircledEnergy,
    getEncircledEnergyAtDistance,
)


class TestRadialProfile(unittest.TestCase):

    def test_compute_radial_profile_keeps_outermost_bin(self):
        image = np.ones((5, 5), dtype=np.float64)
        profile, radial_distance, counts = computeRadialProfile(
            image,
            centerInPxY=2,
            centerInPxX=2,
            return_counts=True,
        )

        self.assertEqual(len(profile), 3)
        self.assertEqual(len(radial_distance), 3)
        np.testing.assert_array_equal(counts, np.array([1, 8, 16]))
        np.testing.assert_allclose(profile, np.ones(3))

    def test_compute_fwhm_from_profile(self):
        fwhm_true = 1.7
        radial_distance = np.linspace(0.0, 5.0, 2000)
        profile = np.exp(-4.0 * np.log(2.0) * (radial_distance / fwhm_true) ** 2)

        fwhm = computeFWHMFromProfile(profile, radial_distance)

        self.assertAlmostEqual(float(fwhm), fwhm_true, places=3)

    def test_compute_encircled_energy_and_value_at_distance(self):
        image = np.ones((5, 5), dtype=np.float64)
        profile, radial_distance, counts = computeRadialProfile(
            image,
            centerInPxY=2,
            centerInPxX=2,
            return_counts=True,
        )

        ee = computeEncircledEnergy(profile, counts)
        ee_at_1p5 = getEncircledEnergyAtDistance(ee, radial_distance, 1.5)

        self.assertTrue(np.all(np.diff(ee) >= 0))
        self.assertAlmostEqual(float(ee[-1]), 1.0, places=12)
        expected_ee = np.interp(1.5, radial_distance, ee)
        self.assertAlmostEqual(float(ee_at_1p5), float(expected_ee), places=12)

    def test_compute_encircled_energy_without_counts_uses_radial_distance(self):
        profile = np.ones(4, dtype=np.float64)
        radial_distance = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)

        ee = computeEncircledEnergy(profile, radialDistance=radial_distance)

        expected_weights = np.array([0.25, 2.0, 4.0, 6.0], dtype=np.float64)
        expected_ee = np.cumsum(expected_weights) / np.sum(expected_weights)

        np.testing.assert_allclose(ee, expected_ee)
