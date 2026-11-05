"""
Tests of display_lgs_geometry() and of the helpers it shares with
calc_noise_cov_elong(): sub_aperture_centres() and geometric_elongation().
"""
import specula
specula.init(0)  # Default target device

import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from specula.lib.calc_noise_cov_elong import sub_aperture_centres, geometric_elongation
from specula.lib.plot_utils import display_lgs_geometry

ARCSEC = 3600.0 * 180.0 / np.pi
D = 8.0
N_SUB = 10
KWARGS = dict(sub_aps_fov=10.0, sh_spot_fwhm=1.0, na_thickness_in_m=10e3, show_plot=False)


def angle_mod180(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 180.0


def diff_mod180(a, b):
    d = (a - b) % 180.0
    return np.minimum(d, 180.0 - d)


class TestDisplayLgsGeometry(unittest.TestCase):

    def setUp(self):
        self.figs = []

    def tearDown(self):
        for fig in self.figs:
            plt.close(fig)

    def run_display(self, *args, **kwargs):
        res = display_lgs_geometry(*args, **{**KWARGS, **kwargs})
        self.figs.append(res['fig'])
        return res

    def assert_points_to_launcher(self, res):
        centres = sub_aperture_centres(res['shown_index'], N_SUB, D)
        expected = angle_mod180(centres - res['launcher_wfs'][:2])
        np.testing.assert_array_less(diff_mod180(res['orientation_deg'], expected), 1e-6)

    def test_orientation_points_to_launcher(self):
        res = self.run_display(D, N_SUB, [30.0, 0.0, 0.0])
        self.assert_points_to_launcher(res)
        # The farthest shown sub-aperture has the largest elongation
        centres = sub_aperture_centres(res['shown_index'], N_SUB, D)
        dist = np.hypot(*(centres - res['launcher_wfs'][:2]).T)
        beta = np.hypot(res['beta1'], res['beta2'])
        self.assertEqual(np.argmax(beta), np.argmax(dist))

    def test_launcher_in_telescope_frame(self):
        res = self.run_display(D, N_SUB, [30.0, 0.0, 0.0], wfs_rotation_in_deg=90.0)
        np.testing.assert_allclose(res['launcher_wfs'], [0.0, 30.0, 0.0], atol=1e-9)
        self.assert_points_to_launcher(res)

    def test_n_valid_and_n_show(self):
        centres = sub_aperture_centres(np.arange(N_SUB**2), N_SUB, D)
        n_inside = int(np.sum(np.hypot(*centres.T) <= D / 2))
        res = self.run_display(D, N_SUB, [30.0, 0.0, 0.0])
        self.assertEqual(res['n_valid'], n_inside)
        self.assertEqual(len(res['shown_index']), 4)

        idx = np.array([5, 22, 37, 44, 55, 61, 78, 90, 95])
        res = self.run_display(D, N_SUB, [30.0, 0.0, 0.0], sub_aps_index=idx, n_show=3)
        self.assertEqual(res['n_valid'], len(idx))
        self.assertEqual(len(res['shown_index']), 9)
        self.assertTrue(set(res['shown_index']) <= set(idx))


class TestLgsGeometryHelpers(unittest.TestCase):

    def test_sub_aperture_centres_2x2(self):
        c = sub_aperture_centres([0, 1, 2, 3], 2, 2.0)
        np.testing.assert_allclose(c, [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]])

    def test_elongation_magnitude(self):
        h, t, r = 90e3, 10e3, 30.0
        xy = r * np.array([[1.0, 0.0], [0.0, 1.0], [-np.cos(0.3), np.sin(0.3)]])  # all at distance r
        b1, b2 = geometric_elongation(xy, h, t)
        mag = np.hypot(b1, b2)
        exact = (np.arctan(r / (h - t / 2)) - np.arctan(r / (h + t / 2))) * ARCSEC
        # Exact for axis-aligned points; the per-axis formula is only
        # first-order consistent with the 2D one for oblique points
        np.testing.assert_allclose(mag[:2], exact, rtol=1e-9)
        np.testing.assert_allclose(mag, exact, rtol=1e-6)
        np.testing.assert_allclose(mag, r * t / h**2 * ARCSEC, rtol=1e-2)
        self.assertAlmostEqual(mag[0], 7.6, delta=0.1)
        # (beta1, beta2) is parallel or antiparallel to (x, y)
        cos = (b1 * xy[:, 0] + b2 * xy[:, 1]) / (mag * np.hypot(*xy.T))
        np.testing.assert_allclose(np.abs(cos), 1.0, atol=1e-6)

    def test_elongation_zero_at_launcher(self):
        b1, b2 = geometric_elongation(np.zeros((1, 2)), 90e3, 10e3)
        np.testing.assert_allclose([b1, b2], 0.0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
