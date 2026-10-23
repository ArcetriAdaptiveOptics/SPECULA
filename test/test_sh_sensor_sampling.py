import unittest

import specula
specula.init(-1)

import numpy as np

from specula import RAD2ASEC
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH, choose_fov_resolution


class TestSHSensorSampling(unittest.TestCase):
    def test_resolution_bounds_both_fields_of_view(self):
        # The requested optical FoV and the detector FoV need not be equal.
        cases = ((0.912, 1.275, 5.1, 4),
                 (0.5, 0.2, 2.0, 6),
                 (0.5, 0.2, 2.0, 8))
        for turbulence_pxscale, sensor_pxscale, wanted_fov, npx in cases:
            with self.subTest(case=(turbulence_pxscale, sensor_pxscale, wanted_fov, npx)):
                resolution = choose_fov_resolution(turbulence_pxscale, sensor_pxscale,
                                                   wanted_fov, npx)
                for fov in (wanted_fov, sensor_pxscale * npx):
                    sampled_fov = 2 * round(fov / resolution / 2) * resolution
                    self.assertLess(abs(sampled_fov / fov - 1), 0.02)

    def test_coarse_sensor_uses_sampling_search(self):
        # AOF-like lambda/d and detector sampling from issue #766, at reduced
        # pupil size. Geometry depends on subaperture width, not pupil size.
        turbulence_pxscale = 0.912
        wavelength_nm = 589
        subap_width = wavelength_nm * 1e-9 * RAD2ASEC / turbulence_pxscale
        ef = ElectricField(80, 80, subap_width / 20, S0=1, target_device_idx=-1)
        sh = SH(wavelengthInNm=wavelength_nm, subap_wanted_fov=5.1,
                sensor_pxscale=1.275, subap_on_diameter=4, subap_npx=4,
                target_device_idx=-1)
        sh._set_in_ef(ef)
        self.assertLess(abs(sh.sensor_pxscale_effective / 1.275 - 1), 0.02)
        self.assertEqual(sh._cutsize % 2, 0)
        self.assertGreaterEqual(sh._fft_size, sh._cutsize)
        self.assertEqual(sh._ovs_ef_size % 8, 0)

    def test_explicit_turbulence_sampling_is_preserved(self):
        ef = ElectricField(80, 80, 0.05, S0=1, target_device_idx=-1)
        sh = SH(wavelengthInNm=500, subap_wanted_fov=5.1,
                sensor_pxscale=1.275, subap_on_diameter=4, subap_npx=4,
                set_fov_res_to_turbpxsc=True, target_device_idx=-1)
        sh._set_in_ef(ef)
        expected = 500e-9 / (20 * 0.05) * RAD2ASEC
        self.assertAlmostEqual(sh._fov_resolution_arcsec, expected)

    def test_no_admissible_candidate_keeps_finite_fallback(self):
        resolution = choose_fov_resolution(0.912, 1.275, 5.1, 4, n_try=1)
        self.assertTrue(np.isfinite(resolution))
        self.assertEqual(resolution, 0.912 / 2)
