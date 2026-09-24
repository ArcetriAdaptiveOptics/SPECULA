import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.source import Source
from specula.lib.source_flux import phot_density_from_source_params, flux_per_pixel

# Constants used by n_phot (c is approximated there as 3e8)
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
E0_V_BAND = 392e-10  # A0 0-mag brightness in V band [J/s/m^2/um], default model


class TestSourceFlux(unittest.TestCase):

    def test_phot_density_analytic_value(self):
        '''V band, mag 0: E0 * lambda / (h c) over a 1 nm (1e-3 um) bandwidth.'''
        lambda_m = 550e-9
        expected = lambda_m * 1e-3 * E0_V_BAND / (H_PLANCK * C_LIGHT)
        res = phot_density_from_source_params(0, 550, band='V')
        self.assertAlmostEqual(res / expected, 1.0, places=10)

    def test_phot_density_matches_source_object(self):
        '''Source.phot_density() delegates to the lib function: guards the wiring.'''
        cases = [
            dict(magnitude=8, wavelengthInNm=750, band='', zero_point=0),
            dict(magnitude=12.5, wavelengthInNm=1650, band='H', zero_point=0),
            dict(magnitude=5, wavelengthInNm=589, band='', zero_point=1e-8),
        ]
        for c in cases:
            with self.subTest(**c):
                src = Source(polar_coordinates=[0, 0], **c)
                self.assertAlmostEqual(
                    phot_density_from_source_params(**c) / src.phot_density(), 1.0, places=12)

    def test_phot_density_magnitude_scaling(self):
        '''+2.5 mag must reduce the flux by exactly a factor 10.'''
        f0 = phot_density_from_source_params(10, 700)
        f1 = phot_density_from_source_params(12.5, 700)
        self.assertAlmostEqual(f0 / f1, 10.0, places=10)

    def test_zero_point_overrides_table(self):
        f_table = phot_density_from_source_params(0, 550, band='V')
        f_zp = phot_density_from_source_params(0, 550, band='V', zero_point=2 * E0_V_BAND)
        self.assertAlmostEqual(f_zp / f_table, 2.0, places=10)

    def test_flux_per_pixel_formula(self):
        mag, wl = 10, 750
        area, bw, dt = 50.0, 300.0, 1e-3
        n_pix, thr, qe, frac = 16, 0.4, 0.8, 0.7

        density = phot_density_from_source_params(mag, wl)
        expected = density * area * bw * dt * thr * qe * frac / n_pix

        res = flux_per_pixel(mag, wl, area, bw, dt, n_pixels=n_pix,
                             throughput=thr, quantum_efficiency=qe,
                             fraction_on_pixel=frac)
        self.assertAlmostEqual(res / expected, 1.0, places=12)

    def test_flux_per_pixel_defaults(self):
        '''Default efficiencies are unity and a single pixel collects everything.'''
        density = phot_density_from_source_params(10, 750)
        res = flux_per_pixel(10, 750, collecting_area_m2=2.0, bandwidth_nm=10.0,
                             integration_time_s=0.5)
        self.assertAlmostEqual(res / (density * 2.0 * 10.0 * 0.5), 1.0, places=12)

    def test_flux_per_pixel_invalid_inputs(self):
        base = dict(magnitude=10, wavelengthInNm=750, collecting_area_m2=1.0,
                    bandwidth_nm=10.0, integration_time_s=1.0)
        bad = [
            dict(collecting_area_m2=0),
            dict(bandwidth_nm=-1),
            dict(integration_time_s=0),
            dict(n_pixels=0),
            dict(throughput=1.1),
            dict(quantum_efficiency=-0.1),
            dict(fraction_on_pixel=2),
        ]
        for override in bad:
            with self.subTest(**override):
                with self.assertRaises(ValueError):
                    flux_per_pixel(**{**base, **override})


if __name__ == '__main__':
    unittest.main()
