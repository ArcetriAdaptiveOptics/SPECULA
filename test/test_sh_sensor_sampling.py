import unittest

import specula
specula.init(-1)

from specula import RAD2ASEC
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH


class TestSHSensorSampling(unittest.TestCase):

    def test_coarse_sensor_pixel_scale_is_exact(self):
        # AOF-like lambda/d and detector sampling from issue #766, at reduced
        # pupil size. The rounding of the FoV gave +7.3% on the pixel scale:
        # with the exact pixel integration it is the requested one.
        turbulence_pxscale = 0.912
        wavelength_nm = 589
        subap_width = wavelength_nm * 1e-9 * RAD2ASEC / turbulence_pxscale
        ef = ElectricField(80, 80, subap_width / 20, S0=1, target_device_idx=-1)
        sh = SH(wavelengthInNm=wavelength_nm, subap_wanted_fov=5.1,
                sensor_pxscale=1.275, subap_on_diameter=4, subap_npx=4,
                target_device_idx=-1)
        sh._set_in_ef(ef)
        self.assertEqual(sh.sensor_pxscale_effective, 1.275)
        self.assertAlmostEqual(sh.subap_real_fov_arcsec, 5.1)
        # Zero padding by 2, whatever the pixel scale
        self.assertEqual(sh._fft_size, 2 * sh._ovs_np_sub)

    def test_fft_period_covers_sensor_fov(self):
        # The FFT period (lambda / pixel pitch of the oversampled field) must
        # cover the sensor FoV, even with fov_ovs_coeff = 1
        ef = ElectricField(80, 80, 0.05, S0=1, target_device_idx=-1)
        sh = SH(wavelengthInNm=500, subap_wanted_fov=4.0, sensor_pxscale=1.0,
                subap_on_diameter=4, subap_npx=4, fov_ovs_coeff=1.0,
                target_device_idx=-1)
        sh._set_in_ef(ef)
        native_period = 500e-9 / 0.05 * RAD2ASEC     # 2.06 arcsec < 4 arcsec FoV
        self.assertGreater(sh._fov_ovs, 1)
        self.assertGreaterEqual(sh._fft_period_arcsec, 4.0)
        self.assertAlmostEqual(sh._fft_period_arcsec, native_period * sh._fov_ovs)

    def test_set_fov_res_to_turbpxsc_is_deprecated(self):
        with self.assertWarns(FutureWarning):
            SH(wavelengthInNm=500, subap_wanted_fov=5.1, sensor_pxscale=1.275,
               subap_on_diameter=4, subap_npx=4, set_fov_res_to_turbpxsc=True,
               target_device_idx=-1)

    def test_circular_field_stop_is_rejected(self):
        with self.assertRaises(ValueError):
            SH(wavelengthInNm=500, subap_wanted_fov=5.1, sensor_pxscale=1.275,
               subap_on_diameter=4, subap_npx=4, squaremask=False,
               target_device_idx=-1)
