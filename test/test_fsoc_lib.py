import specula
specula.init(0)  # Default target device

import unittest
from specula.lib import fsoc_lib
from test.specula_testlib import cpu_and_gpu


class TestFSOClib(unittest.TestCase):

    @cpu_and_gpu
    def setUp(self, target_device_idx, xp):
        """Set up common test data"""
        self.object_speed = 7.6e3
        self.zenith_angle = 75.0
        self.object_height = 400e3
        self.paa = 10.457985086989297

    @cpu_and_gpu
    def test_calc_paa(self, target_device_idx, xp):
        """Test if PAA is calculated correctly"""
        paa = fsoc_lib.calc_paa(self.object_speed, xp)
        xp.testing.assert_allclose(paa, self.paa, rtol=1e-5, atol=1e-8,
                                   err_msg="Effective wind speed does not match manual calculation")

    @cpu_and_gpu
    def test_calc_effective_wind_speed_same_direction(self, target_device_idx, xp):
        """Test if effective wind speed is calculated correctly for object and wind directions equal"""
        object_dir = 90.
        atmo_heights = xp.array([955.53979, 7300.5816, 12353.543, 16227.363, 21897.079])
        wind_speed = xp.array([10., 10., 10., 10., 10.])
        wind_dir = xp.array([90., 90., 90., 90., 90.])

        effective_wind, effective_dir = fsoc_lib.calc_effective_wind_speed(atmo_heights, wind_speed, wind_dir,
                                                                           self.object_height, self.object_speed,
                                                                           object_dir, xp)
        arcsec2rad = xp.pi / (3600. * 180.)
        t_light = self.object_height / 299792458
        speed_at_layer = xp.tan(self.paa * arcsec2rad) * atmo_heights / t_light
        true_wind = wind_speed + speed_at_layer
        xp.testing.assert_allclose(effective_wind, true_wind, rtol=1e-5, atol=1e-8,
                                   err_msg="Effective wind speed does not match manual calculation")
        xp.testing.assert_allclose(effective_dir.round(decimals=0), wind_dir, atol=1,
                                   err_msg="Effective wind direction does not match manual calculation")

    @cpu_and_gpu
    def test_calc_effective_wind_speed_opposite_direction(self, target_device_idx, xp):
        """Test if effective wind speed is calculated correctly for object and wind directions opposite"""
        object_dir = -90.
        atmo_heights = xp.array([955.53979, 7300.5816, 12353.543, 16227.363, 21897.079])
        wind_speed = xp.array([10., 10., 10., 10., 10.])
        wind_dir = xp.array([90., 90., 90., 90., 90.])

        effective_wind, effective_dir = fsoc_lib.calc_effective_wind_speed(atmo_heights, wind_speed, wind_dir,
                                                                           self.object_height, self.object_speed,
                                                                           object_dir, xp)
        arcsec2rad = xp.pi / (3600. * 180.)
        t_light = self.object_height / 299792458
        speed_at_layer = xp.tan(self.paa * arcsec2rad) * atmo_heights / t_light
        true_wind = speed_at_layer - effective_wind
        xp.testing.assert_allclose(true_wind, wind_speed, atol=1,
                                   err_msg="Effective wind speed does not match manual calculation")

    @cpu_and_gpu
    def test_timing_uplink_downlink(self, target_device_idx, xp):
        """Test if timing in upwards and downwards propagations is calculated correctly"""
        atmo_heights = xp.array([955.53979, 7300.5816, 12353.543, 16227.363, 21897.079])
        delta_time_up, delta_time_down = fsoc_lib.calc_timing_uplink_downlink(self.zenith_angle, self.object_height,
                                                                              atmo_heights, xp)

        true_down = xp.array([1.23149270e-05, 9.40893625e-05, 1.59211560e-04, 2.09137069e-04, 2.82207955e-04])
        true_up = xp.array([0.01029803, 0.01021625, 0.01015113, 0.0101012, 0.01002813])

        xp.testing.assert_allclose(delta_time_up, true_up, atol=1,
                                   err_msg="Delta time up does not match manual calculation")
        xp.testing.assert_allclose(delta_time_down, true_down, atol=1,
                                   err_msg="Delta time up does not match manual calculation")
