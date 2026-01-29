import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np

from specula import ASEC2RAD
from specula.data_objects.source import Source
from test.specula_testlib import cpu_and_gpu

degree2rad = np.pi / 180.

class TestSource(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_source.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        source = Source(
                    polar_coordinates = [1.0, 2.0],
                    magnitude = 3.0,
                    wavelengthInNm = 750.0,
                    height = 88e3,
                    band = 'K',
                    zero_point = 0.1,
                    error_coord = (0.2, 0.3),
                    target_device_idx=target_device_idx)
        
        source.save(self.filename)
        source2 = Source.restore(self.filename)

        np.testing.assert_array_equal(source.polar_coordinates, source2.polar_coordinates)
        assert source.height == source2.height
        assert source.magnitude == source2.magnitude
        assert source.wavelengthInNm == source2.wavelengthInNm
        assert source.zero_point == source2.zero_point
        assert source.band == source2.band
                
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_properties(self, target_device_idx, xp):
        source = Source(
                    polar_coordinates = [1.0, 2.0],
                    magnitude = 3.0,
                    wavelengthInNm = 750.0,
                    height = 88e3,
                    band = 'K',
                    zero_point = 0.1,
                    error_coord = (0.0, 0.0),
                    target_device_idx=target_device_idx)

        assert source.r_arcsec == 1.0
        assert source.phi_deg == 2.0
        assert source.r == 1.0 * ASEC2RAD
        assert source.phi == 2.0 * degree2rad
        assert source.height == 88e3
        assert source.x_coord == np.sin(source.r) * source.height  * np.cos(source.phi)
        assert source.y_coord == np.sin(source.r) * source.height  * np.sin(source.phi)

    @cpu_and_gpu
    def test_error_coord(self, target_device_idx, xp):
        source = Source(
                    polar_coordinates = [1.0, 2.0],
                    magnitude = 3.0,
                    wavelengthInNm = 750.0,
                    height = 88e3,
                    band = 'K',
                    zero_point = 0.1,
                    error_coord = (0.2, 0.3),
                    target_device_idx=target_device_idx)

        assert source.r_arcsec == 1.2
        assert source.phi_deg == 2.3
