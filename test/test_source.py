

import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np

from specula.data_objects.source import Source
from test.specula_testlib import cpu_and_gpu

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
                    zeroPoint = 0.1,
                    error_coord = (0.2, 0.3),
                    target_device_idx=target_device_idx)
        
        source.save(self.filename)
        source2 = Source.restore(self.filename)

        np.testing.assert_array_equal(source.polar_coordinates, source2.polar_coordinates)
        assert source.height == source2.height
        assert source.magnitude == source2.magnitude
        assert source.wavelengthInNm == source2.wavelengthInNm
        assert source.zeroPoint == source2.zeroPoint
        assert source.band == source2.band
                
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

