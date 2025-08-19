
import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.data_objects.slopes import Slopes

from test.specula_testlib import cpu_and_gpu

class TestSlopes(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_slopes.fits')

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        id_slopes_before = id(slopes.slopes)
        
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        
        id_slopes_after = id(slopes.slopes)

        assert id_slopes_before == id_slopes_after
        
    @cpu_and_gpu
    def test_set_value_shape_mismatch(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            new_slopes_data = xp.ones(11)
            slopes.set_value(new_slopes_data)
            
    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        expected_value = xp.ones(10)
        slopes.set_value(expected_value)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(expected_value))
        
    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(new_slopes_data))

    @cpu_and_gpu
    def test_slopes_save_restore_roundtrip(self, target_device_idx, xp):
        
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        slopes.save(self.filename)

        slopes2 = Slopes.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(slopes2.slopes))
        np.testing.assert_array_equal(cpuArray(slopes.indicesX), cpuArray(slopes2.indicesX))
        np.testing.assert_array_equal(cpuArray(slopes.indicesY), cpuArray(slopes2.indicesY))


    @cpu_and_gpu
    def test_slopes_save_restore_roundtrip_version2(self, target_device_idx, xp):
        
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        slopes.save(self.filename)

        with fits.open(self.filename, mode="update") as f:
            f[0].header["VERSION"] = 2
            del f[0].header["LENGTH"]

        slopes2 = Slopes.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(slopes2.slopes))
        np.testing.assert_array_equal(cpuArray(slopes.indicesX), cpuArray(slopes2.indicesX))
        np.testing.assert_array_equal(cpuArray(slopes.indicesY), cpuArray(slopes2.indicesY))
