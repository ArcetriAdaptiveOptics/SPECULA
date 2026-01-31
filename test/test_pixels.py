import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np

from astropy.io import fits

from specula import cpuArray
from specula.data_objects.pixels import Pixels

from test.specula_testlib import cpu_and_gpu

class TestPixels(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_pixels.fits')

    @cpu_and_gpu
    def test_pixels_save_restore_roundtrip(self, target_device_idx, xp):
        
        pix_data = xp.arange(9).reshape((3,3))
        pix = Pixels(3, 3, bits=16, signed=0, target_device_idx=target_device_idx)
        pix.set_value(pix_data)
        pix.save(self.filename)

        pix2 = Pixels.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(pix.pixels), cpuArray(pix2.pixels))
        assert pix.bpp == pix2.bpp
        assert pix.dtype == pix2.dtype
        assert pix.bytespp == pix2.bytespp
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):
        
        pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        id_pixels_before = id(pixels.pixels)
        
        pixels.set_value(xp.ones((10, 10), dtype=xp.float32))
        id_pixels_after = id(pixels.pixels)

        assert id_pixels_before == id_pixels_after
    
    @cpu_and_gpu
    def test_set_value_shape_mismatch(self, target_device_idx, xp):
        pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        expected_value = xp.ones((10, 10), dtype=xp.float32)
        pixels.set_value(expected_value)

        np.testing.assert_array_equal(cpuArray(pixels.pixels), cpuArray(expected_value))

    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        expected_value = xp.ones((10, 10), dtype=xp.float32)
        pixels.set_value(expected_value)

        value = pixels.get_value()
        np.testing.assert_array_equal(cpuArray(value), cpuArray(expected_value))
        
    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):
        
        pixels = Pixels(6, 5, bits=8, signed=1, target_device_idx=target_device_idx)
        hdr = pixels.get_fits_header()
        
        assert type(hdr) is fits.Header
        assert hdr['VERSION'] == 1
        assert hdr['OBJ_TYPE'] == 'Pixels'
        assert hdr['TYPE'] == 'int8'
        assert hdr['BPP'] == 8
        assert hdr['BYTESPP'] == 1
        assert hdr['SIGNED'] == 1
        assert hdr['DIMX'] == 6
        assert hdr['DIMY'] == 5

    @cpu_and_gpu
    def test_set_with_invalid_shape(self, target_device_idx, xp):
        pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            pixels.set_value(xp.ones((5, 5), dtype=xp.float32))
