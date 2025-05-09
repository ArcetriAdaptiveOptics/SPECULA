

import specula
specula.init(0)  # Default target device

import unittest

from specula.data_objects.pixels import Pixels

from test.specula_testlib import cpu_and_gpu

class TestPixels(unittest.TestCase):
   
    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):
        
        pixels = Pixels(10, 10, target_device_idx=target_device_idx)
        id_pixels_before = id(pixels.pixels)
        
        pixels.set_value(xp.ones((10, 10), dtype=xp.float32))
        id_pixels_after = id(pixels.pixels)

        assert id_pixels_before == id_pixels_after
        
