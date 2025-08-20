
import specula
specula.init(0)  # Default target device

import tempfile
import os
import numpy as np
import unittest

from specula.data_objects.layer import Layer
from specula import cpuArray
from test.specula_testlib import cpu_and_gpu

class TestLayer(unittest.TestCase):

    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1

        layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=0,
                      shiftXYinPixel=(0.1, 0.2), rotInDeg=3, magnification=4.0,
                      target_device_idx=target_device_idx)

        hdr = layer.get_fits_header()

        assert hdr['VERSION'] == 1
        assert hdr['OBJ_TYPE'] == 'Layer'
        assert hdr['DIMX'] == pixel_pupil
        assert hdr['DIMY'] == pixel_pupil
        assert hdr['PIXPITCH'] == pixel_pitch
        assert hdr['HEIGHT'] == 0
        assert hdr['SHIFTX'] == 0.1
        assert hdr['SHIFTY'] == 0.2
        assert hdr['ROTATION'] == 3
        assert hdr['MAGNIFIC'] == 4.0

    @cpu_and_gpu
    def test_float(self, target_device_idx, xp):
        '''Test that precision=1 results in a single-precision layer'''

        pixel_pupil = 10
        pixel_pitch = 0.1

        layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=0,
                      shiftXYinPixel=(0.1, 0.2), rotInDeg=3, magnification=4.0,
                      target_device_idx=target_device_idx, precision=1)

        assert layer.field.dtype == np.float32
        assert layer.ef_at_lambda(500.0).dtype == np.complex64

    @cpu_and_gpu
    def test_double(self, target_device_idx, xp):
        '''Test that precision=0 results in a double-precision layer'''

        pixel_pupil = 10
        pixel_pitch = 0.1

        layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=0,
                      shiftXYinPixel=(0.1, 0.2), rotInDeg=3, magnification=4.0,
                      target_device_idx=target_device_idx, precision=0)

        assert layer.field.dtype == np.float64
        assert layer.ef_at_lambda(500.0).dtype == np.complex128

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        pixel_pupil = 10
        pixel_pitch = 0.1
        layer = Layer(pixel_pupil, pixel_pupil, pixel_pitch, height=1.23,
                      shiftXYinPixel=(0.1, 0.2), rotInDeg=3, magnification=4.0,
                      target_device_idx=target_device_idx)
        layer.field[0][:] = xp.ones((pixel_pupil, pixel_pupil))
        layer.field[1][:] = xp.arange(pixel_pupil * pixel_pupil, dtype=layer.dtype).reshape((pixel_pupil, pixel_pupil))

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "layer_test.fits")
            layer.save(fname)
            layer2 = Layer.restore(fname, target_device_idx=target_device_idx)

            assert np.allclose(cpuArray(layer.field[0]), cpuArray(layer2.field[0]))
            assert np.allclose(cpuArray(layer.field[1]), cpuArray(layer2.field[1]))
            assert layer2.height == layer.height
            assert np.allclose(cpuArray(layer2.shiftXYinPixel), cpuArray(layer.shiftXYinPixel))
            assert layer2.rotInDeg == layer.rotInDeg
            assert layer2.magnification == layer.magnification