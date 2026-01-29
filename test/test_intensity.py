import unittest
import numpy as np
import tempfile
import os
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.intensity import Intensity
from test.specula_testlib import cpu_and_gpu


class TestIntensity(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 5)

    @cpu_and_gpu
    def test_initialization_and_get_value(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        self.assertEqual(obj.i.shape, self.shape)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), np.zeros(self.shape))

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        new_val = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(new_val)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(new_val))

    @cpu_and_gpu
    def test_set_value_wrong_shape(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        wrong_shape = xp.random.rand(2, 2).astype(xp.float32)
        with self.assertRaises(AssertionError):
            obj.set_value(wrong_shape)

    @cpu_and_gpu
    def test_sum(self, target_device_idx, xp):
        obj1 = Intensity(*self.shape, target_device_idx=target_device_idx)
        obj2 = Intensity(*self.shape, target_device_idx=target_device_idx)
        obj1.set_value(xp.ones(self.shape, dtype=xp.float32))
        obj2.set_value(xp.ones(self.shape, dtype=xp.float32) * 2)

        obj1.sum(obj2, factor=0.5)
        expected = xp.ones(self.shape, dtype=xp.float32) + 0.5 * (xp.ones(self.shape) * 2)
        np.testing.assert_array_equal(cpuArray(obj1.get_value()), cpuArray(expected))

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        hdr = obj.get_fits_header()
        self.assertIsInstance(hdr, fits.Header)
        self.assertEqual(hdr["VERSION"], 1)
        self.assertEqual(hdr["OBJ_TYPE"], "Intensity")
        self.assertEqual(hdr["DIMX"], self.shape[0])
        self.assertEqual(hdr["DIMY"], self.shape[1])

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_intensity.fits")
            obj.save(filename, overwrite=True)

            # Check FITS file structure before restore
            with fits.open(filename) as hdul:
                # There should be exactly two HDUs
                self.assertEqual(len(hdul), 2)

                # Primary HDU: header only, no data
                self.assertEqual(hdul[0].header["OBJ_TYPE"], "Intensity") # pylint: disable=no-member
                self.assertEqual(hdul[0].header["DIMX"], self.shape[0])   # pylint: disable=no-member
                self.assertEqual(hdul[0].header["DIMY"], self.shape[1])   # pylint: disable=no-member
                self.assertIsNone(hdul[0].data)                           # pylint: disable=no-member

                # Second HDU: intensity data
                self.assertEqual(hdul[1].name, "INTENSITY")       # pylint: disable=no-member
                self.assertEqual(hdul[1].data.shape, self.shape)  # pylint: disable=no-member
                np.testing.assert_array_equal(hdul[1].data, cpuArray(data))    # pylint: disable=no-member

            # Now restore and check data consistency
            restored = Intensity.restore(filename, target_device_idx=target_device_idx)
            np.testing.assert_array_equal(cpuArray(restored.get_value()), cpuArray(data))
            self.assertEqual(restored.i.shape, obj.i.shape)

    @cpu_and_gpu
    def test_restore_invalid_obj_type(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "invalid_obj.fits")
            hdr = fits.Header()
            hdr["VERSION"] = 1
            hdr["OBJ_TYPE"] = "WrongType"
            hdr["DIMX"], hdr["DIMY"] = self.shape
            hdu = fits.PrimaryHDU(header=hdr)
            hdu.writeto(filename, overwrite=True)

            with self.assertRaises(ValueError):
                Intensity.restore(filename, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_from_header_and_invalid_version(self, target_device_idx, xp):
        # Valid header should work
        hdr = fits.Header()
        hdr["VERSION"] = 1
        hdr["OBJ_TYPE"] = "Intensity"
        hdr["DIMX"], hdr["DIMY"] = self.shape
        intensity = Intensity.from_header(hdr, target_device_idx=target_device_idx)
        self.assertEqual(intensity.i.shape, self.shape)

        # Invalid version should raise
        hdr["VERSION"] = 99
        with self.assertRaises(ValueError):
            Intensity.from_header(hdr, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_array_for_display(self, target_device_idx, xp):
        obj = Intensity(*self.shape, target_device_idx=target_device_idx)
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(data)
        np.testing.assert_array_equal(cpuArray(obj.array_for_display()), cpuArray(data))


    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):

        intensity = Intensity(dimx=2, dimy=3, target_device_idx=target_device_idx)

        hdr = intensity.get_fits_header()

        assert hdr['OBJ_TYPE'] == 'Intensity'
        assert hdr['VERSION'] == 1
        assert hdr['DIMX'] == 2
        assert hdr['DIMY'] == 3
