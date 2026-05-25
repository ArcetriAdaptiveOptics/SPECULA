import unittest
import numpy as np
import tempfile
import os
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.phasescreen import Phasescreen
from test.specula_testlib import cpu_and_gpu


class TestPhasescreen(unittest.TestCase):
    def setUp(self):
        self.dimxy = (5, 4)
        self.shape = (self.dimxy[1], self.dimxy[0])  # (dimy, dimx) for array shape
    
    def _get_phasescreen(self, target_device_idx):
        return Phasescreen(*self.dimxy, L0=40, seed=1, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_initialization_and_get_value(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        self.assertEqual(obj.phasescreen.shape, self.shape)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), np.zeros(self.shape))

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        new_val = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(new_val)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(new_val))

    @cpu_and_gpu
    def test_set_value_wrong_shape(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        wrong_shape = xp.random.rand(2, 2).astype(xp.float32)
        with self.assertRaises(AssertionError):
            obj.set_value(wrong_shape)

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        hdr = obj.get_fits_header()
        self.assertIsInstance(hdr, fits.Header)
        self.assertEqual(hdr["VERSION"], 1)
        self.assertEqual(hdr["OBJ_TYPE"], "Phasescreen")
        self.assertEqual(hdr["DIMX"], self.shape[1])
        self.assertEqual(hdr["DIMY"], self.shape[0])

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_phasescreen.fits")
            obj.save(filename, overwrite=True)

            # Check FITS file structure before restore
            with fits.open(filename) as hdul:
                # There should be exactly two HDUs
                self.assertEqual(len(hdul), 1)

                # Primary HDU: header only, no data
                self.assertEqual(hdul[0].header["OBJ_TYPE"], "Phasescreen") # pylint: disable=no-member
                self.assertEqual(hdul[0].header["DIMX"], self.shape[1])   # pylint: disable=no-member
                self.assertEqual(hdul[0].header["DIMY"], self.shape[0])   # pylint: disable=no-member
                self.assertEqual(hdul[0].data.shape, self.shape)  # pylint: disable=no-member

                # Second HDU: phasescreen data
                np.testing.assert_array_equal(hdul[0].data, cpuArray(data))    # pylint: disable=no-member

            # Now restore and check data consistency
            restored = Phasescreen.restore(filename, target_device_idx=target_device_idx)
            np.testing.assert_array_equal(cpuArray(restored.get_value()), cpuArray(data))
            self.assertEqual(restored.phasescreen.shape, obj.phasescreen.shape)

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
                Phasescreen.restore(filename, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_from_header_and_invalid_version(self, target_device_idx, xp):
        # Valid header should work
        hdr = fits.Header()
        hdr["VERSION"] = 1
        hdr["OBJ_TYPE"] = "Phasescreen"
        hdr["DIMX"], hdr["DIMY"] = self.dimxy
        hdr['L0'] = 42
        hdr['SEED'] = 1
        phasescreen = Phasescreen.from_header(hdr, target_device_idx=target_device_idx)
        self.assertEqual(phasescreen.phasescreen.shape, self.shape)

        # Invalid version should raise
        hdr["VERSION"] = 99
        with self.assertRaises(ValueError):
            Phasescreen.from_header(hdr, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_array_for_display(self, target_device_idx, xp):
        obj = self._get_phasescreen(target_device_idx)
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(data)
        np.testing.assert_array_equal(cpuArray(obj.array_for_display()), cpuArray(data))


    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):

        phasescreen = Phasescreen(dimx=2, dimy=3, L0=4, seed=5, target_device_idx=target_device_idx)

        hdr = phasescreen.get_fits_header()

        assert hdr['OBJ_TYPE'] == 'Phasescreen'
        assert hdr['VERSION'] == 1
        assert hdr['DIMX'] == 2
        assert hdr['DIMY'] == 3
        assert hdr['L0'] == 4
        assert hdr['SEED'] == 5
