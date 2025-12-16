import unittest
import numpy as np
import tempfile
import os
from astropy.io import fits

import specula
specula.init(0)  # Default target device
from specula import cpuArray
from specula.data_objects.m2c import M2C
from test.specula_testlib import cpu_and_gpu


class TestM2C(unittest.TestCase):
    def setUp(self):
        self.shape = (6, 4)  # 6 actuators × 4 modes

    def test_existing_m2c_file_with_overwrite(self):
        """Test that overwrite=True allows overwriting existing m2c files"""
        m2c_tag = 'test_m2c_overwrite'
        m2c_filename = f'{m2c_tag}.fits'
        m2c_path = os.path.join(tempfile.mkdtemp(), m2c_filename)
        with open(m2c_path, 'w') as f:
            f.write('')
        obj = M2C(m2c=np.random.rand(*self.shape))
        obj.save(m2c_path, overwrite=True)

    @cpu_and_gpu
    def test_initialization_and_get_value(self, target_device_idx, xp):
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(data))
        self.assertEqual(obj.m2c.shape, self.shape)

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        data = xp.zeros(self.shape, dtype=xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        new_val = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(new_val)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(new_val))

    @cpu_and_gpu
    def test_set_value_wrong_shape(self, target_device_idx, xp):
        data = xp.zeros(self.shape, dtype=xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        wrong_shape = xp.zeros((2, 2), dtype=xp.float32)
        with self.assertRaises(AssertionError):
            obj.set_value(wrong_shape)

    @cpu_and_gpu
    def test_nmodes_property(self, target_device_idx, xp):
        data = xp.zeros(self.shape, dtype=xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        self.assertEqual(obj.nmodes, self.shape[1])

    @cpu_and_gpu
    def test_set_nmodes(self, target_device_idx, xp):
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        obj.set_nmodes(2)
        self.assertEqual(obj.m2c.shape, (self.shape[0], 2))

    @cpu_and_gpu
    def test_cut_with_start_and_nmodes(self, target_device_idx, xp):
        data = xp.arange(np.prod(self.shape)).reshape(self.shape).astype(xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        obj.cut(start_mode=1, nmodes=3)
        expected = data[:, 1:3]
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(expected))

    @cpu_and_gpu
    def test_cut_with_idx_modes(self, target_device_idx, xp):
        data = xp.arange(np.prod(self.shape)).reshape(self.shape).astype(xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        obj.cut(idx_modes=[0, 2])
        expected = data[:, [0, 2]]
        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(expected))

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        data = xp.zeros(self.shape, dtype=xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)
        hdr = obj.get_fits_header()
        self.assertIsInstance(hdr, fits.Header)
        self.assertEqual(hdr["VERSION"], 1)

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        data = xp.random.rand(*self.shape).astype(xp.float32)
        obj = M2C(m2c=data, target_device_idx=target_device_idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_m2c.fits")
            obj.save(filename)

            # Check FITS file structure before restore
            with fits.open(filename) as hdul:
                # There should be exactly two HDUs
                self.assertEqual(len(hdul), 2)

                # Primary HDU: header contains version
                self.assertEqual(hdul[0].header["VERSION"], 1)
                self.assertIsNone(hdul[0].data)  # primary HDU has no data

                # Second HDU: M2C data
                self.assertEqual(hdul[1].data.shape, self.shape)
                np.testing.assert_array_equal(hdul[1].data, cpuArray(data))

            # Restore and compare data
            restored = M2C.restore(filename, target_device_idx=target_device_idx)
            np.testing.assert_array_equal(cpuArray(restored.get_value()), cpuArray(data))
            self.assertEqual(restored.m2c.shape, obj.m2c.shape)

    @cpu_and_gpu
    def test_restore_invalid_version(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "invalid_version.fits")
            hdr = fits.Header()
            hdr["VERSION"] = 99  # invalid version
            hdu = fits.PrimaryHDU(header=hdr, data=np.zeros(2))
            hdul = fits.HDUList([hdu, fits.ImageHDU(data=np.zeros(self.shape))])
            hdul.writeto(filename, overwrite=True)
            hdul.close()  # Force close for Windows

            with self.assertRaises(ValueError):
                M2C.restore(filename, target_device_idx=target_device_idx)
