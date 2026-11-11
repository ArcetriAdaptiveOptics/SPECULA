import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.lib import fits_io
from specula.lib.fits_io import load_fits_array

from test.specula_testlib import cpu_and_gpu


class TestLoadFitsArray(unittest.TestCase):

    def setUp(self):
        fd, self.filename = tempfile.mkstemp(suffix='.fits')
        os.close(fd)

    def tearDown(self):
        os.unlink(self.filename)

    def _write(self, *arrays):
        hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(a) for a in arrays]
        fits.HDUList(hdus).writeto(self.filename, overwrite=True)

    @cpu_and_gpu
    def test_dtypes(self, target_device_idx, xp):
        rng = np.random.default_rng(1)
        float_dtype = specula.cpu_float_dtype_list[specula.global_precision]
        cases = [(rng.random((37, 53)).astype(np.float32), float_dtype),
                 (rng.random((37, 53)), float_dtype),
                 (rng.integers(-30000, 30000, (37, 53)).astype(np.int16), np.int16),
                 (rng.integers(-2**31, 2**31 - 1, (37, 53)).astype(np.int32), np.int32),
                 (rng.integers(0, 255, (37, 53)).astype(np.uint8), np.uint8)]
        for data, dtype in cases:
            self._write(data)
            out = load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx)
            self.assertIsInstance(out, xp.ndarray)
            self.assertEqual(out.dtype, dtype)
            self.assertTrue(out.dtype.isnative)
            np.testing.assert_array_equal(cpuArray(out), data.astype(dtype))

    def test_read_empty_array(self):
        '''An empty array, also transposed with empty rows (no division by the row length)'''
        out = fits_io.read_from_file(self.filename, 0, (3, 0), np.float32, np.float32,
                                     byteswap=True, transpose=True)
        self.assertEqual(out.shape, (0, 3))
        out = fits_io.read_from_file(self.filename, 0, (0, 4), np.float32, np.float32, byteswap=True)
        self.assertEqual(out.shape, (0, 4))

    @cpu_and_gpu
    def test_truncated_file(self, target_device_idx, xp):
        data = np.arange(1000 * 77, dtype=np.float32).reshape(1000, 77)
        self._write(data)
        with open(self.filename, 'r+b') as f:
            f.truncate(os.path.getsize(self.filename) - 2880 - 4096)
        with self.assertRaises(EOFError):
            load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_multiple_chunks(self, target_device_idx, xp):
        data = np.arange(1000 * 77, dtype=np.float32).reshape(1000, 77)
        data64 = data.astype(np.float64) + 0.25
        self._write(data, data64)
        # Chunks much smaller than the array, with a partial last chunk
        with patch.object(fits_io, '_CHUNK_BYTES', 4096):
            out = load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx, precision=1)
            out64 = load_fits_array(self.filename, ext=2, target_device_idx=target_device_idx, precision=0)
        np.testing.assert_array_equal(cpuArray(out), data)
        np.testing.assert_array_equal(cpuArray(out64), data64)
        self.assertEqual(out64.dtype, np.float64)

    @unittest.skipIf(specula.cp is None, 'GPU upload only')
    def test_staging_chunks_not_cached(self):
        '''On GPU, the default memory pool grows only by the result: the chunks
        used for the upload come from a temporary pool'''
        data = np.arange(2 * 1024 * 1024, dtype=np.float32).reshape(2048, 1024)   # 8 MB
        self._write(data)
        pool = specula.cp.get_default_memory_pool()
        pool.free_all_blocks()
        before = pool.total_bytes()
        with patch.object(fits_io, '_CHUNK_BYTES', 1 << 20):
            out = load_fits_array(self.filename, ext=1, target_device_idx=0, precision=1)
        self.assertLessEqual(pool.total_bytes() - before, out.nbytes + 1024)
        np.testing.assert_array_equal(cpuArray(out), data)

    @cpu_and_gpu
    def test_transpose(self, target_device_idx, xp):
        '''Transposed in C order: chunks of whole rows (with a partial last chunk), and
        the astropy path for scaled data; 3D data is rejected'''
        data = np.arange(1000 * 77, dtype=np.float32).reshape(1000, 77)
        scaled = np.arange(12, dtype=np.uint16).reshape(3, 4)
        self._write(data, scaled, np.zeros((2, 3, 4), dtype=np.float32))
        with patch.object(fits_io, '_CHUNK_BYTES', 4096):
            out = load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx,
                                  precision=1, transpose=True)
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(cpuArray(out), data.T)
        out = load_fits_array(self.filename, ext=2, target_device_idx=target_device_idx, transpose=True)
        self.assertTrue(out.flags.c_contiguous)
        np.testing.assert_array_equal(cpuArray(out), scaled.T)
        with self.assertRaises(ValueError):
            load_fits_array(self.filename, ext=3, target_device_idx=target_device_idx, transpose=True)

    @cpu_and_gpu
    def test_integer_data_with_precision(self, target_device_idx, xp):
        # Plain int16 (fast path on GPU) and uint16 (BZERO-scaled, astropy path)
        for data in (np.arange(6, dtype=np.int16).reshape(2, 3),
                     np.arange(6, dtype=np.uint16).reshape(2, 3)):
            self._write(data)
            for precision in (0, 1):
                with self.assertRaises(ValueError):
                    load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx, precision=precision)
            out = load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx, precision=None)
            self.assertEqual(out.dtype, data.dtype)
            np.testing.assert_array_equal(cpuArray(out), data)

    @cpu_and_gpu
    def test_no_data(self, target_device_idx, xp):
        # Empty extension followed by another one, and as the last one
        hdus = [fits.PrimaryHDU(), fits.ImageHDU(), fits.ImageHDU(np.ones((2, 2), dtype=np.float32)),
                fits.ImageHDU()]
        fits.HDUList(hdus).writeto(self.filename, overwrite=True)
        self.assertIsNone(load_fits_array(self.filename, ext=0, target_device_idx=target_device_idx))
        self.assertIsNone(load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx))
        self.assertIsNone(load_fits_array(self.filename, ext=3, target_device_idx=target_device_idx))
        np.testing.assert_array_equal(cpuArray(load_fits_array(self.filename, ext=2, target_device_idx=target_device_idx)),
                                      np.ones((2, 2)))

    @cpu_and_gpu
    def test_scaled_data(self, target_device_idx, xp):
        # astropy stores uint16 as int16 with BZERO=32768
        data = np.array([[0, 1, 40000, 65535]], dtype=np.uint16)
        self._write(data)
        out = load_fits_array(self.filename, ext=1, target_device_idx=target_device_idx)
        self.assertIsInstance(out, xp.ndarray)
        np.testing.assert_array_equal(cpuArray(out), data)

    @cpu_and_gpu
    def test_restore_objects(self, target_device_idx, xp):
        from specula.data_objects.recmat import Recmat
        from specula.data_objects.ifunc import IFunc

        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        Recmat(data, norm_factor=1.0, target_device_idx=target_device_idx).save(self.filename, overwrite=True)
        rec = Recmat.restore(self.filename, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(rec.recmat), data)

        mask = np.ones((2, 2), dtype=np.float32)
        IFunc(ifunc=data[:, :4], mask=mask, target_device_idx=target_device_idx).save(self.filename, overwrite=True)
        ifunc = IFunc.restore(self.filename, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(ifunc.influence_function), data[:, :4])


if __name__ == '__main__':
    unittest.main()
