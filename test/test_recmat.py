import os
import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula.data_objects.recmat import Recmat
from specula import cpuArray

from test.specula_testlib import cpu_and_gpu

class TestRecmat(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_recmat.fits')

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_initialization(self, target_device_idx, xp):
        """Test Recmat initialization with provided matrix."""
        data = xp.ones((4, 4), dtype=np.float32)
        obj = Recmat(data, norm_factor=2.0, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(obj.recmat), np.ones((4, 4)))
        self.assertEqual(obj.norm_factor, 2.0)

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        """Test get_value and set_value methods."""
        data = xp.arange(16).reshape((4, 4)).astype(np.float32)
        obj = Recmat(data.copy(), target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), np.arange(16).reshape((4, 4)))

        new_data = xp.ones((4, 4), dtype=np.float32)
        obj.set_value(new_data)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), np.ones((4, 4)))

    @cpu_and_gpu
    def test_set_value_shape_mismatch(self, target_device_idx, xp):
        """Test that set_value raises AssertionError for wrong shape."""
        data = xp.ones((4, 4), dtype=np.float32)
        obj = Recmat(data.copy(), target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            obj.set_value(xp.ones((2, 2), dtype=np.float32))

    @cpu_and_gpu
    def test_reduce_size(self, target_device_idx, xp):
        mat = xp.arange(30).reshape(6, 5)
        recmat = Recmat(mat, target_device_idx=target_device_idx)

        # Reduce modes
        recmat.reduce_size(2)
        assert recmat.recmat.shape == (4, 5)

    @cpu_and_gpu
    def test_reduce_size_raises(self, target_device_idx, xp):
        mat = xp.ones((5, 5))
        recmat = Recmat(mat, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            recmat.reduce_size(5)

    @cpu_and_gpu
    def test_nmodes(self, target_device_idx, xp):
        """Test that set_value raises AssertionError for wrong shape."""
        data = xp.ones((2, 3), dtype=np.float32)
        obj = Recmat(data.copy(), target_device_idx=target_device_idx)
        assert obj.nmodes == 2

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        """Test FITS header creation."""
        obj = Recmat(xp.zeros((2,2)))
        hdr = obj.get_fits_header()
        self.assertEqual(hdr['VERSION'], 1)

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        """Test saving and restoring Recmat."""
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        data = np.arange(10).reshape((5, 2)).astype(np.float32)
        obj = Recmat(data, norm_factor=2.0)

        obj.save(self.filename)
        self.assertTrue(os.path.exists(self.filename))

        restored = Recmat.restore(self.filename)
        np.testing.assert_array_equal(cpuArray(restored.recmat), data)
        self.assertEqual(restored.norm_factor, 2.0)


if __name__ == "__main__":
    unittest.main()
