import os
import unittest
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import np
from specula import cpuArray
from specula.base_value import BaseValue
from test.specula_testlib import cpu_and_gpu


class TestBaseValue(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_basevalue.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip_array(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        data = xp.arange(9).reshape((3,3))
        v = BaseValue(value=data, target_device_idx=target_device_idx)
        v.save(self.filename)
        v2 = BaseValue.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(v.value), cpuArray(v2.value))

        # Check FITS header for ndarray info
        hdr = fits.getheader(self.filename)
        self.assertEqual(hdr["NDARRAY"], 1)

    @cpu_and_gpu
    def test_save_restore_roundtrip_scalar(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        data = 3.1415
        v = BaseValue(value=data, target_device_idx=target_device_idx)
        v.save(self.filename)
        v2 = BaseValue.restore(self.filename)

        assert v2.value == 3.1415

        # Check FITS header for ndarray info
        hdr = fits.getheader(self.filename)
        self.assertEqual(hdr["NDARRAY"], 0)
        self.assertEqual(float(hdr["VALUE"]), 3.1415)


    @cpu_and_gpu
    def test_save_restore_roundtrip_scalar(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        data = 3.1415
        v = BaseValue(value=data, target_device_idx=target_device_idx)
        v.save(self.filename)
        v2 = BaseValue.restore(self.filename)

        assert v2.value == 3.1415

        # Check FITS header for ndarray info
        hdr = fits.getheader(self.filename)
        self.assertEqual(hdr["NDARRAY"], 0)
        self.assertEqual(float(hdr["VALUE"]), 3.1415)

    @cpu_and_gpu
    def test_save_restore_roundtrip_empty(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        v = BaseValue(target_device_idx=target_device_idx)
        v.save(self.filename)
        v2 = BaseValue.restore(self.filename)

        assert v2.value is None

        # Check FITS header for ndarray info
        hdr = fits.getheader(self.filename)
        self.assertEqual(hdr["NDARRAY"], 0)

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_init_with_default_values(self, target_device_idx, xp):
        """Test initializing BaseValue with no arguments"""
        bv = BaseValue(target_device_idx=target_device_idx)

        self.assertEqual(bv.description, "")
        self.assertIsNone(bv.value)

    @cpu_and_gpu
    def test_init_with_description_and_value(self, target_device_idx, xp):
        """Test initializing BaseValue with description and initial value"""
        value = xp.array([1, 2, 3])
        bv = BaseValue(description="test", value=value, target_device_idx=target_device_idx)

        self.assertEqual(bv.description, "test")
        xp.testing.assert_array_equal(bv.value, value)

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        """Test setting and getting a value"""
        bv = BaseValue(target_device_idx=target_device_idx)

        # Set value when value is None
        val1 = xp.array([1, 2, 3])
        bv.set_value(val1)
        xp.testing.assert_array_equal(bv.get_value(), val1)

        # Set value again when value already exists (in-place update)
        val2 = xp.array([4, 5, 6])
        bv.set_value(val2)
        xp.testing.assert_array_equal(bv.get_value(), val2)

    @cpu_and_gpu
    def test_array_for_display(self, target_device_idx, xp):
        """Test array_for_display returns the stored value"""
        val = xp.array([10, 20, 30])
        bv = BaseValue(value=val, target_device_idx=target_device_idx)
        xp.testing.assert_array_equal(bv.array_for_display(), val)

    @cpu_and_gpu
    def test_init_with_precision(self, target_device_idx, xp):
        """Test initializing BaseValue with precision argument"""
        val = xp.array([1, 2, 3])
        bv = BaseValue(value=val, target_device_idx=target_device_idx, precision=1)
        # Check that the dtype is as requested
        self.assertEqual(str(bv.value.dtype), 'float32')
        
        bv64 = BaseValue(value=val, target_device_idx=target_device_idx, precision=0)
        # Check that the dtype is as requested
        self.assertEqual(str(bv64.value.dtype), 'float64')

    @cpu_and_gpu
    def test_init_scalar_with_precision(self, target_device_idx, xp):
        """Test initializing BaseValue with scalar and precision argument"""
        bv = BaseValue(value=3.14, target_device_idx=target_device_idx, precision=1)
        self.assertEqual(str(type(bv.value)), "<class 'numpy.float32'>")
        self.assertAlmostEqual(bv.value, 3.14, places=6)
        bv64 = BaseValue(value=3.14, target_device_idx=target_device_idx, precision=0)
        self.assertEqual(str(type(bv64.value)), "<class 'numpy.float64'>")
