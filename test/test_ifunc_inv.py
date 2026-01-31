import unittest
import numpy as np
import os
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.data_objects.ifunc_inv import IFuncInv
from test.specula_testlib import cpu_and_gpu


class TestIFuncInv(unittest.TestCase):
    def setUp(self):
        # Shape used in all tests
        self.shape = (4, 4)
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.inv_filename = os.path.join(self.datadir, 'ifunc_inv.fits')
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    def tearDown(self):
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_size_property(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        self.assertEqual(obj.size, self.shape)

    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(ifunc_inv))

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        new_val = xp.random.rand(*self.shape).astype(xp.float32)
        obj.set_value(new_val)

        np.testing.assert_array_equal(cpuArray(obj.get_value()), cpuArray(new_val))

    @cpu_and_gpu
    def test_set_value_wrong_shape(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        wrong_shape = xp.random.rand(2, 2).astype(xp.float32)
        with self.assertRaises(AssertionError):
            obj.set_value(wrong_shape)

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        hdr = obj.get_fits_header()
        self.assertIsInstance(hdr, fits.Header)
        self.assertEqual(hdr["VERSION"], 1)

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        obj.save(self.inv_filename, overwrite=True)
        restored = IFuncInv.restore(self.inv_filename, target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(restored.ifunc_inv), cpuArray(obj.ifunc_inv))
        np.testing.assert_array_equal(cpuArray(restored.mask_inf_func), cpuArray(obj.mask_inf_func))
        self.assertEqual(restored.size, obj.size)

    @cpu_and_gpu
    def test_restore_exten_offset(self, target_device_idx, xp):
        ifunc_inv = xp.random.rand(*self.shape).astype(xp.float32)
        mask = xp.random.choice([0, 1], size=self.shape).astype(xp.uint8)
        obj = IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

        obj.save(self.inv_filename, overwrite=True)
        restored = IFuncInv.restore(self.inv_filename, target_device_idx=target_device_idx, exten=1)

        np.testing.assert_array_equal(cpuArray(restored.ifunc_inv), cpuArray(obj.ifunc_inv))
        np.testing.assert_array_equal(cpuArray(restored.mask_inf_func), cpuArray(obj.mask_inf_func))

    @cpu_and_gpu
    def test_from_header_raises(self, target_device_idx, xp):
        with self.assertRaises(NotImplementedError):
            IFuncInv.from_header({})

