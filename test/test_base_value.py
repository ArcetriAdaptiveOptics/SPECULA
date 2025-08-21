

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray
from specula.base_value import BaseValue
from test.specula_testlib import cpu_and_gpu

class TestBaseValue(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_basevalue.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        data = xp.arange(9).reshape((3,3))
        v = BaseValue(value=data, target_device_idx=target_device_idx)
        v.save(self.filename)
        v2 = BaseValue.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(v.value), cpuArray(v2.value))
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

