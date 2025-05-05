

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray
from specula.data_objects.intmat import Intmat
from test.specula_testlib import cpu_and_gpu

class TestIntmat(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_im.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        
        im_data = xp.arange(9).reshape((3,3))
        im = Intmat(im_data, target_device_idx=target_device_idx)
        
        im.save(self.filename)
        im2 = Intmat.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(im._intmat), cpuArray(im2._intmat))
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

