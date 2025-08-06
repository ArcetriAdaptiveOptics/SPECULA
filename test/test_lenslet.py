

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula.data_objects.lenslet import Lenslet
from test.specula_testlib import cpu_and_gpu

class TestLenslet(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_lenslet.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        lenslet = Lenslet(n_lenses=4, target_device_idx=target_device_idx)
        
        lenslet.save(self.filename)
        lenslet2 = Lenslet.restore(self.filename)

        assert lenslet.n_lenses == lenslet2.n_lenses
        assert lenslet2.n_lenses == 4
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

