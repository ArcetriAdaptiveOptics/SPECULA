

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray
from specula.data_objects.time_history import TimeHistory
from test.specula_testlib import cpu_and_gpu

class TestTimeHistory(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_timehistory.fits')

    def _remove(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        
        self._remove()

        data = xp.arange(9).reshape((3,3))
        th = TimeHistory(data, target_device_idx=target_device_idx)
        
        th.save(self.filename)
        th2 = TimeHistory.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(th.time_history), cpuArray(th2.time_history))
        
    def tearDown(self):
        self._remove()

