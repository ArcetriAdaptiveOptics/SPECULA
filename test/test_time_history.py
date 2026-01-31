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

    @cpu_and_gpu
    def test_init_with_description_and_value(self, target_device_idx, xp):
        """Test initializing BaseValue with description and initial value"""
        data = xp.array([1, 2, 3])
        th = TimeHistory(data, target_device_idx=target_device_idx)
        xp.testing.assert_array_equal(th.time_history, data)

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        """Test setting and getting a value"""
        data = xp.zeros(3)
        th = TimeHistory(data, target_device_idx=target_device_idx)

        # Set value when value is None
        val1 = xp.array([1, 2, 3])
        th.set_value(val1)
        xp.testing.assert_array_equal(th.get_value(), val1)

        # Set value again when value already exists (in-place update)
        val2 = xp.array([4, 5, 6])
        th.set_value(val2)
        xp.testing.assert_array_equal(th.get_value(), val2)

    @cpu_and_gpu
    def test_array_for_display(self, target_device_idx, xp):
        """Test array_for_display rasies"""
        data = xp.array([1, 2, 3])
        th = TimeHistory(data, target_device_idx=target_device_idx)
        xp.testing.assert_array_equal(th.array_for_display(), data)

    @cpu_and_gpu
    def test_from_header(self, target_device_idx, xp):
        """Test array_for_display rasies"""
        with self.assertRaises(NotImplementedError):
            _ = TimeHistory.from_header({})

