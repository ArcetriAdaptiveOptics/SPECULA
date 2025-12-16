
import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cp
from specula.base_value import BaseValue


class TestBaseDataObj(unittest.TestCase):

    def test_copy_from_cpu_to_cpu(self):
        '''
        Test that copyTo() from CPU to CPU works fine
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = a.copyTo(target_device_idx=-1)

        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1
        np.testing.assert_array_equal(b.value, [0, 1])

    def test_update_from_cpu_to_cpu(self):
        '''
        Test that transferDataTo() from CPU to CPU works fine
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = BaseValue(value=np.zeros(2), target_device_idx=-1)
        a.transferDataTo(b)

        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1
        np.testing.assert_array_equal(b.value, [0, 1])

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_copy_from_cpu_to_gpu(self):
        '''
        Test that copyTo() with target_device_idx >= 0
        allocates a new object on the GPU with the correct contents
        '''
        a = BaseValue(value=np.arange(2), target_device_idx=-1)
        b = a.copyTo(target_device_idx=0)

        assert type(b.value) == cp.ndarray
        assert b.target_device_idx == 0
        np.testing.assert_array_equal(b.value.get(), [0, 1])

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_update_from_cpu_to_gpu(self):
        '''
        Test that transferDataTo() with a GPU object
        correctly updates the contents
        '''
        a1 = BaseValue(value=np.arange(2), target_device_idx=-1)
        a2 = BaseValue(value=np.arange(2)+2, target_device_idx=-1)
        b = a1.copyTo(target_device_idx=0)
        _ = a2.transferDataTo(b)

        assert type(b.value) == cp.ndarray
        np.testing.assert_array_equal(b.value.get(), [2, 3])
        assert b.target_device_idx == 0

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_copy_from_gpu_to_cpu(self):
        '''
        Test that copyTo() with target_device_idx == -1
        allocates a new object on the CPU with the correct contents
        '''
        a = BaseValue(value=cp.arange(2), target_device_idx=0)
        b = a.copyTo(target_device_idx=-1)

        np.testing.assert_array_equal(b.value, [0, 1])
        assert type(b.value) == np.ndarray
        assert b.target_device_idx == -1

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_update_from_gpu_to_cpu(self):
        '''
        Test that transferDataTo() with a CPU object
        correctly updates the contents
        '''
        a1 = BaseValue(value=cp.arange(2), target_device_idx=0)
        a2 = BaseValue(value=cp.arange(2)+2, target_device_idx=0)
        b = a1.copyTo(target_device_idx=-1)
        _ = a2.transferDataTo(b)

        assert type(b.value) == np.ndarray
        np.testing.assert_array_equal(b.value, [2, 3])
        assert b.target_device_idx == -1

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_copy_from_gpu_to_gpu(self):
        '''
        Test that copyTo() from a GPU device to another
        allocates a new object on the target GPU with the correct contents
        '''        
        a = BaseValue(value=cp.arange(2), target_device_idx=0)
        b = a.copyTo(target_device_idx=1)

        np.testing.assert_array_equal(b.value.get(), [0, 1])
        assert type(b.value) == cp.ndarray
        assert b.target_device_idx == 1

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'at least 2 GPUs are needed')
    def test_update_from_gpu_to_gpu(self):
        '''
        Test that transferDataTo() from a GPU device to another
        correctly updates the contents
        '''
        a1 = BaseValue(value=cp.arange(2), target_device_idx=0)
        a2 = BaseValue(value=cp.arange(2)+2, target_device_idx=0)
        b = a1.copyTo(target_device_idx=1)
        _ = a2.transferDataTo(b)

        assert type(b.value) == cp.ndarray
        np.testing.assert_array_equal(b.value.get(), [2, 3])
        assert b.target_device_idx == 1
