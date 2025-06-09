'''
Test file for functions in specula/__init__.py
'''

import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cp
from specula import cpuArray


class TestInit(unittest.TestCase):
   
    def test_cpuArray_from_cpu_to_cpu_without_copy(self):
        data = np.arange(3)
        assert id(data) == id(cpuArray(data))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_cpuArray_from_gpu_to_cpu(self):
        data = cp.arange(3)
        data_cpu = cpuArray(data)
        assert isinstance(data_cpu, np.ndarray)
        np.testing.assert_array_equal(np.arange(3), data_cpu)

