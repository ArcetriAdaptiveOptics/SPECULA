
import specula
specula.init(0)  # Default target device

import numpy as np
import unittest

from specula import cp, cpuArray
from specula.base_processing_obj import BaseProcessingObj


class TestInit(unittest.TestCase):
   
    def test_xp_array_from_cpu_to_cpu(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        data_cpu = np.arange(3)
        assert id(data_cpu) == id(obj.xp_array(data_cpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_xp_array_from_gpu_to_cpu(self):
        obj = BaseProcessingObj(target_device_idx=-1)
        data_gpu = cp.arange(3)
        data_cpu = obj.xp_array(data_gpu)
        assert isinstance(data_cpu, np.ndarray)
        np.testing.assert_array_equal(data_cpu, cpuArray(data_gpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_xp_array_from_cpu_to_gpu(self):
        obj = BaseProcessingObj(target_device_idx=0)
        data_cpu = np.arange(3)
        data_gpu = obj.xp_array(data_cpu)
        assert isinstance(data_gpu, cp.ndarray)
        np.testing.assert_array_equal(data_cpu, cpuArray(data_gpu))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_xp_array_from_gpu_to_gpu(self):
        obj = BaseProcessingObj(target_device_idx=0)
        data_gpu = cp.arange(3)
        assert id(data_gpu) == id(obj.xp_array(data_gpu))
