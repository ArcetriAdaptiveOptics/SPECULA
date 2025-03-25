

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField

from test.specula_testlib import cpu_and_gpu

class TestElectricField(unittest.TestCase):
   
    @cpu_and_gpu
    def test_reset_does_not_reallocate(self, target_device_idx, xp):
        
        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)
        
        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)
        
        ef.reset()
        
        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)
        
        assert id_A_before == id_A_after
        assert id_p_before == id_p_after

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):
        
        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)
        
        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)
        
        ef.set_value([xp.ones(100).reshape(10,10), xp.zeros(100).reshape(10,10)])
        
        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)
        
        assert id_A_before == id_A_after
        assert id_p_before == id_p_after