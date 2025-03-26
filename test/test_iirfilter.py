

import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray

from specula.data_objects.iir_filter_data import IirFilterData

from test.specula_testlib import cpu_and_gpu

class TestIirFilterData(unittest.TestCase):

    @cpu_and_gpu
    def test_numerator_from_gain_and_ff(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)
        
        assert all(cpuArray(f.num[:, 0]) == 0)
        assert all(cpuArray(f.num[:, 1]) == 0.2)

    @cpu_and_gpu
    def test_denominator_from_gain_and_ff_num(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)
        
        assert all(cpuArray(f.den[:, 0]) == -1)
        assert all(cpuArray(f.den[:, 1]) == 1)

    @cpu_and_gpu
    def test_num_and_den_shape_from_gain_and_ff_num(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)
        
        assert f.num.shape == (nmodes, 2)
        assert f.den.shape == (nmodes, 2)
        
        

