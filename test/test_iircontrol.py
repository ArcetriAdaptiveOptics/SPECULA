

import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.iir_filter import IirFilter
from specula.processing_objects.integrator import Integrator
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestIirFilter(unittest.TestCase):
   
    # We just check that it goes through.
    @cpu_and_gpu
    def test_iir_filter_instantiation(self, target_device_idx, xp):
        iir_filter = IirFilterData(ordnum=(1,1), ordden=(1,1), num=xp.ones((2,2)), den=xp.ones((2,2)),
                                   target_device_idx=target_device_idx)
        simulParams = SimulParams(time_step=0.001)
        iir_control = IirFilter(simulParams, iir_filter)
        
    @cpu_and_gpu
    def test_integrator_instantiation(self, target_device_idx, xp):
        simulParams = SimulParams(time_step=0.001)
        integrator = Integrator(simulParams, int_gain=[0.5,0.4,0.3], ff=[0.99,0.95,0.90], n_modes= [2,3,4],
                                   target_device_idx=target_device_idx)
        # check that the iir_filter_data is set up correctly by comparing gain and [0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3]
        self.assertEqual(np.sum(np.abs(cpuArray(integrator.iir_filter_data.gain) - np.array([0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3]))),0)

