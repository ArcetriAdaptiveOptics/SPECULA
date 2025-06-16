
import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.ccd import CCD
from specula.data_objects.intensity import Intensity
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestCCD(unittest.TestCase):

    @cpu_and_gpu
    def test_ccd_wrong_dt(self, target_device_idx, xp):
        simul_params = SimulParams(time_step = 2)

        # A non-multiple of time_step raises ValueError
        with self.assertRaises(ValueError):
            ccd = CCD(simul_params, size=(2,2), dt=5, bandw=300,
                      target_device_idx=target_device_idx)

        # A multiple of time_step does not raise
        _ = CCD(simul_params, size=(2,2), dt=4, bandw=300,
                      target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_ccd_raises_on_missing_input(self, target_device_idx, xp):

        simul_params = SimulParams(time_step = 2)
        ccd = CCD(simul_params, size=(2,2), dt=2, bandw=300,
                       target_device_idx=target_device_idx)

        i = Intensity(dimx=2, dimy=2, target_device_idx=target_device_idx)
        
        # Raises because of missing input
        with self.assertRaises(ValueError):
            ccd.setup()

        ccd.inputs['in_i'].set(i)

        # Does not raise anymore
        ccd.setup()

if __name__ == '__main__':
    unittest.main()