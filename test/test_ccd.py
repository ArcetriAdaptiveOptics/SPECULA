
import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.ccd import CCD
from specula.data_objects.intensity import Intensity

from test.specula_testlib import cpu_and_gpu

class TestCCD(unittest.TestCase):

    @cpu_and_gpu
    def test_ccd_wrong_dt(self, target_device_idx, xp):
        dt = 3
        ccd = CCD(size=(2,2), dt=dt, bandw=300,
                       target_device_idx=target_device_idx)

        i = Intensity(dimx=2, dimy=2, target_device_idx=target_device_idx)
        
        ccd.inputs['in_i'].set(i)
        # TODO dt and seconds_to_t() must be revised
#        with self.assertRaises(ValueError):
        ccd.setup(5, loop_niters=1)
        
        # A multiple of dt does not raise
        ccd.setup(loop_dt=dt*2, loop_niters=1)

    @cpu_and_gpu
    def test_ccd_raises_on_missing_input(self, target_device_idx, xp):

        dt = 1
        ccd = CCD(size=(2,2), dt=dt, bandw=300,
                       target_device_idx=target_device_idx)

        i = Intensity(dimx=2, dimy=2, target_device_idx=target_device_idx)
        
        # Raises because of missing input
        with self.assertRaises(ValueError):
            ccd.setup(loop_dt=dt, loop_niters=1)

        ccd.inputs['in_i'].set(i)

        # Does not raise anymore
        ccd.setup(loop_dt=dt, loop_niters=1)

if __name__ == '__main__':
    unittest.main()