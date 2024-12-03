

import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.processing_objects.pyr_slopec import PyrSlopec


class TestSlopec(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_slopec_gpu(self):
        self._test_slopec(target_device_idx=0, xp=cp)

    def test_slopec_cpu(self):
        self._test_slopec(target_device_idx=-1, xp=np)
        
    def _test_slopec(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        slopec = PyrSlopec(pupdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']
        
        s1 = cpuArray(slopes.slopes)
        np.testing.assert_array_almost_equal(s1, np.array([-0.21276595, -0.29787233,  0. , -0.04255319]))

