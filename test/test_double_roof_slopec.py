
import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes
from specula.processing_objects.double_roof_slopec import DoubleRoofSlopec

from test.specula_testlib import cpu_and_gpu

class TestDrSlopec(unittest.TestCase):

    @cpu_and_gpu
    def test_slopec(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        pupdata.framesize = (4,4)

        slopec = DoubleRoofSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        s1 = cpuArray(slopes.slopes)
        np.testing.assert_array_almost_equal(s1, np.array([-0.042553, -0.021277, 0.042553, 0.06383]))

    @cpu_and_gpu
    def test_drslopec_slopesnull(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        pupdata.framesize = (4,4)
        sn = Slopes(slopes=xp.arange(4), target_device_idx=target_device_idx)

        slopec1 = DoubleRoofSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec2 = DoubleRoofSlopec(pupdata, sn=sn, norm_factor=None, target_device_idx=target_device_idx)
        slopec1.inputs['in_pixels'].set(pixels)
        slopec2.inputs['in_pixels'].set(pixels)
        slopec1.check_ready(1)
        slopec2.check_ready(1)
        slopec1.trigger()
        slopec2.trigger()
        slopec1.post_trigger()
        slopec2.post_trigger()
        slopes1 = slopec1.outputs['out_slopes']
        slopes2 = slopec2.outputs['out_slopes']

        np.testing.assert_array_almost_equal(cpuArray(slopes2.slopes),
                                             cpuArray(slopes1.slopes - sn.slopes))


    @cpu_and_gpu
    def test_pyrslopec_interleaved_slopesnull(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        pupdata.framesize = (4,4)
        sn = Slopes(slopes=xp.arange(4), interleave=True, target_device_idx=target_device_idx)

        slopec1 = DoubleRoofSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec2 = DoubleRoofSlopec(pupdata, sn=sn, norm_factor=None, target_device_idx=target_device_idx)
        slopec1.inputs['in_pixels'].set(pixels)
        slopec2.inputs['in_pixels'].set(pixels)
        slopec1.check_ready(1)
        slopec2.check_ready(1)
        slopec1.trigger()
        slopec2.trigger()
        slopec1.post_trigger()
        slopec2.post_trigger()
        slopes1 = slopec1.outputs['out_slopes']
        slopes2 = slopec2.outputs['out_slopes']

        np.testing.assert_array_almost_equal(cpuArray(slopes2.slopes),
                                             cpuArray(slopes1.slopes - xp.array([0,2,1,3])))
