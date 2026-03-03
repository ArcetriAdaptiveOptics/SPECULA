import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.processing_objects.zwfs_slopec import ZwfsSlopec

from test.specula_testlib import cpu_and_gpu

class TestSlopec(unittest.TestCase):

    @cpu_and_gpu
    def test_slopec(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1

        slopec = ZwfsSlopec(diameter=3, ccd_size=(5,5), target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        pix_in_pupil = cpuArray(pixels.pixels[1:4,1:4]).flatten()
        want = pix_in_pupil / (xp.mean(pix_in_pupil))

        got = cpuArray(slopes.slopes)
        np.testing.assert_array_almost_equal(got, want)

    @cpu_and_gpu
    def test_zernslopec_slopesnull(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        sn = Slopes(slopes=np.arange(9)/9, target_device_idx=target_device_idx)

        slopec1 = ZwfsSlopec(diameter=3, ccd_size=(5,5), target_device_idx=target_device_idx)
        slopec2 = ZwfsSlopec(diameter=3, ccd_size=(5,5), sn=sn, target_device_idx=target_device_idx)
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

        np.testing.assert_equal(cpuArray(sn.xslopes),np.arange(9)[:4]/9) # ensure sn can handle an odd number of elements
        np.testing.assert_equal(cpuArray(sn.yslopes),np.arange(9)[4:]/9) # ensure sn can handle an odd number of elements
        np.testing.assert_array_almost_equal(cpuArray(slopes2.slopes),
                                             cpuArray(slopes1.slopes - sn.slopes))

    @cpu_and_gpu
    def test_flux_outputs(self, target_device_idx, xp):
        """
        Test that verifies flux_per_subaperture, total_counts, and subap_counts outputs
        for pyramid WFS.
        """
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25, dtype=xp.uint16).reshape((5, 5))
        pixels.generation_time = 1

        slopec = ZwfsSlopec(diameter=3, ccd_size=(5,5), target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        # Get outputs
        flux_per_subap = slopec.outputs['out_flux_per_subaperture'].value
        total_counts = slopec.outputs['out_total_counts'].value
        subap_counts = slopec.outputs['out_subap_counts'].value

        expected_flux = xp.array((pixels.pixels[1:4,1:4]).flatten()
        , dtype=slopec.dtype)

        # Verify flux_per_subaperture
        np.testing.assert_array_almost_equal(cpuArray(flux_per_subap),
                                             cpuArray(expected_flux), decimal=5)

        # Verify total_counts
        expected_total = xp.sum(expected_flux)
        np.testing.assert_almost_equal(cpuArray(total_counts[0]),
                                       cpuArray(expected_total), decimal=5)

        # Verify subap_counts
        expected_mean = xp.mean(expected_flux)
        np.testing.assert_almost_equal(cpuArray(subap_counts[0]),
                                       cpuArray(expected_mean), decimal=5)

        # Verify generation times are set
        self.assertEqual(slopec.outputs['out_flux_per_subaperture'].generation_time, 1)
        self.assertEqual(slopec.outputs['out_total_counts'].generation_time, 1)
        self.assertEqual(slopec.outputs['out_subap_counts'].generation_time, 1)
