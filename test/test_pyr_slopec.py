import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes
from specula.processing_objects.pyr_slopec import PyrSlopec

from test.specula_testlib import cpu_and_gpu

class TestSlopec(unittest.TestCase):

    @cpu_and_gpu
    def test_slopec(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        pupdata.framesize = (4,4)

        slopec = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        s1 = cpuArray(slopes.slopes)
        np.testing.assert_array_almost_equal(s1, np.array([-0.21276595, -0.29787233,  0. , -0.04255319]))

    @cpu_and_gpu
    def test_pyrslopec_slopesnull(self, target_device_idx, xp):
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25,  dtype=xp.uint16).reshape((5,5))
        pixels.generation_time = 1
        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1,3,6,8], [15,16,21,24]], dtype=int)
        pupdata.framesize = (4,4)
        sn = Slopes(slopes=xp.arange(4), target_device_idx=target_device_idx)

        slopec1 = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec2 = PyrSlopec(pupdata, sn=sn, norm_factor=None, target_device_idx=target_device_idx)
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

        slopec1 = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec2 = PyrSlopec(pupdata, sn=sn, norm_factor=None, target_device_idx=target_device_idx)
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

    @cpu_and_gpu
    def test_flux_outputs(self, target_device_idx, xp):
        """
        Test that verifies flux_per_subaperture, total_counts, and subap_counts outputs
        for pyramid WFS.
        """
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25, dtype=xp.uint16).reshape((5, 5))
        pixels.generation_time = 1

        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1, 3, 6, 8], [15, 16, 21, 24]], dtype=int)
        pupdata.framesize = (4, 4)

        slopec = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        # Get outputs
        flux_per_subap = slopec.outputs['out_flux_per_subaperture'].value
        total_counts = slopec.outputs['out_total_counts'].value
        subap_counts = slopec.outputs['out_subap_counts'].value

        # pupdata.ind_pup has shape (2, 4) meaning 2 subapertures
        # For each subaperture, we have pixels from 4 pupils (A, B, C, D)
        # Subaperture 0: A[1]+B[1]+C[1]+D[1], A[3]+B[3]+C[3]+D[3], etc.
        # The pixels array values are: 0,1,2,...,24
        # A (pupil 0): [1,3,6,8]
        # B (pupil 1): [1,3,6,8]
        # C (pupil 2): [15,16,21,24]
        # D (pupil 3): [15,16,21,24]
        # Sum for subap 0: 1+1+15+15 = 32
        # Sum for subap 1: 3+3+16+16 = 38
        # Wait, that's not matching...

        # Actually ind_pup[0] are pixels for subaperture 0: [1,3,6,8]
        # ind_pup[1] are pixels for subaperture 1: [15,16,21,24]
        # Each subaperture appears in all 4 pupils
        expected_flux = xp.array([
            1+3+6+8,      # subaperture 0
            15+16+21+24   # subaperture 1
        ], dtype=slopec.dtype)

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
