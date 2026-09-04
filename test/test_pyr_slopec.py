import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
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

    @cpu_and_gpu
    def test_pyrslopec_shlike_vectorial_normalization(self, target_device_idx, xp):
        """
        Test that verifies sh_like normalization is vectorial (per subaperture)
        and not global.
        
        Each subaperture should be normalized by its own flux, not the total flux.
        """
        pixels = Pixels(5, 5, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(25, dtype=xp.uint16).reshape((5, 5))
        pixels.generation_time = 1

        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[1, 3, 6, 8], [15, 16, 21, 24]], dtype=int)
        pupdata.framesize = (4, 4)

        slopec = PyrSlopec(pupdata, shlike=True, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        slopes = slopec.outputs['out_slopes']
        s1 = cpuArray(slopes.slopes)

        # Expected calculation:
        # Pixels: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        # A = pixels[[1, 15]] = [1, 15]
        # B = pixels[[3, 16]] = [3, 16]
        # C = pixels[[6, 21]] = [6, 21]
        # D = pixels[[8, 24]] = [8, 24]
        # flux_per_subap = A + B + C + D = [18, 76]
        # sx = (A + B - C - D) / flux_per_subap = [(1+3-6-8)/18, (15+16-21-24)/76]
        #    = [-10/18, -14/76]
        # sy = (B + C - A - D) / flux_per_subap = [(3+6-1-8)/18, (16+21-15-24)/76]
        #    = [0/18, -2/76]

        expected_sx = xp.array([-10.0/18, -14.0/76])
        expected_sy = xp.array([0.0/18, -2.0/76])
        expected_slopes = xp.concatenate([expected_sx, expected_sy])

        np.testing.assert_array_almost_equal(s1, cpuArray(expected_slopes), decimal=5)

    def _make_quadrant_pupdata_and_pixels(self, target_device_idx, xp):
        """
        Build a small but *realistic* 4-quadrant pyramid pupil layout (unlike
        the synthetic ind_pup used elsewhere in this file, whose indices do
        not actually fall inside their nominal quadrant, so single_mask()/
        local_display_map() would come out empty for it).

        Pupil 0 (A) occupies the top-right quadrant of a 4x4 frame at flat
        indices [2, 7]; pupils 1/2/3 (B/C/D) are the same local pattern
        translated to the top-left, bottom-left and bottom-right quadrants
        respectively.
        """
        pixels = Pixels(4, 4, target_device_idx=target_device_idx)
        pixels.pixels = xp.arange(16, dtype=xp.uint16).reshape((4, 4))
        pixels.generation_time = 1

        pupdata = PupData(target_device_idx=target_device_idx)
        pupdata.ind_pup = xp.array([[2, 0, 8, 10], [7, 5, 13, 15]], dtype=int)
        pupdata.framesize = (4, 4)

        return pixels, pupdata

    @cpu_and_gpu
    def test_out_pixels_subap_and_sum(self, target_device_idx, xp):
        """
        Test the new out_pixels_subap / out_pixels_subap_sum outputs of
        PyrSlopec: raw (pre-threshold) per-pupil intensities remapped to
        2d subaperture images, and their sum.
        """
        pixels, pupdata = self._make_quadrant_pupdata_and_pixels(target_device_idx, xp)

        slopec = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        subap_shape = pupdata.single_mask().shape
        self.assertEqual(subap_shape, (2, 2))

        out_pixels_subap = slopec.outputs['out_pixels_subap']
        out_pixels_subap_sum = slopec.outputs['out_pixels_subap_sum']

        self.assertIsInstance(out_pixels_subap, BaseValue)
        self.assertIsInstance(out_pixels_subap_sum, BaseValue)
        self.assertEqual(out_pixels_subap.value.shape, (4,) + subap_shape)
        self.assertEqual(out_pixels_subap_sum.value.shape, subap_shape)

        # A (pupil 0) = pixels[[2, 7]] = [2, 7]
        # B (pupil 1) = pixels[[0, 5]] = [0, 5]
        # C (pupil 2) = pixels[[8, 13]] = [8, 13]
        # D (pupil 3) = pixels[[10, 15]] = [10, 15]
        # remapped at local positions (0,0) and (1,1)
        expected_pixels_subap = np.array([
            [[2, 0], [0, 7]],
            [[0, 0], [0, 5]],
            [[8, 0], [0, 13]],
            [[10, 0], [0, 15]],
        ])
        expected_pixels_subap_sum = np.array([[20, 0], [0, 40]])

        np.testing.assert_array_almost_equal(cpuArray(out_pixels_subap.value), expected_pixels_subap)
        np.testing.assert_array_almost_equal(cpuArray(out_pixels_subap_sum.value), expected_pixels_subap_sum)

        self.assertEqual(out_pixels_subap.generation_time, 1)
        self.assertEqual(out_pixels_subap_sum.generation_time, 1)

    @cpu_and_gpu
    def test_out_slopes_map(self, target_device_idx, xp):
        """
        Test the new out_slopes_map output of Slopec (as inherited/populated
        by PyrSlopec): a 2d remap of the flat slopes vector using the
        single_mask/display_map machinery.
        """
        pixels, pupdata = self._make_quadrant_pupdata_and_pixels(target_device_idx, xp)

        slopec = PyrSlopec(pupdata, norm_factor=None, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()

        subap_shape = pupdata.single_mask().shape

        out_slopes_map = slopec.outputs['out_slopes_map']
        out_slopes = slopec.outputs['out_slopes']

        self.assertIsInstance(out_slopes_map, BaseValue)
        self.assertEqual(out_slopes_map.value.shape, (2,) + subap_shape)

        # Reconstructing from out_slopes_map should recover the same slope
        # values placed at the same subaperture positions as computed
        # directly via slopec.slopes.get2d().
        expected = cpuArray(slopec.slopes.get2d())
        np.testing.assert_array_almost_equal(cpuArray(out_slopes_map.value), expected)

        # The nonzero entries at local positions (0,0) and (1,1) must match
        # the flat sx/sy values in out_slopes (2 subapertures: sx then sy).
        slopes_flat = cpuArray(out_slopes.slopes)
        n_subap = pupdata.n_subap
        sx = slopes_flat[:n_subap]
        sy = slopes_flat[n_subap:]
        slopes_map = cpuArray(out_slopes_map.value)
        np.testing.assert_array_almost_equal(
            np.array([slopes_map[0, 0, 0], slopes_map[0, 1, 1]]), sx)
        np.testing.assert_array_almost_equal(
            np.array([slopes_map[1, 0, 0], slopes_map[1, 1, 1]]), sy)

        self.assertEqual(out_slopes_map.generation_time, 1)
