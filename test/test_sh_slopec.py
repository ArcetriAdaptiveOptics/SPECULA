import specula
specula.init(0)  # Default target device

import unittest
import os
import glob

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from test.specula_testlib import cpu_and_gpu

class TestShSlopec(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        test_dir = os.path.dirname(__file__)
        for fpath in glob.glob(os.path.join(test_dir, 'ConvolutionKernel*.fits')):
            if os.path.isfile(fpath):
                os.remove(fpath)

    def get_sh(self, target_device_idx, xp, with_laser_launch=False):
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        # 2x2 subapertures
        subap_on_diameter = 2
        # lambda is 500 nm and lambda/D is 0.206 arcsec so 0.1 means 2 pixels per lambda/D
        wavelengthInNm = 500
        pxscale_arcsec = 0.1
        # big subaperture to avoid edge effects
        subap_npx = 12
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        # ------------------------------------------------------------------------------
        # Set up inputs for ShSlopec
        idxs = {}
        map = {}
        mask_subap = np.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))

        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask_subap *= 0
                mask_subap[i*subap_npx:(i+1)*subap_npx,j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = np.where(mask_subap == 1)
                map[count] = j * subap_on_diameter + i
                count += 1

        v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = np.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = np.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map[k]

        # "simple" SH
        if not with_laser_launch:
            sh = SH(wavelengthInNm=wavelengthInNm,
                    subap_wanted_fov=subap_npx * pxscale_arcsec,
                    sensor_pxscale=pxscale_arcsec,
                    subap_on_diameter=subap_on_diameter,
                    subap_npx=subap_npx,
                    target_device_idx=target_device_idx)

        # SH with laser launch
        else:
            laser_launch_tel = LaserLaunchTelescope(spot_size=pxscale_arcsec,
                                target_device_idx=target_device_idx)

            sh = SH(wavelengthInNm=wavelengthInNm,
                    subap_wanted_fov=subap_npx * pxscale_arcsec,
                    sensor_pxscale=pxscale_arcsec,
                    subap_on_diameter=subap_on_diameter,
                    subap_npx=subap_npx,
                    laser_launch_tel=laser_launch_tel,
                    target_device_idx=target_device_idx)

        flat_ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        flat_ef.generation_time = t

        subapdata = SubapData(idxs=v, display_map = m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)

        return sh, v, m, flat_ef, subapdata

    @cpu_and_gpu
    def test_pixelscale_and_slopes(self, target_device_idx, xp):
        """
        Test that verifies both pixel scale and slope computation for SH.
        A tilt that shifts the spot by 1 pixel should produce a slope of 1/(sh.subap_npx/2).
        """
        # Flat wavefront
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        t = 1
        pxscale_arcsec = 0.1
        subap_npx = 12

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2, tilt_value / 2, pixel_pupil)
        
        # Tilted wavefront
        flat_ef.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        flat_ef.generation_time = t+1

        sh.check_ready(t+1)
        sh.trigger()
        sh.post_trigger()
        tilted = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*tilted.shape, target_device_idx=target_device_idx)
        pixels.pixels = tilted
        pixels.generation_time = t+1

        # Create the slope computer object
        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t+1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        # Expected value: 1/(subap_npx/2)
        expected_slope = 1.0 / (subap_npx / 2)

        # All X slopes (all slopes are valid) should be close to the expected value
        np.testing.assert_allclose(cpuArray(slopes.xslopes), expected_slope, rtol=1e-2, atol=1e-2)

    @cpu_and_gpu
    def test_weight_int_pixel_dt(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt.
        """
        
        weight_int_pixel_dt = 3.0
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity =sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        slopec = ShSlopec(subapdata, weight_int_pixel_dt=weight_int_pixel_dt, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Simulate 3 frames with known values
        for i in range(4):
            current_time = t*(i+1)
            if i == 2:
                # shift the pixels to simulate a change
                pixels.pixels = xp.roll(pixels.pixels, shift=1, axis=0)
            pixels.generation_time = current_time
            slopec.check_ready(current_time)
            slopec.trigger()
            slopec.post_trigger()

        # After two steps, the weight map should be the average of the two frames
        # normalized to the maximum intensity
        last_weights = slopec.int_pixels_weight

        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d_flat = last_weights_2d.flatten()
        last_weights_2d_flat[slopec.subap_idx.flatten()] = last_weights.T.flatten()
        last_weights_2d = last_weights_2d_flat.reshape(last_weights_2d.shape)

        # the expected weights are the average of frames
        expected_weights = 2 * intensity + xp.roll(intensity, shift=1, axis=0)
        expected_weights = expected_weights / expected_weights.max()

        np.testing.assert_allclose(cpuArray(last_weights_2d), cpuArray(expected_weights), atol=1e-3)

        # Then compares slopec.int_pixels.pixels and slopec.pixels.pixels:
        # they must be equal because the accumulation was resetted
        expected_int_pixels = pixels.pixels.astype(slopec.dtype)
        np.testing.assert_allclose(cpuArray(slopec.int_pixels.pixels), cpuArray(expected_int_pixels), atol=1e-3)

    @cpu_and_gpu
    def test_weight_int_pixel_dt_window(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt and window_int_pixel.
        """
        weight_int_pixel_dt = 2.0
        window_int_threshold = 1.0

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity/intensity.max()*10
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        slopec = ShSlopec(subapdata, weight_int_pixel_dt=weight_int_pixel_dt, window_int_threshold=window_int_threshold,
                          window_int_pixel=True, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Simulate 2 frames with known values
        for i in range(2):
            current_time = t*(i+1)
            pixels.generation_time = current_time
            slopec.check_ready(current_time)
            slopec.trigger()
            slopec.post_trigger()

        # After two steps, the weight map should be 4 square of 4x4 pixels
        # and value of 1.0 in the square and 0 outside
        last_weights = slopec.int_pixels_weight

        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d_flat = last_weights_2d.flatten()
        last_weights_2d_flat[slopec.subap_idx.flatten()] = last_weights.T.flatten()
        last_weights_2d = last_weights_2d_flat.reshape(last_weights_2d.shape)

        expected_weights = xp.zeros_like(last_weights_2d)
        expected_weights[4:8,   4:8] = 1.0
        expected_weights[16:20, 16:20] = 1.0
        expected_weights[4:8,   16:20] = 1.0
        expected_weights[16:20, 4:8] = 1.0

        np.testing.assert_equal(cpuArray(last_weights_2d), cpuArray(expected_weights), err_msg="Weight map does not match expected values.")

    @cpu_and_gpu
    def test_xy_weights_dtype_follows_precision(self, target_device_idx, xp):
        """
        Test that mask_weighted/xweights/yweights/xcweights/ycweights follow the
        object's own dtype, in both the default (weighted CoG) mode and quadcell mode.
        """
        # Minimal subap data: 1 subaperture, 4x4 pixels, no dependency on a real SH.
        np_sub = 4
        idxs = xp.arange(np_sub * np_sub).reshape(1, np_sub * np_sub)
        display_map = xp.array([0])
        subapdata = SubapData(idxs=idxs, display_map=display_map, nx=1, ny=1,
                              target_device_idx=target_device_idx)

        slopec32 = ShSlopec(subapdata, weightedPixRad=1.0, precision=1,
                            target_device_idx=target_device_idx)
        for name in ('mask_weighted', 'xweights', 'yweights', 'xcweights', 'ycweights'):
            self.assertEqual(getattr(slopec32, name).dtype, xp.float32,
                             f"{name} dtype does not match precision=1")

        # quadcell mode uses a different branch in computeXYweights
        slopec32.quadcell_mode = True
        slopec32.set_xy_weights()
        for name in ('mask_weighted', 'xweights', 'yweights', 'xcweights', 'ycweights'):
            self.assertEqual(getattr(slopec32, name).dtype, xp.float32,
                             f"{name} dtype does not match precision=1 (quadcell mode)")

        # precision=0 (double) must stay float64
        slopec64 = ShSlopec(subapdata, weightedPixRad=1.0, precision=0,
                            target_device_idx=target_device_idx)
        for name in ('mask_weighted', 'xweights', 'yweights', 'xcweights', 'ycweights'):
            self.assertEqual(getattr(slopec64, name).dtype, xp.float64,
                             f"{name} dtype does not match precision=0")

    @cpu_and_gpu
    def test_vec_wei_pix_rad_t_uses_last_valid_time(self, target_device_idx, xp):
        """
        Test that vecWeiPixRadT selects the last valid row based on time.
        """
        t_sh = int(1e9)
        t_slopec = int(2e9)

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t_sh)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t_slopec

        vec_wei_pix_rad_t = xp.asarray([
            [0.5, 0.2],
            [1.5, 1.0],
            [3.0, 3.0],
        ])

        slopec = ShSlopec(subapdata, vecWeiPixRadT=vec_wei_pix_rad_t,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t_slopec)
        slopec.trigger()
        slopec.post_trigger()

        # At t=2.0 s, rows at 0.2 s and 1.0 s are valid, so selected radius must be 1.5.
        self.assertAlmostEqual(float(slopec.weighted_pix_rad), 1.5)

        expected = ShSlopec(subapdata, weightedPixRad=1.5, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(slopec.mask_weighted),
                                   cpuArray(expected.mask_weighted), atol=1e-6)


    @cpu_and_gpu
    def test_shslopec_slopesnull(self, target_device_idx, xp):
        '''
        Test that a SH Slopec correctly subtracts slope nulls (non-interleaved)
        '''
        # Flat wavefront
        # pupil is 1m
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        sn = Slopes(slopes=xp.arange(len(m)*2), interleave=False, target_device_idx=target_device_idx)

        slopec1 = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec2 = ShSlopec(subapdata, sn=sn, target_device_idx=target_device_idx)

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
    def test_shslopec_interleaved_slopesnull(self, target_device_idx, xp):
        '''
        Test that a SH Slopec correctly subtracts slope nulls (interleaved)
        '''

        # Flat wavefront
        # pupil is 1m
        t=1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters

        sn = Slopes(slopes=xp.arange(len(m)*2), interleave=True, target_device_idx=target_device_idx)

        slopec1 = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec2 = ShSlopec(subapdata, sn=sn, target_device_idx=target_device_idx)

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

        np.testing.assert_array_almost_equal(cpuArray(slopes2.xslopes),
                                             cpuArray(slopes1.xslopes - sn.xslopes))

        np.testing.assert_array_almost_equal(cpuArray(slopes2.yslopes),
                                             cpuArray(slopes1.yslopes - sn.yslopes))

    @cpu_and_gpu
    def test_flux_outputs(self, target_device_idx, xp):
        """
        Test that verifies flux_per_subaperture, total_counts, and subap_counts outputs.
        """
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        # Get outputs
        flux_per_subap = slopec.outputs['out_flux_per_subaperture'].value
        total_counts = slopec.outputs['out_total_counts'].value
        subap_counts = slopec.outputs['out_subap_counts'].value

        # Verify flux_per_subaperture has correct shape
        self.assertEqual(flux_per_subap.shape[0], len(m))

        # Verify total_counts is sum of all flux
        expected_total = xp.sum(flux_per_subap)
        np.testing.assert_almost_equal(cpuArray(total_counts[0]),
                                       cpuArray(expected_total), decimal=5)

        # Verify subap_counts is mean of flux_per_subaperture
        expected_mean = xp.mean(flux_per_subap)
        np.testing.assert_almost_equal(cpuArray(subap_counts[0]),
                                       cpuArray(expected_mean), decimal=5)

        # Verify all values are positive
        self.assertTrue(xp.all(flux_per_subap >= 0))
        self.assertTrue(total_counts[0] >= 0)
        self.assertTrue(subap_counts[0] >= 0)

        # Verify generation times are set
        self.assertEqual(slopec.outputs['out_flux_per_subaperture'].generation_time, t)
        self.assertEqual(slopec.outputs['out_total_counts'].generation_time, t)
        self.assertEqual(slopec.outputs['out_subap_counts'].generation_time, t)

    def _build_multi_subap_pixels(self, blocks, target_device_idx, xp):
        """
        Build a Pixels/SubapData pair from a list of 2D numpy blocks, one per
        subaperture, laid out side by side in a single row (no gaps, no overlap).
        Returns (subapdata, pixels).
        """
        np_sub = blocks[0].shape[0]
        n_subaps = len(blocks)
        dimy = np_sub
        dimx = np_sub * n_subaps

        full = np.zeros((dimy, dimx))
        idxs_list = []
        display_map = []
        for i, block in enumerate(blocks):
            full[:, i * np_sub:(i + 1) * np_sub] = block
            mask = np.zeros((dimy, dimx))
            mask[:, i * np_sub:(i + 1) * np_sub] = 1
            idx = np.where(mask == 1)
            idxs_list.append(np.ravel_multi_index(idx, full.shape))
            display_map.append(i)

        idxs_arr = np.array(idxs_list)
        display_map = np.array(display_map)

        subapdata = SubapData(idxs=idxs_arr, display_map=display_map, nx=n_subaps, ny=1,
                              target_device_idx=target_device_idx)

        pixels = Pixels(dimx, dimy, target_device_idx=target_device_idx)
        pixels.pixels = xp.array(full)
        pixels.generation_time = 1

        return subapdata, pixels

    @cpu_and_gpu
    def test_thr_ratio_value_per_subaperture(self, target_device_idx, xp):
        """
        Test that thr_ratio_value applies an independent threshold to each
        subaperture (thr_ratio_value * subap.max()), as opposed to thr_value
        which applies a single absolute threshold to all subapertures.
        """
        # Three subapertures with different brightness, background pedestal
        # and spot shape/location, so that each one gets a different absolute
        # threshold from the same thr_ratio_value.
        block0 = np.array([[10., 10., 10., 10.],
                            [10., 100., 80., 10.],
                            [10., 90., 70., 10.],
                            [10., 10., 10., 10.]])
        block1 = np.array([[20., 20., 20., 20.],
                            [20., 20., 20., 20.],
                            [20., 20., 50., 45.],
                            [20., 20., 55., 40.]])
        block2 = np.array([[5., 5., 5., 5.],
                            [5., 5., 5., 5.],
                            [5., 5., 5., 200.],
                            [5., 5., 150., 120.]])
        blocks = [block0, block1, block2]

        subapdata, pixels = self._build_multi_subap_pixels(blocks, target_device_idx, xp)

        thr_ratio_value = 0.3
        slopec = ShSlopec(subapdata, thr_ratio_value=thr_ratio_value,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        # Weights are pure geometry (unrelated to thresholding), reuse them
        # from the object itself instead of re-deriving make_xy/exp_weight.
        xweights = cpuArray(slopec.xweights)
        yweights = cpuArray(slopec.yweights)
        mask_weighted = cpuArray(slopec.mask_weighted)

        expected_sx = np.zeros(len(blocks))
        expected_sy = np.zeros(len(blocks))
        subap_tot_list = np.zeros(len(blocks))
        for i, block in enumerate(blocks):
            thr = thr_ratio_value * block.max()
            clipped = np.clip(block - thr, 0, None)
            subap_tot = np.sum(clipped * mask_weighted)
            subap_tot_list[i] = subap_tot
            expected_sx[i] = np.sum(clipped * xweights) / subap_tot
            expected_sy[i] = np.sum(clipped * yweights) / subap_tot

        # Sanity check: none of the subapertures falls under the low-flux
        # clamp in trigger_code (factor forced to 0), otherwise the reference
        # formula above would not apply.
        mean_tot = subap_tot_list.mean()
        self.assertTrue(np.all(subap_tot_list > mean_tot * 1e-3))

        np.testing.assert_allclose(cpuArray(slopes.xslopes), expected_sx, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cpuArray(slopes.yslopes), expected_sy, rtol=1e-5, atol=1e-6)

        # A single absolute threshold (thr_value) equal to subaperture 0's own
        # per-subap threshold must give a different result overall: subap 0
        # matches (same absolute threshold), but subaps 1 and 2 do not, since
        # their own max (hence their ratio-based threshold) differs.
        thr_abs = thr_ratio_value * block0.max()
        slopec_abs = ShSlopec(subapdata, thr_value=thr_abs, target_device_idx=target_device_idx)
        slopec_abs.inputs['in_pixels'].set(pixels)
        slopec_abs.check_ready(1)
        slopec_abs.trigger()
        slopec_abs.post_trigger()
        slopes_abs = slopec_abs.outputs['out_slopes']

        np.testing.assert_allclose(cpuArray(slopes_abs.xslopes)[0], expected_sx[0], rtol=1e-5, atol=1e-6)
        self.assertFalse(np.allclose(cpuArray(slopes_abs.xslopes), expected_sx))

    @cpu_and_gpu
    def test_thr_value_and_thr_ratio_value_raises(self, target_device_idx, xp):
        """
        Test that setting both thr_value > 0 and thr_ratio_value > 0 raises
        ValueError when the object is triggered.
        """
        np_sub = 4
        block = np.ones((np_sub, np_sub)) * 10.0
        subapdata, pixels = self._build_multi_subap_pixels([block], target_device_idx, xp)

        slopec = ShSlopec(subapdata, thr_value=1.0, thr_ratio_value=0.3,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(1)
        with self.assertRaises(ValueError):
            slopec.trigger()

    @cpu_and_gpu
    def test_slopec_interleave(self, target_device_idx, xp):
        """
        Test that verifies the interleave option in Slopec.
        """
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Non-interleaved
        slopec_non = ShSlopec(subapdata, interleave=False, target_device_idx=target_device_idx)
        slopec_non.inputs['in_pixels'].set(pixels)
        slopec_non.check_ready(t)
        slopec_non.trigger()
        slopec_non.post_trigger()

        # Interleaved
        slopec_int = ShSlopec(subapdata, interleave=True, target_device_idx=target_device_idx)
        slopec_int.inputs['in_pixels'].set(pixels)
        slopec_int.check_ready(t)
        slopec_int.trigger()
        slopec_int.post_trigger()

        # Verify that both compute the same xslopes and yslopes values
        np.testing.assert_array_almost_equal(cpuArray(slopec_non.slopes.xslopes),
                                             cpuArray(slopec_int.slopes.xslopes))
        np.testing.assert_array_almost_equal(cpuArray(slopec_non.slopes.yslopes),
                                             cpuArray(slopec_int.slopes.yslopes))

        # Verify that the internal layout is different (interleaved vs non-interleaved)
        self.assertFalse(slopec_non.slopes.interleave)
        self.assertTrue(slopec_int.slopes.interleave)

        # Verify indices are different
        np.testing.assert_array_equal(cpuArray(slopec_non.slopes.indx()),
                                      cpuArray(xp.arange(0, len(m))))
        np.testing.assert_array_equal(cpuArray(slopec_int.slopes.indx()),
                                      cpuArray(xp.arange(0, len(m)*2, 2)))

    def _run_slopec_steps(self, subapdata, frames, target_device_idx, use_setup, cls=ShSlopec,
                          **kwargs):
        """Run a ShSlopec (or *cls*) over *frames* and return the outputs of each step"""
        pixels = Pixels(frames[0].shape[1], frames[0].shape[0], target_device_idx=target_device_idx)
        slopec = cls(subapdata, target_device_idx=target_device_idx, **kwargs)
        slopec.inputs['in_pixels'].set(pixels)
        if use_setup:
            slopec.setup()
        results = []
        for i, frame in enumerate(frames):
            t = slopec.seconds_to_t(i + 1)
            # Written in place: the CUDA graph always reads the same input array
            pixels.pixels[:] = pixels.xp.asarray(frame)
            pixels.generation_time = t
            slopec.check_ready(t)
            slopec.trigger()
            slopec.post_trigger()
            results.append([cpuArray(x).copy() for x in (
                slopec.slopes.slopes, slopec.flux_per_subaperture_vector.value,
                slopec.total_counts.value, slopec.subap_counts.value, slopec.slopes_map.value)])
        return slopec, results

    def test_cuda_graph_matches_eager_and_cpu(self):
        """
        The CUDA graph built by setup() must give the same results as
        the eager GPU execution, and the GPU results must match the CPU ones,
        with changing input pixels and all the slope corrections.
        """
        rng = np.random.default_rng(0)
        np_sub = 6
        n_blocks = 5
        blocks = [np.zeros((np_sub, np_sub)) for _ in range(n_blocks)]
        subapdata_cpu, pixels = self._build_multi_subap_pixels(blocks, -1, np)
        n_subaps = subapdata_cpu.n_subaps
        frames = [rng.poisson(20, pixels.pixels.shape).astype(np.float32) for _ in range(6)]
        # One dark subaperture, whose slopes must be set to zero
        for frame in frames:
            frame[:, :np_sub] = 0

        filtmat = [rng.normal(size=(2 * n_subaps, 2)), rng.normal(size=(2 * n_subaps, 2)) * 0.1]
        configs = [
            dict(),
            dict(thr_value=15, interleave=True),
            dict(thr_ratio_value=0.5, weightedPixRad=2),
            dict(weight_int_pixel_dt=2, window_int_pixel=True, window_int_threshold=10, filtmat=filtmat),
            dict(weight_int_pixel_dt=3, filtmat=filtmat, interleave=True),
            dict(thr_value=5, filtmat=filtmat, precision=0),
        ]
        targets = [-1] if specula.cp is None else [-1, 0]
        for kwargs in configs:
            for interleave_sn in [False, True]:
                results = {}
                for target_device_idx in targets:
                    subapdata = SubapData(idxs=subapdata_cpu.idxs, display_map=subapdata_cpu.display_map,
                                          nx=n_blocks, ny=1, target_device_idx=target_device_idx)
                    sn = Slopes(slopes=rng.normal(size=2 * n_subaps) * 0.01 if target_device_idx < 0
                                else results['sn'], interleave=interleave_sn,
                                target_device_idx=target_device_idx)
                    results['sn'] = cpuArray(sn.slopes)
                    for use_setup in ([False] if target_device_idx < 0 else [False, True]):
                        slopec, res = self._run_slopec_steps(subapdata, frames, target_device_idx,
                                                             use_setup, sn=sn, **kwargs)
                        results[(target_device_idx, use_setup)] = res
                        if use_setup:
                            self.assertIsNotNone(slopec.cuda_graph)

                msg = f'{kwargs=} {interleave_sn=}'
                for step_cpu, step_eager, step_graph in zip(results[(-1, False)],
                                                            results.get((0, False), results[(-1, False)]),
                                                            results.get((0, True), results[(-1, False)])):
                    for cpu_value, eager_value, graph_value in zip(step_cpu, step_eager, step_graph):
                        # Not bit-identical: without a graph, matrix products use cuBLAS
                        np.testing.assert_allclose(eager_value, graph_value, rtol=1e-6, atol=1e-7,
                                                   err_msg=msg)
                        np.testing.assert_allclose(cpu_value, eager_value, rtol=1e-5, atol=1e-6, err_msg=msg)
                slopes = results[(-1, False)][-1][0]
                self.assertFalse(np.any(np.isnan(slopes)), msg)
                self.assertTrue(np.any(slopes != 0), msg)
                # Dark subaperture
                self.assertEqual(results[(-1, False)][-1][1][0], 0)

    def _frames_and_subapdata(self, target_device_idx, xp, nframes=4):
        np_sub = 6
        blocks = [np.zeros((np_sub, np_sub)) for _ in range(4)]
        subapdata, pixels = self._build_multi_subap_pixels(blocks, target_device_idx, xp)
        rng = np.random.default_rng(1)
        frames = [rng.poisson(20, pixels.pixels.shape).astype(np.float32) for _ in range(nframes)]
        return subapdata, pixels, frames

    @cpu_and_gpu
    def test_derived_class_slope_null_applied_once(self, target_device_idx, xp):
        """
        A derived class extending compute_slopes() gets the slope null subtracted
        exactly once, and uses a CUDA graph only if it opts in.
        """
        class DerivedShSlopec(ShSlopec):
            def compute_slopes(self):
                super().compute_slopes()

        class GraphShSlopec(DerivedShSlopec):
            def setup(self):
                super().setup()
                self.build_stream(capture=False)

        subapdata, _, frames = self._frames_and_subapdata(target_device_idx, xp)
        sn = Slopes(slopes=np.random.default_rng(2).normal(size=2 * subapdata.n_subaps),
                    target_device_idx=target_device_idx)
        _, ref_no_sn = self._run_slopec_steps(subapdata, frames, target_device_idx, True)
        for cls in [ShSlopec, DerivedShSlopec, GraphShSlopec]:
            slopec, res = self._run_slopec_steps(subapdata, frames, target_device_idx, True,
                                                 sn=sn, cls=cls)
            uses_graph = target_device_idx >= 0 and cls is not DerivedShSlopec
            self.assertEqual(slopec.cuda_graph is not None, uses_graph, cls.__name__)
            for step, ref_step in zip(res, ref_no_sn):
                np.testing.assert_allclose(step[0], ref_step[0] - cpuArray(sn.slopes),
                                           rtol=1e-5, atol=1e-6, err_msg=cls.__name__)

    def test_derived_class_overriding_trigger_code_raises(self):
        """trigger_code() applies the slope corrections, and must not be overridden"""
        with self.assertRaisesRegex(TypeError, 'compute_slopes'):
            class WrongShSlopec(ShSlopec):
                def trigger_code(self):
                    self.calc_slopes_nofor()

    @unittest.skipIf(specula.cp is None, 'GPU not available')
    def test_reallocated_input_raises_with_cuda_graph(self):
        """With a CUDA graph, an input reallocated by its producer raises an error,
        instead of the graph silently reading the old array"""
        subapdata, _, frames = self._frames_and_subapdata(0, specula.cp)
        pixels = Pixels(frames[0].shape[1], frames[0].shape[0], target_device_idx=0)
        slopec = ShSlopec(subapdata, target_device_idx=0)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.setup()
        for i, frame in enumerate(frames):
            t = slopec.seconds_to_t(i + 1)
            if i < 2:
                pixels.pixels[:] = specula.cp.asarray(frame)
            else:
                pixels.pixels = specula.cp.asarray(frame)
            pixels.generation_time = t
            slopec.check_ready(t)
            if i < 2:
                slopec.trigger()
                slopec.post_trigger()
            else:
                with self.assertRaisesRegex(RuntimeError, 'in_pixels has been reallocated'):
                    slopec.trigger()
                break

    @cpu_and_gpu
    def test_dynamic_dark_calibrator_chain(self, target_device_idx, xp):
        """
        Pixels -> DynamicDarkCalibrator -> ShSlopec, with a CUDA graph on GPU:
        the slopes must follow the input pixels at every step.
        """
        from specula.processing_objects.dynamic_dark_calibrator import DynamicDarkCalibrator

        subapdata, _, frames = self._frames_and_subapdata(target_device_idx, xp, nframes=5)
        pixels = Pixels(frames[0].shape[1], frames[0].shape[0], target_device_idx=target_device_idx)
        dark = DynamicDarkCalibrator(data_dir='', nframes=1, target_device_idx=target_device_idx)
        dark.inputs['in_pixels'].set(pixels)
        dark.setup()
        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(dark.outputs['out_subtracted_pixels'])
        slopec.setup()

        _, expected = self._run_slopec_steps(subapdata, frames, target_device_idx, False)
        for i, frame in enumerate(frames):
            t = slopec.seconds_to_t(i + 1)
            pixels.pixels[:] = xp.asarray(frame)
            pixels.generation_time = t
            for obj in [dark, slopec]:
                obj.check_ready(t)
                obj.trigger()
                obj.post_trigger()
            np.testing.assert_allclose(cpuArray(slopec.slopes.slopes), expected[i][0],
                                       rtol=1e-5, atol=1e-6, err_msg=f'step {i}')
        self.assertEqual(slopec.cuda_graph is not None, target_device_idx >= 0)
