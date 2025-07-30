
import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
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
        subap_npx = 120

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

        return sh, v, m

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
        subap_on_diameter = 2
        pxscale_arcsec = 0.1
        subap_npx = 120

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh, v, m = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2, tilt_value / 2, pixel_pupil)
        
        # Tilted wavefront
        ef.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        ef.generation_time = t+1

        sh.check_ready(t+1)
        sh.trigger()
        sh.post_trigger()
        tilted = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*tilted.shape, target_device_idx=target_device_idx)
        pixels.pixels = tilted
        pixels.generation_time = t+1

        # Create the slope computer object
        subapdata = SubapData(idxs=v, display_map=m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)
        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t+1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        # Expected value: 1/(subap_npx/2)
        expected_slope = 1.0 / (subap_npx / 2)
        s_x = cpuArray(slopes.xslopes)

        # All X slopes (all slopes are valid) should be close to the expected value
        np.testing.assert_allclose(s_x, expected_slope, rtol=1e-2, atol=1e-2)

    @cpu_and_gpu
    def test_weight_int_pixel_dt(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt.
        """
        
        # Flat wavefront
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        t = 1
        subap_on_diameter = 2
        weight_int_pixel_dt = 3.0

        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh, v, m = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(ef)
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
        subapdata = SubapData(idxs=v, display_map=m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)
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

        flat_idx = slopec.subap_idx.flatten()
        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d.flat[flat_idx] = last_weights.T.flatten()

        # first step is skipped in the int_pixels computation, so the expected weights
        # are the average of the second and third frame
        expected_weights = (intensity + xp.roll(intensity, shift=1, axis=0))
        expected_weights = expected_weights / expected_weights.max()

        np.testing.assert_allclose(last_weights_2d, expected_weights, atol=1e-3)

        # Then compares slopec.int_pixels.pixels and slopec.pixels.pixels:
        # they must be equal because the accumulation was resetted
        # expect for a scalar factor equal factor = float(self.seconds_to_t(t-self.t_previous)) / float(self.weight_int_pixel_dt)
        factor = t_seconds / weight_int_pixel_dt
        expected_int_pixels = pixels.pixels.astype(slopec.dtype) * factor
        np.testing.assert_allclose(slopec.int_pixels.pixels, expected_int_pixels, atol=1e-3)

    @cpu_and_gpu
    def test_weight_int_pixel_dt_window(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt and window_int_pixel.
        """
        pixel_pupil = 20
        pixel_pitch = 0.05
        subap_on_diameter = 2
        weight_int_pixel_dt = 3.0

        sh, v, m = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        t = 1
        # Flat wavefront
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)
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
        subapdata = SubapData(idxs=v, display_map=m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)
        slopec = ShSlopec(subapdata, weight_int_pixel_dt=weight_int_pixel_dt, window_int_pixel=True, target_device_idx=target_device_idx)
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

        flat_idx = slopec.subap_idx.flatten()
        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d.flat[flat_idx] = last_weights.T.flatten()

        expected_weights = xp.zeros_like(last_weights_2d)
        expected_weights[4:8,   4:8] = 1.0
        expected_weights[16:20, 16:20] = 1.0
        expected_weights[4:8,   16:20] = 1.0
        expected_weights[16:20, 4:8] = 1.0

        np.testing.assert_equal(last_weights_2d, expected_weights, err_msg="Weight map does not match expected values.")


    @cpu_and_gpu
    def test_shslopec_slopesnull(self, target_device_idx, xp):
        '''
        Test that a SH Slopec correctly subtracts slope nulls (non-interleaved)
        '''
        # Flat wavefront
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        subap_on_diameter = 2
        t = 1
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh, v, m = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(ef)
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
        subapdata = SubapData(idxs=v, display_map = m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)

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
        pixel_pupil = 20
        pixel_pitch = 0.05
        subap_on_diameter = 2
        t=1
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh, v, m = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(ef)
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
        subapdata = SubapData(idxs=v, display_map = m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)

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
