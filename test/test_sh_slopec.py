
import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from test.specula_testlib import cpu_and_gpu


class TestShSlopec(unittest.TestCase):

    @cpu_and_gpu
    def test_pixelscale_and_slopes(self, target_device_idx, xp):
        """
        Test that verifies both pixel scale and slope computation for SH.
        A tilt that shifts the spot by 1 pixel should produce a slope of 1/(sh.subap_npx/2).
        """
        t = 1
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
        mask_subap = xp.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))

        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask_subap *= 0
                mask_subap[i*subap_npx:(i+1)*subap_npx,j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = xp.where(mask_subap == 1)
                map[count] = j * subap_on_diameter + i
                count += 1

        v = xp.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = xp.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = xp.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map[k]
        # ------------------------------------------------------------------------------

        # Create the SH object
        laser_launch_tel = LaserLaunchTelescope(spot_size=pxscale_arcsec,
                             target_device_idx=target_device_idx)

        sh = SH(wavelengthInNm=wavelengthInNm,
                subap_wanted_fov=subap_npx * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=subap_on_diameter,
                subap_npx=subap_npx,
                laser_launch_tel=laser_launch_tel,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
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