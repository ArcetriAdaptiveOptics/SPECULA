

import specula
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
specula.init(0)  # Default target device

import unittest
from scipy.ndimage import rotate

from specula import cp, np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from test.specula_testlib import cpu_and_gpu


class TestSH(unittest.TestCase):

    @cpu_and_gpu
    def test_sh_flux(self, target_device_idx, xp):
        
        ref_S0 = 100
        t = 1
        
        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=20,
                subap_npx=6,
                target_device_idx=target_device_idx)
        
        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']
        
        np.testing.assert_almost_equal(xp.sum(intensity.i), ref_S0 * ef.masked_area())

    @cpu_and_gpu
    def test_pixelscale(self, target_device_idx, xp):
        '''
        Test that pixelscale is correctly handled, by comparing spots from a flat 
        wavefront and from a tilted one. The introduced tilt corresponds to exactly 1 pixel,
        and we verify that the resulting intensity field is indeed shifted by 1 pixel
        in the correct direction
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        flat = sh.outputs['out_i'].i.copy()
        
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
        
        flat_shifted = np.roll(flat, (0, 1))

        # Remove the left column edges on each subap (comparison is invalid after roll)
        flat_shifted[:, ::sh_npix] = 0
        tilted[:, ::sh_npix] = 0
        
        np.testing.assert_array_almost_equal(cpuArray(tilted), cpuArray(flat_shifted), decimal=3) 
        

    @cpu_and_gpu
    def test_zeros_cache(self, target_device_idx, xp):
        '''
        Test that arrays are re-used between SH instances on the same target
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        sh1 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh2 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh3 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=30,  # Different
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh1.inputs['in_ef'].set(ef)
        sh2.inputs['in_ef'].set(ef)
        sh3.inputs['in_ef'].set(ef)

        sh1.setup()
        sh2.setup()
        sh3.setup()
        
        assert id(sh1._wf3) == id(sh2._wf3) 
        assert id(sh1._wf3) != id(sh3._wf3) 

    @cpu_and_gpu
    def test_sh_rotation(self, target_device_idx, xp):
        '''
        Test that input EF rotation is correctly handled, by comparing
        the SH output with a non-rotated EF input plus a rotation parameter,
        and a rotated EF without the rotatio parameter
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6
        rotAnglePhInDeg = 42.

        simul_params = SimulParams(pixel_pitch=pixel_pitch, pixel_pupil=pixel_pupil)
        sh_non_rotated = SH(wavelengthInNm=500,
                            subap_wanted_fov= sh_npix * pxscale_arcsec,
                            sensor_pxscale=pxscale_arcsec,
                            subap_on_diameter=20,
                            subap_npx=sh_npix,
                            rotAnglePhInDeg=0,
                            target_device_idx=target_device_idx)

        sh_rotated = SH(wavelengthInNm=500,
                        subap_wanted_fov= sh_npix * pxscale_arcsec,
                        sensor_pxscale=pxscale_arcsec,
                        subap_on_diameter=20,
                        subap_npx=sh_npix,
                        rotAnglePhInDeg=rotAnglePhInDeg,
                        target_device_idx=target_device_idx)

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2, tilt_value / 2, pixel_pupil)

        # Tilted wavefront, non-rotated
        ef_non_rotated = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef_non_rotated.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        ef_non_rotated.generation_time = t

        ef_rotated = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef_rotated.phaseInNm[:] = xp.array(rotate(cpuArray(ef_non_rotated.phaseInNm), rotAnglePhInDeg, reshape=False))
        ef_rotated.generation_time = t

        pupilstop = Pupilstop(simul_params=simul_params, target_device_idx=target_device_idx)
        ef_non_rotated.A *= pupilstop.A
        ef_rotated.A *= pupilstop.A        

        sh_non_rotated.inputs['in_ef'].set(ef_rotated)
        sh_rotated.inputs['in_ef'].set(ef_non_rotated)

        sh_non_rotated.setup()
        sh_non_rotated.check_ready(t)
        sh_non_rotated.trigger()
        sh_non_rotated.post_trigger()
        i_non_rotated = sh_non_rotated.outputs['out_i'].i

        sh_rotated.setup()
        sh_rotated.check_ready(t)
        sh_rotated.trigger()
        sh_rotated.post_trigger()
        i_rotated = sh_rotated.outputs['out_i'].i

        abstol = 0.005
        bad_elements = abs((i_non_rotated - i_rotated)) > abstol

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(cpuArray(i_non_rotated))
        # plt.figure()
        # plt.imshow(cpuArray(i_rotated))
        # plt.figure()
        # plt.imshow(cpuArray(i_non_rotated - i_rotated))
        # plt.figure()
        # plt.imshow(cpuArray(bad_elements))
        # plt.show()
        
        # Some pixels on the edges are different, but most are the same
        assert xp.count_nonzero(bad_elements) < 3
