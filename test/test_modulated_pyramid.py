

import specula
specula.init(0)  # Default target device

import unittest
from scipy.ndimage import rotate

from specula import cp, np
from specula import cpuArray

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from test.specula_testlib import cpu_and_gpu


class TestModulatedPyramid(unittest.TestCase):

    @cpu_and_gpu
    def test_pyramid_rotation(self, target_device_idx, xp):
        '''
        Test that input EF rotation is correctly handled, by comparing
        the pyramid output with a non-rotated EF input plus a rotation parameter,
        and a rotated EF without the rotatio parameter
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        rotAnglePhInDeg = 1.

        simul_params =  SimulParams(pixel_pupil=pixel_pupil,
                                    pixel_pitch=pixel_pitch,
                                    )

        pyr_not_rotated = ModulatedPyramid(simul_params=simul_params,
                                           wavelengthInNm=750,
                                           fov=2.0,
                                           pup_diam=30,
                                           output_resolution=80,
                                           mod_amp=3,
                                           rotAnglePhInDeg=0,
                                           target_device_idx=target_device_idx)

        pyr_rotated = ModulatedPyramid(simul_params=simul_params,
                                       wavelengthInNm=750,
                                       fov=2.0,
                                       pup_diam=30,
                                       output_resolution=80,
                                       mod_amp=3,
                                       rotAnglePhInDeg=rotAnglePhInDeg,
                                       target_device_idx=target_device_idx)

        # tilt corresponding to pxscale_arcsec
        tilt_value = xp.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = xp.linspace(-tilt_value / 2, tilt_value / 2, pixel_pupil)

        # Tilted wavefront, non-rotated
        ef_non_rotated = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef_non_rotated.phaseInNm[:] = xp.array(xp.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        ef_non_rotated.generation_time = t

        ef_rotated = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef_rotated.phaseInNm[:] = xp.array(rotate(cpuArray(ef_non_rotated.phaseInNm), rotAnglePhInDeg, reshape=False))
        ef_rotated.generation_time = t

        pupilstop = Pupilstop(simul_params=simul_params, target_device_idx=target_device_idx)
        ef_non_rotated.A *= pupilstop.A
        ef_rotated.A *= pupilstop.A        

        pyr_not_rotated.inputs['in_ef'].set(ef_rotated)
        pyr_rotated.inputs['in_ef'].set(ef_non_rotated)

        pyr_not_rotated.setup()
        pyr_not_rotated.check_ready(t)
        pyr_not_rotated.prepare_trigger(t)
        pyr_not_rotated.trigger()
        pyr_not_rotated.post_trigger()
        i_non_rotated = pyr_not_rotated.outputs['out_i'].i

        pyr_rotated.setup()
        pyr_rotated.check_ready(t)
        pyr_rotated.prepare_trigger(t)
        pyr_rotated.trigger()
        pyr_rotated.post_trigger()
        i_rotated = pyr_rotated.outputs['out_i'].i

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(cpuArray(i_non_rotated))
        plt.figure()
        plt.imshow(cpuArray(i_rotated))
        plt.show()
        np.testing.assert_array_almost_equal(cpuArray(i_non_rotated), cpuArray(i_rotated), decimal=3) 
        