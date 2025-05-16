
import specula
from specula.data_objects.ifunc import IFunc
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.dm import DM
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestCM(unittest.TestCase):

    @cpu_and_gpu
    def test_pupilstop_from_cpu(self, target_device_idx, xp):
        '''Test that a DM can be initialized with a pupilstop from any device'''
        simul_params = SimulParams(time_step = 2, pixel_pupil=10, pixel_pitch=1)
        pupilstop = Pupilstop(simul_params)

        # does not raise in any case
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               pupilstop=pupilstop, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_dm_nmodes_is_mandatory_with_zernike(self, target_device_idx, xp):
        '''Test that the nmodes parameter is mandatory with DM of zernike type'''
        simul_params = SimulParams(time_step = 2, pixel_pupil=10, pixel_pitch=1)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)

        # Missing nmodes
        with self.assertRaises(ValueError):
            dm = DM(simul_params, height=0, type_str='zernike',
                    pupilstop=pupilstop, npixels=5, target_device_idx=target_device_idx)

        # nmodes present, does not raise
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4, 
               pupilstop=pupilstop, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_dm_npixels_matches_pupilstop_mask(self, target_device_idx, xp):
        '''Test that the npixels, if given, is checked against the pupilstop shape'''
        simul_params = SimulParams(time_step = 2, pixel_pupil=10, pixel_pitch=1)
        pupilstop = Pupilstop(simul_params, target_device_idx=target_device_idx)

        # Npixels different from pixel_pitch
        with self.assertRaises(ValueError):
            dm = DM(simul_params, height=0, type_str='zernike', nmodes=4,
                    pupilstop=pupilstop, npixels=5, target_device_idx=target_device_idx)

        # Npixels same as from pixel_pitch
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               pupilstop=pupilstop, npixels=10, target_device_idx=target_device_idx)

        # Npixels not given
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               pupilstop=pupilstop, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_dm_npixels_matches_ifunc_mask(self, target_device_idx, xp):
        '''Test that the npixels, if given, is checked against the ifunc mask shape'''
        simul_params = SimulParams(time_step = 2, pixel_pupil=3, pixel_pitch=1)
        ifunc = IFunc(xp.ones((9,3)), mask=xp.ones((3,3)))

        # Npixels different from pixel_pitch
        with self.assertRaises(ValueError):
            dm = DM(simul_params, height=0, type_str='zernike', nmodes=4,
                    ifunc=ifunc, npixels=5, target_device_idx=target_device_idx)

        # Npixels same as from pixel_pitch
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               ifunc=ifunc, npixels=3, target_device_idx=target_device_idx)

        # Npixels not given
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               ifunc=ifunc, target_device_idx=target_device_idx)



if __name__ == '__main__':
    unittest.main()