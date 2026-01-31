import specula
specula.init(0)  # Default target device

import unittest

from specula.base_value import BaseValue
from specula.processing_objects.dm import DM
from specula.data_objects.ifunc import IFunc
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu


class TestDM(unittest.TestCase):

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
            _ = DM(simul_params, height=0, type_str='zernike',
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
            _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
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
            _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
                    ifunc=ifunc, npixels=5, target_device_idx=target_device_idx)

        # Npixels same as from pixel_pitch
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               ifunc=ifunc, npixels=3, target_device_idx=target_device_idx)

        # Npixels not given
        _ = DM(simul_params, height=0, type_str='zernike', nmodes=4,
               ifunc=ifunc, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_dm_double_mode_selection(self, target_device_idx, xp):
        ''' Test that double mode selection:
            - nmodes and start_mode are OK
            - idx_modes is OK
            - nmodes with idx_modes raises an error
            - start_mode with idx_modes raises an error'''
        simul_params = SimulParams(time_step = 2, pixel_pupil=5, pixel_pitch=1)

        # Input command with 3 values (for the 6 nmodes, starting from mode 3)
        in_dm = BaseValue(xp.ones(3), target_device_idx=target_device_idx)
        t = 1
        in_dm.value = xp.ones(3)
        in_dm.generation_time = t

        dm1 = DM(simul_params, height=0, type_str='zernike', nmodes=6, start_mode=3, target_device_idx=target_device_idx)
        dm1.inputs['in_command'].set(in_dm)

        # Should NOT raise ValueError or IndexError
        dm1.setup()
        dm1.check_ready(t)
        dm1.trigger()
        dm1.post_trigger()

        idx_modes = [2,3,4]
        dm2 = DM(simul_params, height=0, type_str='zernike', idx_modes=idx_modes, target_device_idx=target_device_idx)
        dm2.inputs['in_command'].set(in_dm)

        # Should NOT raise ValueError or IndexError
        dm2.setup()
        dm2.check_ready(t)
        dm2.trigger()
        dm2.post_trigger()

        with self.assertRaises(ValueError):
            dm3 = DM(simul_params, height=0, type_str='zernike', nmodes=6, idx_modes=idx_modes, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            dm4 = DM(simul_params, height=0, type_str='zernike', start_mode=3, idx_modes=idx_modes, target_device_idx=target_device_idx)