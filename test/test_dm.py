import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula.base_value import BaseValue
from specula.processing_objects.dm import DM
from specula.data_objects.ifunc import IFunc
from specula.data_objects.m2c import M2C
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

from specula import cpuArray
from numpy.testing import assert_array_almost_equal


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

    
    @cpu_and_gpu
    def test_dm_stroke_clipping(self, target_device_idx, xp):
        """ Test command clipping """
        simul_params = SimulParams(time_step = 1, pixel_pupil=5, pixel_pitch=1)
        in_dm = BaseValue(xp.ones(6), target_device_idx=target_device_idx)
        t = 1
        in_dm.value = xp.ones(6)*(-1)**xp.arange(1,7)
        in_dm.generation_time = t

        # Single value clipping
        max_amp = 0.5
        dm1 = DM(simul_params, height=0, type_str='zernike', nmodes=6, target_device_idx=target_device_idx, stroke=max_amp)
        dm1.inputs['in_command'].set(in_dm)
        dm1.setup()
        dm1.check_ready(t)
        dm1.trigger()
        dm1.post_trigger()

        got = dm1.outputs['out_clipped_command'].value
        want = (-1)**xp.arange(1,7)*max_amp
        assert_array_almost_equal(cpuArray(got),cpuArray(want))

        # Passing a list of values
        max_amps = [0.6,0.5,0.4,0.3,0.2,0.1]
        dm2 = DM(simul_params, height=0, type_str='zernike', nmodes=6, target_device_idx=target_device_idx, stroke=max_amps)
        dm2.inputs['in_command'].set(in_dm)
        dm2.setup()
        dm2.check_ready(t)
        dm2.trigger()
        dm2.post_trigger()

        got = dm2.outputs['out_clipped_command'].value
        want = xp.array(max_amps)*(-1)**xp.arange(1,7)
        assert_array_almost_equal(cpuArray(got),cpuArray(want))

        # test passing a list of incorrect length
        with self.assertRaises(ValueError):
            _ = DM(simul_params, height=0, type_str='zernike', nmodes=6, target_device_idx=target_device_idx, stroke=[0.3,0.2,0.1])

    @cpu_and_gpu
    def test_dm_output_phase_vs_reference(self, target_device_idx, xp):
        '''Test out_layer.phaseInNm against an explicit reference, for the three
        mode-selection paths: (start_mode, nmodes) slice, idx_modes, and m2c.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        sign = -1

        def run_dm(dm, cmd):
            in_dm = BaseValue(value=xp.asarray(cmd), target_device_idx=target_device_idx)
            in_dm.generation_time = t
            dm.inputs['in_command'].set(in_dm)
            dm.setup()
            dm.check_ready(t)
            dm.trigger()
            dm.post_trigger()

        # (a) start_mode + nmodes (slice path)
        start_mode, nmodes = 2, 6
        cmd_a = np.array([0.3, -0.5, 0.2, -0.1])
        dm_a = DM(simul_params, height=0, type_str='zernike', nmodes=nmodes, start_mode=start_mode,
                  target_device_idx=target_device_idx)
        run_dm(dm_a, cmd_a)

        ifunc_a = cpuArray(dm_a.ifunc)
        idx_a = dm_a.ifunc_obj.idx_inf_func
        ref_a = sign * cmd_a @ ifunc_a[start_mode:nmodes, :]
        got_a = cpuArray(dm_a.outputs['out_layer'].phaseInNm[idx_a])
        assert_array_almost_equal(got_a, ref_a)

        # values outside the mask must be zero
        mask_a = cpuArray(dm_a.mask)
        outside_a = np.where(mask_a == 0)
        assert_array_almost_equal(cpuArray(dm_a.outputs['out_layer'].phaseInNm)[outside_a], 0)

        # (b) idx_modes
        idx_modes = [1, 3, 4]
        cmd_b = np.array([0.4, 0.15, -0.25])
        dm_b = DM(simul_params, height=0, type_str='zernike', idx_modes=idx_modes,
                  target_device_idx=target_device_idx)
        run_dm(dm_b, cmd_b)

        ifunc_b = cpuArray(dm_b.ifunc)
        idx_b = dm_b.ifunc_obj.idx_inf_func
        ref_b = sign * cmd_b @ ifunc_b[idx_modes, :]
        got_b = cpuArray(dm_b.outputs['out_layer'].phaseInNm[idx_b])
        assert_array_almost_equal(got_b, ref_b)

        # (c) m2c, combined with start_mode/nmodes as in MORFEO configs
        start_mode_c, nmodes_c = 1, 4
        rng = np.random.RandomState(42)
        m2c_arr = rng.randn(6, 5)  # (n_ifunc_modes, n_m2c_modes)
        cmd_c = np.array([0.2, -0.3, 0.05])

        ifunc_c = IFunc(type_str='zernike', npixels=16, nmodes=6, target_device_idx=target_device_idx)
        m2c_obj = M2C(m2c_arr, target_device_idx=target_device_idx)
        dm_c = DM(simul_params, height=0, ifunc=ifunc_c, m2c=m2c_obj,
                  nmodes=nmodes_c, start_mode=start_mode_c, target_device_idx=target_device_idx)
        run_dm(dm_c, cmd_c)

        ifunc_c_full = cpuArray(dm_c.ifunc)
        idx_c = dm_c.ifunc_obj.idx_inf_func
        actuator_cmd = m2c_arr[:, start_mode_c:nmodes_c] @ cmd_c
        ref_c = sign * actuator_cmd @ ifunc_c_full
        got_c = cpuArray(dm_c.outputs['out_layer'].phaseInNm[idx_c])
        assert_array_almost_equal(got_c, ref_c)

    @cpu_and_gpu
    def test_dm_precision(self, target_device_idx, xp):
        '''Test that phaseInNm, out_clipped_command and stroke dtype follow the
        requested precision (0=double, 1=single), regardless of the input command dtype.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=8, pixel_pitch=1)
        t = 1
        nmodes = 4
        cmd = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float64)

        for precision in (0, 1):
            expected_dtype = np.float64 if precision == 0 else np.float32
            for stroke in (None, 0.3, [0.3, 0.2, 0.1, 0.05]):
                with self.subTest(precision=precision, stroke=stroke):
                    # Input command is always float64, to verify the DM does not
                    # let it promote the (possibly single-precision) output.
                    in_dm = BaseValue(value=xp.asarray(cmd, dtype=xp.float64),
                                      target_device_idx=target_device_idx, precision=0)
                    in_dm.generation_time = t
                    dm = DM(simul_params, height=0, type_str='zernike', nmodes=nmodes,
                            target_device_idx=target_device_idx, precision=precision, stroke=stroke)
                    dm.inputs['in_command'].set(in_dm)
                    dm.setup()
                    dm.check_ready(t)
                    dm.trigger()
                    dm.post_trigger()

                    self.assertEqual(cpuArray(dm.outputs['out_layer'].phaseInNm).dtype, expected_dtype)
                    self.assertEqual(cpuArray(dm.outputs['out_clipped_command'].value).dtype, expected_dtype)
                    if stroke is not None:
                        self.assertEqual(cpuArray(dm.stroke).dtype, expected_dtype)


    @cpu_and_gpu
    def test_dm_m2c_rows_must_match_ifunc_modes(self, target_device_idx, xp):
        '''m2c with a number of rows different from the ifunc modes raises ValueError'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=6, target_device_idx=target_device_idx)
        m2c = M2C(np.ones((5, 4)), target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, target_device_idx=target_device_idx)
