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
    def test_dm_force_limiting_diagonal(self, target_device_idx, xp):
        '''Force limiting with a diagonal stiffness and m2c=identity: each mode
        drives a single actuator, so the expected number of kept modes and the
        resulting forces can be derived analytically.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        nmodes = 6

        def run_dm(dm, cmd):
            in_dm = BaseValue(value=xp.asarray(cmd), target_device_idx=target_device_idx)
            in_dm.generation_time = t
            dm.inputs['in_command'].set(in_dm)
            dm.setup()
            dm.check_ready(t)
            dm.trigger()
            dm.post_trigger()

        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=nmodes, target_device_idx=target_device_idx)
        m2c = M2C(np.eye(nmodes), target_device_idx=target_device_idx)
        k = np.array([1., 2., 3., 4., 5., 6.])
        K = np.diag(k)
        cmd = np.ones(nmodes)

        # Scalar max_force: mode index 5 alone (force=6) violates fmax=5 -> keep 5 modes
        dm1 = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=5.0,
                 target_device_idx=target_device_idx)
        run_dm(dm1, cmd)

        expected_cmd = cmd.copy()
        expected_cmd[5:] = 0
        expected_forces = K @ expected_cmd

        self.assertEqual(int(cpuArray(dm1.outputs['out_force_nmodes'].value)[0]), 5)
        assert_array_almost_equal(cpuArray(dm1.outputs['out_clipped_command'].value), expected_cmd)
        assert_array_almost_equal(cpuArray(dm1.outputs['out_forces'].value), expected_forces)
        self.assertTrue(np.all(np.abs(cpuArray(dm1.outputs['out_forces'].value)) <= 5.0 + 1e-6))

        # Per-actuator max_force list: violation appears at mode index 3 (force=4 > 3.5)
        max_force_list = [1.5, 2.5, 3.5, 3.5, 5.5, 6.5]
        dm2 = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=max_force_list,
                 target_device_idx=target_device_idx)
        run_dm(dm2, cmd)

        expected_cmd2 = cmd.copy()
        expected_cmd2[3:] = 0
        expected_forces2 = K @ expected_cmd2

        self.assertEqual(int(cpuArray(dm2.outputs['out_force_nmodes'].value)[0]), 3)
        assert_array_almost_equal(cpuArray(dm2.outputs['out_clipped_command'].value), expected_cmd2)
        assert_array_almost_equal(cpuArray(dm2.outputs['out_forces'].value), expected_forces2)
        self.assertTrue(np.all(np.abs(cpuArray(dm2.outputs['out_forces'].value)) <= np.array(max_force_list) + 1e-6))

    @cpu_and_gpu
    def test_dm_force_limiting_realistic_stiffness(self, target_device_idx, xp):
        '''Force limiting with a physically-motivated stiffness: squared graph
        Laplacian of a 4x4 actuator grid, m2c = eigenvectors sorted by increasing
        eigenvalue (modes ordered by increasing spatial frequency/stiffness, as
        force limiting assumes). The expected number of kept modes is computed
        independently by brute force, without reusing the dm.py logic.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        n_side = 4
        n_act = n_side * n_side

        # Graph Laplacian of the 4x4 grid (4-connectivity), squared for a
        # curvature-like force cost that grows with spatial frequency.
        adj = np.zeros((n_act, n_act))
        for i in range(n_side):
            for j in range(n_side):
                idx = i * n_side + j
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_side and 0 <= nj < n_side:
                        adj[idx, ni * n_side + nj] = 1.0
        lap = np.diag(adj.sum(axis=1)) - adj
        K = lap @ lap

        eigvals, eigvecs = np.linalg.eigh(K)  # ascending eigenvalues
        m2c_arr = eigvecs

        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=n_act, target_device_idx=target_device_idx)
        m2c = M2C(m2c_arr, target_device_idx=target_device_idx)

        rng = np.random.RandomState(123)
        cmd = rng.uniform(-1, 1, n_act)
        fmax = 0.5

        # Independent brute-force reference: largest n with max|K m2c[:, :n] c[:n]| <= fmax
        n_keep_expected = 0
        applied_expected = np.zeros(n_act)
        for n in range(n_act + 1):
            applied_cmd = m2c_arr[:, :n] @ cmd[:n]
            forces = K @ applied_cmd
            if np.max(np.abs(forces)) <= fmax:
                n_keep_expected = n
                applied_expected = applied_cmd
        assert n_keep_expected not in (0, n_act)  # sanity check: the test is non-trivial

        dm = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=fmax,
                target_device_idx=target_device_idx)

        in_dm = BaseValue(value=xp.asarray(cmd), target_device_idx=target_device_idx)
        in_dm.generation_time = t
        dm.inputs['in_command'].set(in_dm)
        dm.setup()
        dm.check_ready(t)
        dm.trigger()
        dm.post_trigger()

        self.assertEqual(int(cpuArray(dm.outputs['out_force_nmodes'].value)[0]), n_keep_expected)
        assert_array_almost_equal(cpuArray(dm.outputs['out_clipped_command'].value), applied_expected, decimal=5)
        assert_array_almost_equal(cpuArray(dm.outputs['out_forces'].value), K @ applied_expected, decimal=5)

    @cpu_and_gpu
    def test_dm_force_and_stroke_combinations(self, target_device_idx, xp):
        '''out_forces / out_force_nmodes in the four combinations of stroke
        and stiffness/max_force: only stroke, only force limiting, both
        (force limiting first, then stroke clipping), and stiffness without
        max_force (forces monitored but never truncated).'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        nmodes = 4

        def run_dm(dm, cmd):
            in_dm = BaseValue(value=xp.asarray(cmd), target_device_idx=target_device_idx)
            in_dm.generation_time = t
            dm.inputs['in_command'].set(in_dm)
            dm.setup()
            dm.check_ready(t)
            dm.trigger()
            dm.post_trigger()

        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=nmodes, target_device_idx=target_device_idx)
        m2c = M2C(np.eye(nmodes), target_device_idx=target_device_idx)
        K = np.diag([1., 2., 3., 4.])
        cmd = np.array([1.0, 1.0, 1.0, 1.0])

        # (a) only stroke: out_forces is empty, out_force_nmodes reports all modes
        dm_a = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stroke=0.5,
                  target_device_idx=target_device_idx)
        run_dm(dm_a, cmd)
        self.assertEqual(cpuArray(dm_a.outputs['out_forces'].value).size, 0)
        self.assertEqual(int(cpuArray(dm_a.outputs['out_force_nmodes'].value)[0]), nmodes)
        assert_array_almost_equal(cpuArray(dm_a.outputs['out_clipped_command'].value), np.full(nmodes, 0.5))

        # (b) only force limiting: mode index 3 (force=4) violates fmax=3.5 -> keep 3 modes
        dm_b = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=3.5,
                  target_device_idx=target_device_idx)
        run_dm(dm_b, cmd)
        expected_cmd_b = np.array([1., 1., 1., 0.])
        self.assertEqual(int(cpuArray(dm_b.outputs['out_force_nmodes'].value)[0]), 3)
        assert_array_almost_equal(cpuArray(dm_b.outputs['out_clipped_command'].value), expected_cmd_b)
        assert_array_almost_equal(cpuArray(dm_b.outputs['out_forces'].value), K @ expected_cmd_b)

        # (c) force limiting + stroke: truncation happens first, stroke then clips the survivors
        dm_c = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=3.5,
                  stroke=0.8, target_device_idx=target_device_idx)
        run_dm(dm_c, cmd)
        expected_cmd_c = np.array([0.8, 0.8, 0.8, 0.])
        self.assertEqual(int(cpuArray(dm_c.outputs['out_force_nmodes'].value)[0]), 3)
        assert_array_almost_equal(cpuArray(dm_c.outputs['out_clipped_command'].value), expected_cmd_c)
        assert_array_almost_equal(cpuArray(dm_c.outputs['out_forces'].value), K @ expected_cmd_c)

        # (d) stiffness without max_force: forces are monitored, but never truncated
        dm_d = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, stroke=0.8,
                  target_device_idx=target_device_idx)
        run_dm(dm_d, cmd)
        expected_cmd_d = np.full(nmodes, 0.8)
        self.assertEqual(int(cpuArray(dm_d.outputs['out_force_nmodes'].value)[0]), nmodes)
        assert_array_almost_equal(cpuArray(dm_d.outputs['out_clipped_command'].value), expected_cmd_d)
        assert_array_almost_equal(cpuArray(dm_d.outputs['out_forces'].value), K @ expected_cmd_d)

    @cpu_and_gpu
    def test_dm_force_limiting_edge_cases(self, target_device_idx, xp):
        '''A command already within the force limits is left unchanged; a
        command that already violates the limit at the very first mode
        zeroes the whole applied command (out_force_nmodes == 0).'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        nmodes = 4

        def run_dm(dm, cmd):
            in_dm = BaseValue(value=xp.asarray(cmd), target_device_idx=target_device_idx)
            in_dm.generation_time = t
            dm.inputs['in_command'].set(in_dm)
            dm.setup()
            dm.check_ready(t)
            dm.trigger()
            dm.post_trigger()

        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=nmodes, target_device_idx=target_device_idx)
        m2c = M2C(np.eye(nmodes), target_device_idx=target_device_idx)
        K = np.diag([1., 2., 3., 4.])
        cmd = np.array([1., 1., 1., 1.])

        # Within limits: nothing is discarded
        dm1 = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=100.,
                 target_device_idx=target_device_idx)
        run_dm(dm1, cmd)
        self.assertEqual(int(cpuArray(dm1.outputs['out_force_nmodes'].value)[0]), nmodes)
        assert_array_almost_equal(cpuArray(dm1.outputs['out_clipped_command'].value), cmd)

        # First mode alone already exceeds the limit -> whole command is zeroed
        dm2 = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=0.5,
                 target_device_idx=target_device_idx)
        run_dm(dm2, cmd)
        self.assertEqual(int(cpuArray(dm2.outputs['out_force_nmodes'].value)[0]), 0)
        assert_array_almost_equal(cpuArray(dm2.outputs['out_clipped_command'].value), np.zeros(nmodes))
        assert_array_almost_equal(cpuArray(dm2.outputs['out_forces'].value), np.zeros(nmodes))

    @cpu_and_gpu
    def test_dm_force_limiting_errors(self, target_device_idx, xp):
        '''stiffness/max_force parameter validation.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        nmodes = 4
        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=nmodes, target_device_idx=target_device_idx)
        m2c = M2C(np.eye(nmodes), target_device_idx=target_device_idx)
        K = np.eye(nmodes)

        # stiffness requires m2c
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, stiffness=K, target_device_idx=target_device_idx)

        # stiffness must be (nact, nact), with nact = m2c rows
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=np.eye(nmodes + 1),
               target_device_idx=target_device_idx)

        # max_force requires stiffness
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, max_force=1.0, target_device_idx=target_device_idx)

        # max_force list length must match the number of actuators
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=[1.0, 2.0],
               target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_dm_precision_with_stiffness(self, target_device_idx, xp):
        '''out_forces and out_clipped_command dtype follow precision (0=double,
        1=single) when stiffness/max_force are active.'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        t = 1
        nmodes = 4
        cmd = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        K = np.diag([1., 2., 3., 4.])

        for precision in (0, 1):
            expected_dtype = np.float64 if precision == 0 else np.float32
            ifunc = IFunc(type_str='zernike', npixels=16, nmodes=nmodes,
                          target_device_idx=target_device_idx, precision=precision)
            m2c = M2C(np.eye(nmodes), target_device_idx=target_device_idx, precision=precision)

            # Input command is always float64, to verify the DM does not let it
            # promote the (possibly single-precision) outputs.
            in_dm = BaseValue(value=xp.asarray(cmd, dtype=xp.float64),
                              target_device_idx=target_device_idx, precision=0)
            in_dm.generation_time = t
            dm = DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, stiffness=K, max_force=3.5,
                    target_device_idx=target_device_idx, precision=precision)
            dm.inputs['in_command'].set(in_dm)
            dm.setup()
            dm.check_ready(t)
            dm.trigger()
            dm.post_trigger()

            self.assertEqual(cpuArray(dm.outputs['out_forces'].value).dtype, expected_dtype)
            self.assertEqual(cpuArray(dm.outputs['out_clipped_command'].value).dtype, expected_dtype)

    @cpu_and_gpu
    def test_dm_m2c_rows_must_match_ifunc_modes(self, target_device_idx, xp):
        '''m2c with a number of rows different from the ifunc modes raises ValueError'''
        simul_params = SimulParams(time_step=1, pixel_pupil=16, pixel_pitch=1)
        ifunc = IFunc(type_str='zernike', npixels=16, nmodes=6, target_device_idx=target_device_idx)
        m2c = M2C(np.ones((5, 4)), target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            DM(simul_params, height=0, ifunc=ifunc, m2c=m2c, target_device_idx=target_device_idx)
