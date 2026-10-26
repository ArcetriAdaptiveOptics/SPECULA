import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray, np
from specula.loop_control import LoopControl
from specula.base_value import BaseValue
from specula.processing_objects.phase_unwrapper import PhaseUnwrapper

from test.specula_testlib import cpu_and_gpu


# Wavelengths (nm) used by most tests. synthetic_lambda = 1500*1600/100 = 24000,
# default max_capture = 12000, so max_k = max(1, 12000 // 1500 - 1) = 7.
LAMBDA_1 = 1500.0
LAMBDA_2 = 1600.0


def _wrap(x, wavelength):
    return ((x + wavelength / 2.0) % wavelength) - wavelength / 2.0


class TestPhaseUnwrapper(unittest.TestCase):

    def _run_once(self, unwrapper, p1_value, p2_value, target_device_idx, xp):
        '''Drive the object through a single LoopControl step and return (out_pistons, out_pistonsU).'''
        p1 = BaseValue(value=xp.array(p1_value), target_device_idx=target_device_idx)
        p2 = BaseValue(value=xp.array(p2_value), target_device_idx=target_device_idx)
        p1.generation_time = p1.seconds_to_t(1)
        p2.generation_time = p2.seconds_to_t(1)

        unwrapper.inputs['in_pistons_1'].set(p1)
        unwrapper.inputs['in_pistons_2'].set(p2)

        loop = LoopControl()
        loop.add(unwrapper, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        return (cpuArray(unwrapper.outputs['out_pistons'].value),
                cpuArray(unwrapper.outputs['out_pistonsU'].value))

    # ------------------------------------------------------------------ #
    # Construction / parameter validation
    # ------------------------------------------------------------------ #
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=-1.0, lambda_2=LAMBDA_2)
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_1)
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           temporal_filtering_mode='bogus')
        with self.assertRaises(ValueError):
            # max_capture must exceed lambda_1
            PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2, max_capture=1000.0)

    def test_derived_quantities(self):
        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)
        self.assertAlmostEqual(u.synthetic_lambda, 24000.0, places=6)
        self.assertAlmostEqual(u.max_capture, 12000.0, places=6)
        self.assertEqual(u.max_k, 7)
        self.assertGreaterEqual(u.max_k, 1)
        # temporal history buffers are bounded to the window size
        self.assertEqual(u.estimate_history.maxlen, u.temporal_window_size)

    def test_wrap_phase_range(self):
        x = np.linspace(-5000, 5000, 501)
        wrapped = PhaseUnwrapper.wrap_phase(x, LAMBDA_1)
        self.assertTrue(np.all(wrapped >= -LAMBDA_1 / 2 - 1e-6))
        self.assertTrue(np.all(wrapped < LAMBDA_1 / 2 + 1e-6))
        # wrapping is idempotent modulo the wavelength
        np.testing.assert_allclose(_wrap(x, LAMBDA_1), wrapped, atol=1e-6)

    # ------------------------------------------------------------------ #
    # Behaviour flags
    # ------------------------------------------------------------------ #
    @cpu_and_gpu
    def test_unwrap_disabled_is_passthrough(self, target_device_idx, xp):
        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           unwrap_enabled=False, target_device_idx=target_device_idx)
        p1_value = [10.0, -25.0, 123.0]
        out_pistons, out_pistonsU = self._run_once(
            u, p1_value, [0.0, 0.0, 0.0], target_device_idx, xp)
        np.testing.assert_array_almost_equal(out_pistons, p1_value)
        np.testing.assert_array_almost_equal(out_pistonsU, p1_value)

    @cpu_and_gpu
    def test_stage1_keeps_p1_when_residual_small(self, target_device_idx, xp):
        # True piston is tiny: p1 and p2 agree, Stage 1 residual is ~0, so the
        # estimate must stay at p1 (regression: it used to be forced to 0).
        true_piston = np.array([30.0, -12.0, 5.0])
        p1_value = _wrap(true_piston, LAMBDA_1)
        p2_value = _wrap(true_piston, LAMBDA_2)

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           two_stage_enabled=True, target_device_idx=target_device_idx)
        _, out_pistonsU = self._run_once(u, p1_value, p2_value, target_device_idx, xp)

        np.testing.assert_array_almost_equal(out_pistonsU, p1_value)
        self.assertFalse(np.allclose(out_pistonsU, 0.0))

    @cpu_and_gpu
    def test_stage2_recovers_wrapped_piston(self, target_device_idx, xp):
        # A piston larger than lambda_1: p1 alone is ambiguous, the two-wavelength
        # combination must recover it onto the lambda_1 grid.
        true_piston = np.array([3200.0, -4700.0, 6100.0])
        p1_value = _wrap(true_piston, LAMBDA_1)
        p2_value = _wrap(true_piston, LAMBDA_2)

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           two_stage_enabled=True, target_device_idx=target_device_idx)
        _, out_pistonsU = self._run_once(u, p1_value, p2_value, target_device_idx, xp)

        # estimate = p1_wrapped + k * lambda_1, and must match the true piston
        np.testing.assert_array_almost_equal(out_pistonsU, true_piston, decimal=3)

    @cpu_and_gpu
    def test_full_unwrapping_mode_matches_two_stage(self, target_device_idx, xp):
        true_piston = np.array([3200.0, -4700.0, 150.0])
        p1_value = _wrap(true_piston, LAMBDA_1)
        p2_value = _wrap(true_piston, LAMBDA_2)

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           two_stage_enabled=False, target_device_idx=target_device_idx)
        _, out_pistonsU = self._run_once(u, p1_value, p2_value, target_device_idx, xp)

        np.testing.assert_array_almost_equal(out_pistonsU, true_piston, decimal=3)

    @cpu_and_gpu
    def test_out_pistons_is_passthrough_of_input_1(self, target_device_idx, xp):
        true_piston = np.array([3200.0, -20.0])
        p1_value = _wrap(true_piston, LAMBDA_1)
        p2_value = _wrap(true_piston, LAMBDA_2)

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           target_device_idx=target_device_idx)
        out_pistons, _ = self._run_once(u, p1_value, p2_value, target_device_idx, xp)
        np.testing.assert_array_almost_equal(out_pistons, p1_value)

    @cpu_and_gpu
    def test_output_shape_and_generation_time(self, target_device_idx, xp):
        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           target_device_idx=target_device_idx)
        self._run_once(u, [10.0, 20.0, 30.0, 40.0], [0.0, 0.0, 0.0, 0.0],
                       target_device_idx, xp)
        self.assertEqual(u.outputs['out_pistons'].value.shape, (4,))
        self.assertEqual(u.outputs['out_pistonsU'].value.shape, (4,))
        # post_trigger() stamps both outputs with the trigger time
        gen_time = u.outputs['out_pistons'].generation_time
        self.assertGreater(gen_time, 0)
        self.assertEqual(u.outputs['out_pistonsU'].generation_time, gen_time)

    @cpu_and_gpu
    def test_shape_mismatch_raises_in_setup(self, target_device_idx, xp):
        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           target_device_idx=target_device_idx)
        p1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        p2 = BaseValue(value=xp.array([1.0, 2.0]), target_device_idx=target_device_idx)
        p1.generation_time = p1.seconds_to_t(1)
        p2.generation_time = p2.seconds_to_t(1)
        u.inputs['in_pistons_1'].set(p1)
        u.inputs['in_pistons_2'].set(p2)
        with self.assertRaises(ValueError):
            u.setup()

    # ------------------------------------------------------------------ #
    # Temporal filtering (driven manually over several steps)
    # ------------------------------------------------------------------ #
    def _drive(self, u, frames, target_device_idx, xp):
        '''Feed a list of (p1, p2) arrays one per timestep; return list of out_pistonsU.'''
        p1 = BaseValue(value=xp.array(frames[0][0]), target_device_idx=target_device_idx)
        p2 = BaseValue(value=xp.array(frames[0][1]), target_device_idx=target_device_idx)
        u.inputs['in_pistons_1'].set(p1)
        u.inputs['in_pistons_2'].set(p2)
        u.setup()

        results = []
        for step, (p1_value, p2_value) in enumerate(frames, start=1):
            t = u.seconds_to_t(step)
            p1.set_value(xp.array(p1_value))
            p2.set_value(xp.array(p2_value))
            p1.generation_time = t
            p2.generation_time = t
            u.check_ready(t)
            u.trigger()
            u.post_trigger()
            results.append(cpuArray(u.outputs['out_pistonsU'].value).copy())
        return results

    @cpu_and_gpu
    def test_temporal_none_is_pure_passthrough(self, target_device_idx, xp):
        true_piston = np.array([30.0])
        frame = (_wrap(true_piston, LAMBDA_1), _wrap(true_piston, LAMBDA_2))

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           temporal_filtering_mode='none',
                           target_device_idx=target_device_idx)
        results = self._drive(u, [frame] * 4, target_device_idx, xp)
        for r in results:
            np.testing.assert_array_almost_equal(r, frame[0])
        # history is still recorded, bounded to the window size
        self.assertLessEqual(len(u.estimate_history), u.temporal_window_size)

    @cpu_and_gpu
    def test_temporal_median_blends_towards_history(self, target_device_idx, xp):
        # Three clean frames at a stable value, then one frame with a spike on the
        # Stage-1 estimate. With 'median' blending the output is
        # 0.7 * current + 0.3 * median(history), so the spike is attenuated.
        stable = np.array([40.0])
        stable_frame = (_wrap(stable, LAMBDA_1), _wrap(stable, LAMBDA_2))

        spike = np.array([120.0])
        spike_frame = (_wrap(spike, LAMBDA_1), _wrap(spike, LAMBDA_2))

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           temporal_filtering_mode='median',
                           temporal_window_size=4,
                           target_device_idx=target_device_idx)
        results = self._drive(
            u, [stable_frame, stable_frame, stable_frame, spike_frame],
            target_device_idx, xp)

        # First frame: no usable history yet -> pure passthrough
        np.testing.assert_array_almost_equal(results[0], stable_frame[0])
        # Last frame: blended, so strictly between the stable value and the spike
        blended = results[3][0]
        self.assertGreater(blended, stable_frame[0][0] + 1e-3)
        self.assertLess(blended, spike_frame[0][0] - 1e-3)
        expected = 0.7 * spike_frame[0][0] + 0.3 * stable_frame[0][0]
        np.testing.assert_allclose(blended, expected, rtol=1e-4)

    @cpu_and_gpu
    def test_history_bounded_to_window(self, target_device_idx, xp):
        true_piston = np.array([30.0, -10.0])
        frame = (_wrap(true_piston, LAMBDA_1), _wrap(true_piston, LAMBDA_2))

        u = PhaseUnwrapper(lambda_1=LAMBDA_1, lambda_2=LAMBDA_2,
                           temporal_filtering_mode='weighted_average',
                           temporal_window_size=3,
                           target_device_idx=target_device_idx)
        self._drive(u, [frame] * 10, target_device_idx, xp)
        self.assertEqual(len(u.estimate_history), 3)
        self.assertEqual(len(u.confidence_history), 3)


# Wavelengths (nm) of the interval-mode tests (dual-wavelength LIFT case):
# lambda_2 - lambda_1 = 245 nm, synthetic_lambda = 10750 nm, max_capture = 5375 nm.
LIFT_L1 = 1505.0
LIFT_L2 = 1750.0


def _interval_reference(p1, p2, err_1, err_2, max_capture, npts=200001):
    '''Brute-force reference of the interval mode on a grid of trial pistons x:
    x is compatible with p_i if |wrap(x - p_i, lambda_i)| <= err_i.'''
    x = np.linspace(-max_capture, max_capture, npts)
    mask = (np.abs(_wrap(x - p1, LIFT_L1)) <= err_1) & (np.abs(_wrap(x - p2, LIFT_L2)) <= err_2)
    xs = x[mask]
    if xs.size == 0:
        return 0.0
    if xs.max() - xs.min() <= 0.5 * (LIFT_L1 + LIFT_L2) / 2:
        return xs.mean()
    if np.all(xs > 0):
        return LIFT_L1
    if np.all(xs < 0):
        return -LIFT_L1
    return 0.0


class TestPhaseUnwrapperInterval(unittest.TestCase):
    '''Tests of the interval-overlap mode (interval_unwrap=True).'''

    def _unwrapper(self, err, target_device_idx=-1, **kwargs):
        return PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2, interval_unwrap=True,
                              err_1=err, err_2=err, target_device_idx=target_device_idx,
                              **kwargs)

    def _single(self, u, p1, p2):
        '''Interval estimate for one segment, as a host float.'''
        return float(u._interval_unwrap(u.dtype(p1), u.dtype(p2))[0])

    def _run_once(self, u, p1_value, p2_value, target_device_idx, xp):
        return TestPhaseUnwrapper._run_once(self, u, p1_value, p2_value, target_device_idx, xp)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2, interval_unwrap=True)
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2, interval_unwrap=True, err_1=50.0)
        with self.assertRaises(ValueError):
            PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2, interval_unwrap=True,
                           err_1=-1.0, err_2=50.0)

    def test_default_is_not_interval(self):
        u = PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2)
        self.assertFalse(u.interval_unwrap)

    def test_search_range_covers_max_capture(self):
        u = self._unwrapper(100.0)
        self.assertAlmostEqual(u.max_capture, 5375.0, places=6)
        self.assertGreaterEqual(u.max_k_int * LIFT_L1, u.max_capture + LIFT_L1)
        self.assertGreaterEqual(u.max_m_int * LIFT_L2, u.max_capture + LIFT_L2)

    # ------------------------------------------------------------------ #
    # Decision rules
    # ------------------------------------------------------------------ #
    def test_noiseless_piston_recovered(self):
        # Exact estimates: the overlap is centred on the true piston, which is
        # recovered also when it is not a multiple of lambda_1.
        u = self._unwrapper(100.0)
        for true in [0.0, 37.5, -620.0, 1505.0, -3010.0, 4000.7, -4515.0]:
            est = self._single(u, _wrap(true, LIFT_L1), _wrap(true, LIFT_L2))
            self.assertAlmostEqual(est, true, places=2, msg=f'true piston {true}')

    def test_unique_overlap_returns_its_centre(self):
        # p1 = 20 +/- 75 and p2 = 60 +/- 75 overlap on [-15, 95]: centre 40
        u = self._unwrapper(75.0)
        self.assertAlmostEqual(self._single(u, 20.0, 60.0), 40.0, places=4)

    def test_ambiguous_mixed_signs_returns_zero(self):
        # Overlaps at k = 0 ([-100, 20], containing 0) and at k = -1 (around -1668):
        # mixed signs -> no correction. (Regression: comparing only the overlap
        # centres, both negative, returned -lambda_1 here.)
        u = self._unwrapper(100.0)
        self.assertEqual(self._single(u, -80.0, 0.0), 0.0)

    def test_ambiguous_same_sign_returns_one_lambda_1_step(self):
        # Overlaps at k = 0 ([50, 100]) and at k = +1 ([1650, 1755]), all positive
        u = self._unwrapper(100.0)
        self.assertEqual(self._single(u, 150.0, 0.0), LIFT_L1)
        self.assertEqual(self._single(u, -150.0, 0.0), -LIFT_L1)

    def test_no_overlap_returns_zero(self):
        u = self._unwrapper(10.0)
        self.assertEqual(self._single(u, 0.0, 300.0), 0.0)

    def test_overlaps_outside_max_capture_are_ignored(self):
        true = 4545.0
        p1, p2 = _wrap(true, LIFT_L1), _wrap(true, LIFT_L2)
        self.assertAlmostEqual(self._single(self._unwrapper(100.0), p1, p2), true, places=2)
        clipped = self._single(self._unwrapper(100.0, max_capture=3500.0), p1, p2)
        self.assertLessEqual(abs(clipped), 3500.0)
        self.assertNotAlmostEqual(clipped, true, places=0)

    def test_margins_larger_than_wavelength_difference(self):
        # A lambda_1 error reads -245 nm at lambda_2: with err_1 + err_2 <= 245 nm it
        # is corrected, above that the k = 0 and k = 1 solutions always overlap
        # with mixed signs and it is never corrected.
        p1, p2 = 0.0, _wrap(LIFT_L1, LIFT_L2)
        self.assertEqual(self._single(self._unwrapper(120.0), p1, p2), LIFT_L1)
        self.assertEqual(self._single(self._unwrapper(130.0), p1, p2), 0.0)

    def test_matches_brute_force_reference(self):
        # Random noisy estimates against a grid implementation of the same rules
        u = self._unwrapper(100.0)
        rng = np.random.default_rng(1234)
        for _ in range(200):
            true = rng.uniform(-4500.0, 4500.0)
            p1 = _wrap(true + rng.normal(0.0, 60.0), LIFT_L1)
            p2 = _wrap(true + rng.normal(0.0, 60.0), LIFT_L2)
            ref = _interval_reference(p1, p2, 100.0, 100.0, u.max_capture)
            est = self._single(u, p1, p2)
            self.assertAlmostEqual(est, ref, delta=0.5, msg=f'p1={p1}, p2={p2}')

    def test_two_stage_flag_is_ignored(self):
        p1, p2 = -80.0, 0.0
        a = self._single(self._unwrapper(100.0, two_stage_enabled=True), p1, p2)
        b = self._single(self._unwrapper(100.0, two_stage_enabled=False), p1, p2)
        self.assertEqual(a, b)

    # ------------------------------------------------------------------ #
    # Through the processing object
    # ------------------------------------------------------------------ #
    @cpu_and_gpu
    def test_trigger_vector_of_segments(self, target_device_idx, xp):
        # One segment per decision rule: unique (continuous), unique (k = 3),
        # ambiguous with mixed signs, ambiguous with the same sign
        p1_value = [20.0, 30.0, -80.0, 150.0]
        p2_value = [60.0, _wrap(4545.0, LIFT_L2), 0.0, 0.0]
        u = PhaseUnwrapper(lambda_1=LIFT_L1, lambda_2=LIFT_L2, interval_unwrap=True,
                           err_1=100.0, err_2=100.0, target_device_idx=target_device_idx)
        out_pistons, out_pistonsU = self._run_once(u, p1_value, p2_value, target_device_idx, xp)
        np.testing.assert_allclose(out_pistons, p1_value, atol=1e-3)
        np.testing.assert_allclose(out_pistonsU[:2], [40.0, 4545.0], atol=1e-2)
        self.assertEqual(out_pistonsU[2], 0.0)
        self.assertEqual(out_pistonsU[3], LIFT_L1)


if __name__ == '__main__':
    unittest.main()
