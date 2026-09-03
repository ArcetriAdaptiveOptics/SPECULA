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


if __name__ == '__main__':
    unittest.main()
