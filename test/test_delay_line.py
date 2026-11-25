"""
Tests of DelayLine, the pure time delay of a value.

Expected physics
----------------
Closed loop without optics (step -> residual -> sensor of N steps -> Integrator(gain 1,
delay 0) -> DelayLine(tau) -> ':-1' feedback). The lag from the first sensor output that
sees the step to the first change of the residual is

    lag = tau + 1 time step      (the ':-1' adds one step)

exactly, whatever the sensor frame N. With tau = 0 the DelayLine is transparent. The
DelayLine output changes only when a value is released, so a frame-refreshed sensor
(N = 4) gives one refresh per frame, not one per step.

Fractional delay: delay_s need not be a multiple of the time step. A value produced at
t is active from t + delay_s; if that falls inside step n (r = fraction of the step still
with the previous value) the output of step n is r*previous + (1-r)*new, from step n+1 the
new value alone, and the generation time is refreshed at both steps. Expectations:
  - single rate (N = 1): the filter's fractional delay out[n] = r*c[n-k-1] + (1-r)*c[n-k]
    (delay = k + r) is the step mean of a command delayed by k + r steps, so
    Integrator(delay=1.75) == Integrator(delay=0) + DelayLine(1.75 steps), also for 0.5;
  - N = 4, delay 7.5 steps: the residual first changes at lag 8 steps by half of the full
    first correction, and has the full correction from lag 9 (':-1' adds one step);
  - unit: input every 4 steps, delay 2.5 -> 0.5*prev + 0.5*new, then new, then held;
    input every step, delay 1.25 -> out[n] = 0.25*x[n-2] + 0.75*x[n-1].

Direct unit tests: an input refreshed at every step with delay of 3 steps gives
output(t) = input(t - 3); a negative delay raises ValueError.
"""
import os
import shutil
import tempfile
import unittest

import specula
specula.init(-1)  # CPU only, no optics in these loops

import numpy as np
import yaml
from astropy.io import fits

from specula.simul import Simul
from specula.base_value import BaseValue
from iteras.data_objects.sim_params import SimParams
from iteras.processing_objects.delay_line import DelayLine

TIME_STEP = 0.001   # [s]
STEP_TIME = 0.008   # [s] the unit step starts at the beginning of a sensor frame for N = 1, 2, 4, 8
TOTAL_TIME = 0.060  # [s]
THRESHOLD = 1e-6
GAIN = 1.0          # integrator gain of the single-rate fractional-delay comparison


def _read(path):
    """Return (times [s], first column of the data) of a DataStore fits file."""
    with fits.open(path) as h:
        times = np.asarray(h['TIMES'].data, dtype=float).ravel()
        values = np.asarray(h[0].data, dtype=float).reshape(len(times), -1)[:, 0]
    times = times * (1e-9 if times.max() > 1e3 else 1.0)  # ns ticks or seconds
    return times, values


class TestDelayLineLoop(unittest.TestCase):
    """Closed loop with a DelayLine after the Integrator."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, n_frame, tau_steps, use_delay_line=True, int_delay=0, int_gain=1.0):
        """Run the loop. Return (lag in steps, residual (t, v), out_value (t, v))."""
        if use_delay_line:
            feedback = 'latency.out_value:-1'
        else:
            feedback = 'control.out_comm:-1'
        params = {
            'main': {'class': 'SimulParams', 'root_dir': self.tmp, 'time_step': TIME_STEP,
                     'total_time': TOTAL_TIME},
            'step': {'class': 'ScheduleGenerator', 'scheduled_values': [[0.0], [1.0]],
                     'scheduled_times': [STEP_TIME], 'modes_per_group': 1},
            'residual': {'class': 'BaseOperation', 'sub': True,
                         'inputs': {'in_value1': 'step.output', 'in_value2': feedback}},
            'sensor': {'class': 'WindowedIntegration', 'simul_params_ref': 'main', 'n_elem': 1,
                       'dt': n_frame * TIME_STEP, 'update_time_on_dt': True,
                       'inputs': {'input': 'residual.out_value'}},
            'control': {'class': 'Integrator', 'int_gain': [int_gain], 'delay': int_delay,
                        'inputs': {'delta_comm': 'sensor.output'}},
            'data_store': {'class': 'DataStore', 'store_dir': self.tmp, 'create_tn': False,
                           'inputs': {'input_list': ['res-residual.out_value',
                                                     'meas-sensor.output']}},
        }
        if use_delay_line:
            params['latency'] = {'class': 'DelayLine', 'simul_params_ref': 'main',
                                 'delay_s': tau_steps * TIME_STEP,
                                 'inputs': {'in_value': 'control.out_comm'}}
            params['data_store']['inputs']['input_list'].append('lat-latency.out_value')
        path = os.path.join(self.tmp, 'params.yml')
        with open(path, 'w') as f:
            yaml.dump(params, f)
        Simul(path, accelerate=False, speed_report=False).run()

        t_meas, meas = _read(os.path.join(self.tmp, 'meas.fits'))
        t_res, res = _read(os.path.join(self.tmp, 'res.fits'))
        lat = _read(os.path.join(self.tmp, 'lat.fits')) if use_delay_line else None

        t_first_meas = t_meas[np.flatnonzero(np.abs(meas) > THRESHOLD)[0]]
        after = (t_res > t_first_meas) & (np.abs(res - 1.0) > THRESHOLD)
        t_corr = t_res[np.flatnonzero(after)[0]]
        lag = int(round((t_corr - t_first_meas) / TIME_STEP))
        return lag, (t_res, res), lat

    def test_lag_is_tau_plus_one_step(self):
        """(N, tau) -> lag = tau + 1, also when it is not a multiple of the frame."""
        for n_frame, tau, expected in [(4, 7, 8), (4, 5, 6), (1, 1, 2), (4, 0, 1)]:
            with self.subTest(n_frame=n_frame, tau=tau):
                self.assertEqual(self._run(n_frame, tau)[0], expected)

    def test_n4_tau7_is_two_frames(self):
        """N=4, tau=7 steps: lag 8 steps = exactly 2 frames."""
        self.assertEqual(self._run(4, 7)[0], 2 * 4)

    def test_n1_tau1_same_as_integrator_delay_1(self):
        """Classic 2-frame loop: DelayLine(1 step) = Integrator delay 1 when N = 1."""
        _, (_, res_dl), _ = self._run(1, 1)
        lag_ref, (_, res_ref), _ = self._run(1, 0, use_delay_line=False, int_delay=1)
        self.assertEqual(lag_ref, 2)
        np.testing.assert_allclose(res_dl, res_ref, atol=1e-12)

    def test_single_rate_matches_filter_fractional_delay(self):
        """N=1: Integrator(delay=d) == Integrator(delay=0) + DelayLine(d steps), d < 1 and > 1."""
        for d in (0.5, 1.75):
            with self.subTest(delay=d):
                _, (t_dl, res_dl), _ = self._run(1, d, int_gain=GAIN)
                _, (t_ref, res_ref), _ = self._run(1, 0, use_delay_line=False, int_delay=d,
                                                   int_gain=GAIN)
                self.assertTrue(np.all(np.isfinite(res_ref)))
                np.testing.assert_allclose(t_dl, t_ref)
                np.testing.assert_allclose(res_dl, res_ref, atol=1e-12)

    def test_n4_fractional_delay_half_correction_then_full(self):
        """N=4, delay 7.5 steps: first change at lag 8 is half of the full correction,
        full from lag 9 (until the second command, 4 steps later)."""
        lag, (t_res, res), _ = self._run(4, 7.5)
        self.assertEqual(lag, 8)
        i0 = np.flatnonzero((t_res >= STEP_TIME) & (np.abs(res - 1.0) > THRESHOLD))[0]
        np.testing.assert_allclose(np.diff(t_res[i0:i0 + 4]), TIME_STEP, atol=1e-9)
        half, full = 1.0 - res[i0], 1.0 - res[i0 + 1]
        self.assertAlmostEqual(half / full, 0.5, delta=1e-12 * abs(full))
        np.testing.assert_allclose(res[i0 + 1:i0 + 4], res[i0 + 1], atol=1e-12)

    def test_tau0_is_transparent(self):
        """tau = 0: same residual time series as the loop without DelayLine."""
        _, (t_dl, res_dl), _ = self._run(4, 0)
        _, (t_ref, res_ref), _ = self._run(4, 0, use_delay_line=False)
        np.testing.assert_allclose(t_dl, t_ref)
        np.testing.assert_allclose(res_dl, res_ref, atol=1e-12)

    def test_output_refreshes_once_per_frame(self):
        """The DataStore saves a sample only when the generation time of the stored value
        is the current time, so the samples of latency.out_value are the refreshes.
        With N = 4 they are spaced by exactly one frame (4 steps), and there are
        about TOTAL_TIME/frame of them, not one per step."""
        for tau in (0, 5, 7):
            with self.subTest(tau=tau):
                _, _, (t_lat, _) = self._run(4, tau)
                np.testing.assert_allclose(np.diff(t_lat), 4 * TIME_STEP, atol=1e-9)
                n_steps = int(round(TOTAL_TIME / TIME_STEP))
                self.assertLessEqual(len(t_lat), n_steps // 4)
                self.assertGreaterEqual(len(t_lat), n_steps // 4 - 2 - tau // 4)


class TestDelayLineUnit(unittest.TestCase):
    """DelayLine driven by hand, without Simul."""

    def _make(self, delay_steps):
        dl = DelayLine(simul_params=SimParams(time_step=TIME_STEP),
                       delay_s=delay_steps * TIME_STEP, target_device_idx=-1)
        self.t = dl.seconds_to_t(TIME_STEP)  # ticks per step
        self.src = BaseValue(value=np.zeros(1, dtype=dl.dtype), target_device_idx=-1)
        self.src.generation_time = -1
        dl.inputs['in_value'].set(self.src)
        dl.setup()
        return dl

    def _step(self, dl, k, value=None):
        """Run step k (time k * dt). If value is not None the input is refreshed to it."""
        t = k * self.t
        if value is not None:
            self.src.value[0] = value
            self.src.generation_time = t
        dl.check_ready(t)
        dl.trigger()
        dl.post_trigger()
        return float(dl.output.value[0]), dl.output.generation_time

    def test_input_every_step_delay_3(self):
        """Two or more inputs pending at once: output(t) = input(t - 3)."""
        dl = self._make(3)
        for k in range(1, 12):
            out, gen = self._step(dl, k, value=10.0 * k)
            if k < 4:
                self.assertEqual(out, 0.0)           # nothing released yet
            else:
                self.assertEqual(out, 10.0 * (k - 3))
                self.assertEqual(gen, k * self.t)    # refreshed at every step

    def test_slow_input_held_and_generation_time_changes_only_on_release(self):
        """Input every 4 steps (k=1,5,9), delay 3: released at k=4,8,12 and held in between."""
        dl = self._make(3)
        gens = {}
        outs = {}
        for k in range(1, 13):
            fresh = 10.0 * k if (k - 1) % 4 == 0 else None
            outs[k], gens[k] = self._step(dl, k, value=fresh)
        for k, expected in [(4, 10.0), (5, 10.0), (7, 10.0), (8, 50.0), (11, 50.0), (12, 90.0)]:
            self.assertEqual(outs[k], expected, k)
        for k in (4, 8, 12):
            self.assertEqual(gens[k], k * self.t)
        for k, last_release in [(5, 4), (6, 4), (7, 4), (9, 8), (10, 8), (11, 8)]:
            self.assertEqual(gens[k], last_release * self.t, k)

    def test_zero_delay_is_pass_through(self):
        dl = self._make(0)
        for k in range(1, 5):
            out, gen = self._step(dl, k, value=float(k))
            self.assertEqual((out, gen), (float(k), k * self.t))

    def test_negative_delay_raises(self):
        with self.assertRaises(ValueError):
            DelayLine(simul_params=SimParams(time_step=TIME_STEP), delay_s=-TIME_STEP)

    def test_fractional_delay_slow_input(self):
        """Input every 4 steps (k=1,5,9), delay 2.5: active at k+2.5, so at step k+2 the
        output is 0.5*previous + 0.5*new, at k+3 the new value, then held. The generation
        time changes exactly at those two steps per value."""
        dl = self._make(2.5)
        outs, gens = {}, {}
        for k in range(1, 13):
            fresh = 10.0 * k if (k - 1) % 4 == 0 else None
            outs[k], gens[k] = self._step(dl, k, value=fresh)
        expected = {1: 0, 2: 0, 3: 5, 4: 10, 5: 10, 6: 10, 7: 30, 8: 50, 9: 50, 10: 50,
                    11: 70, 12: 90}
        for k, v in expected.items():
            self.assertAlmostEqual(outs[k], v, msg=k)
        g0 = gens[1]   # initial generation time, unchanged until the first activation
        last = None
        for k in range(1, 13):
            if k % 4 in (3, 0):   # the two steps of each activation
                last = k
            self.assertEqual(gens[k], g0 if last is None else last * self.t, k)

    def test_fractional_delay_every_step(self):
        """Input every step, delay 1.25: out[n] = 0.25*x[n-2] + 0.75*x[n-1] (x[0] = 0)."""
        dl = self._make(1.25)
        x = {0: 0.0}
        for k in range(1, 12):
            x[k] = 10.0 * k
            out, gen = self._step(dl, k, value=x[k])
            if k == 1:
                self.assertEqual(out, 0.0)
            else:
                self.assertAlmostEqual(out, 0.25 * x[k - 2] + 0.75 * x[k - 1])
                self.assertEqual(gen, k * self.t)

if __name__ == '__main__':
    unittest.main()
