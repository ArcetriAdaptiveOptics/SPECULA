"""
Tests of BaseFilter.check_ready: a fractional filter delay needs a filter triggered at every step.

Expected physics
----------------
The filter delay is counted in filter triggers. A fractional delay k + r gives
out[n] = r*c[n-k-1] + (1-r)*c[n-k], the step mean of a command delayed by k + r steps, only
if the filter is triggered at every step. With a sensor slower than the time step (N > 1)
it would mix two sensor frames, so it must be rejected:
  - fractional delay and, after the first trigger, a step with the input not refreshed:
    ValueError (message mentions DelayLine, the object for a delay in seconds);
  - integer delay: never raises, whatever the sensor frame;
  - before the first trigger nothing is checked (sensor with a late start_time).
Closed loops: N=4 with delay 1.5 raises; N=4 with delay 1 and N=1 with delay 1.5 run.
Unit tests: input refreshed at steps 1, 2, 3 and not at step 4: fractional delay raises at
check_ready(4), integer delay returns False.
"""
import os
import shutil
import tempfile
import unittest

import specula
specula.init(-1)  # CPU only, no optics in these loops

import numpy as np
import yaml

from specula.simul import Simul
from specula.base_value import BaseValue
from iteras.processing_objects.integrator import Integrator

TIME_STEP = 0.001   # [s]
STEP_TIME = 0.008   # [s]
TOTAL_TIME = 0.030  # [s]


class TestFractionalDelayLoop(unittest.TestCase):
    """step -> residual -> sensor of N steps -> Integrator(delay) -> ':-1' feedback."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, n_frame, delay, sensor_start_time=0.0):
        params = {
            'main': {'class': 'SimulParams', 'root_dir': self.tmp, 'time_step': TIME_STEP,
                     'total_time': TOTAL_TIME},
            'step': {'class': 'ScheduleGenerator', 'scheduled_values': [[0.0], [1.0]],
                     'scheduled_times': [STEP_TIME], 'modes_per_group': 1},
            'residual': {'class': 'BaseOperation', 'sub': True,
                         'inputs': {'in_value1': 'step.output',
                                    'in_value2': 'control.out_comm:-1'}},
            'sensor': {'class': 'WindowedIntegration', 'simul_params_ref': 'main', 'n_elem': 1,
                       'dt': n_frame * TIME_STEP, 'start_time': sensor_start_time,
                       'update_time_on_dt': True,
                       'inputs': {'input': 'residual.out_value'}},
            'control': {'class': 'Integrator', 'int_gain': [1.0], 'delay': delay,
                        'inputs': {'delta_comm': 'sensor.output'}},
        }
        path = os.path.join(self.tmp, 'params.yml')
        with open(path, 'w') as f:
            yaml.dump(params, f)
        Simul(path, accelerate=False, speed_report=False).run()

    def test_slow_sensor_fractional_delay_raises(self):
        with self.assertRaisesRegex(ValueError, 'DelayLine'):
            self._run(4, 1.5)

    def test_slow_sensor_integer_delay_runs(self):
        self._run(4, 1)

    def test_single_rate_fractional_delay_runs(self):
        self._run(1, 1.5)

    def test_late_sensor_start_is_not_checked_before_first_trigger(self):
        """N=1, delay 1.5, sensor silent until 5 steps: the filter is not ready before its
        first trigger (nothing checked), then it is triggered at every step."""
        self._run(1, 1.5, sensor_start_time=5 * TIME_STEP)


class TestFractionalDelayUnit(unittest.TestCase):
    """Integrator driven by hand, without Simul."""

    def _make(self, delay):
        filt = Integrator(int_gain=[1.0], delay=delay, target_device_idx=-1)
        self.t = filt.seconds_to_t(TIME_STEP)  # ticks per step
        self.src = BaseValue(value=np.zeros(1, dtype=filt.dtype), target_device_idx=-1)
        self.src.generation_time = -1
        filt.inputs['delta_comm'].set(self.src)
        filt.setup()
        return filt

    def _step(self, filt, k, refreshed):
        """Run step k (time k * dt); return the check_ready result."""
        t = k * self.t
        if refreshed:
            self.src.value[0] = float(k)
            self.src.generation_time = t
        ready = filt.check_ready(t)
        if ready:
            filt.trigger()
            filt.post_trigger()
        return ready

    def test_fractional_delay_raises_when_input_not_refreshed(self):
        filt = self._make(1.5)
        for k in (1, 2, 3):
            self.assertTrue(self._step(filt, k, refreshed=True))
        with self.assertRaisesRegex(ValueError, 'DelayLine'):
            self._step(filt, 4, refreshed=False)

    def test_integer_delay_does_not_raise(self):
        filt = self._make(1)
        for k in (1, 2, 3):
            self.assertTrue(self._step(filt, k, refreshed=True))
        self.assertFalse(self._step(filt, 4, refreshed=False))

    def test_fractional_delay_not_checked_before_first_trigger(self):
        filt = self._make(1.5)
        for k in (0, 1, 2):
            self.assertFalse(self._step(filt, k, refreshed=False))
        self.assertTrue(self._step(filt, 3, refreshed=True))

    def test_fractional_delay_refreshed_every_step_never_raises(self):
        filt = self._make(1.5)
        for k in range(1, 10):
            self.assertTrue(self._step(filt, k, refreshed=True))


if __name__ == '__main__':
    unittest.main()
