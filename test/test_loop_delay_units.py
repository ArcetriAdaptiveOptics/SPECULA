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

TIME_STEP = 0.001   # [s]
STEP_TIME = 0.008   # [s] the unit step starts at the beginning of a sensor frame for N = 1, 2, 4, 8
TOTAL_TIME = 0.060  # [s]
THRESHOLD = 1e-6


def _read(path):
    """Return (times [s], first column of the data) of a DataStore fits file."""
    with fits.open(path) as h:
        times = np.asarray(h['TIMES'].data, dtype=float).ravel()
        values = np.asarray(h[0].data, dtype=float).reshape(len(times), -1)[:, 0]
    times = times * (1e-9 if times.max() > 1e3 else 1.0)  # ns ticks or seconds
    return times, values


class TestLoopDelayUnits(unittest.TestCase):
    """Closed loop without optics:
    step -> residual = step - command(:-1) -> sensor (frame of N steps) -> Integrator(gain 1, delay d).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _run(self, n_frame, delay, frame_refresh):
        """Run the loop and return (lag in time steps, n sensor samples, n command samples)."""
        params = {
            'main': {'class': 'SimulParams', 'root_dir': self.tmp, 'time_step': TIME_STEP,
                     'total_time': TOTAL_TIME},
            'step': {'class': 'ScheduleGenerator', 'scheduled_values': [[0.0], [1.0]],
                     'scheduled_times': [STEP_TIME], 'modes_per_group': 1},
            'residual': {'class': 'BaseOperation', 'sub': True,
                         'inputs': {'in_value1': 'step.output', 'in_value2': 'control.out_comm:-1'}},
            'sensor': {'class': 'WindowedIntegration', 'simul_params_ref': 'main', 'n_elem': 1,
                       'dt': n_frame * TIME_STEP, 'update_time_on_dt': frame_refresh,
                       'inputs': {'input': 'residual.out_value'}},
            'control': {'class': 'Integrator', 'int_gain': [1.0], 'delay': delay,
                        'inputs': {'delta_comm': 'sensor.output'}},
            'data_store': {'class': 'DataStore', 'store_dir': self.tmp, 'create_tn': False,
                           'inputs': {'input_list': ['res-residual.out_value',
                                                     'meas-sensor.output',
                                                     'comm-control.out_comm']}},
        }
        path = os.path.join(self.tmp, 'params.yml')
        with open(path, 'w') as f:
            yaml.dump(params, f)
        Simul(path, accelerate=False, speed_report=False).run()

        t_meas, meas = _read(os.path.join(self.tmp, 'meas.fits'))
        t_res, res = _read(os.path.join(self.tmp, 'res.fits'))
        t_comm, _ = _read(os.path.join(self.tmp, 'comm.fits'))

        # First non-zero sensor output = first measurement that sees the step
        t_first_meas = t_meas[np.flatnonzero(np.abs(meas) > THRESHOLD)[0]]
        # First change of the residual away from 1 (the correction) after that
        after = (t_res > t_first_meas) & (np.abs(res - 1.0) > THRESHOLD)
        t_corr = t_res[np.flatnonzero(after)[0]]
        lag = int(round((t_corr - t_first_meas) / TIME_STEP))
        return lag, len(t_meas), len(t_comm)

    def test_n1_d1_frame_refresh(self):
        """Sensor frame = 1 step, delay 1: lag = 1*1 + 1 = 2 steps."""
        self.assertEqual(self._run(1, 1, True)[0], 2)

    def test_n1_d2_frame_refresh(self):
        """Sensor frame = 1 step, delay 2: lag = 2*1 + 1 = 3 steps."""
        self.assertEqual(self._run(1, 2, True)[0], 3)

    def test_n4_d1_frame_refresh(self):
        """Delay counted in sensor frames (CCD-like): 1 frame + 1 step = 5 steps = 1.25 frames."""
        self.assertEqual(self._run(4, 1, True)[0], 5)

    def test_n4_d2_frame_refresh(self):
        """Delay counted in sensor frames: 2 frames + 1 step = 9 steps."""
        self.assertEqual(self._run(4, 2, True)[0], 9)

    def test_n4_d1_step_refresh(self):
        """Sensor output refreshed every step: the controller triggers every step, so
        delay 1 = 1 step, lag = 1 + 1 = 2 steps."""
        self.assertEqual(self._run(4, 1, False)[0], 2)

    def test_n4_d2_step_refresh(self):
        """Sensor output refreshed every step: delay 2 = 2 steps, lag = 2 + 1 = 3 steps."""
        self.assertEqual(self._run(4, 2, False)[0], 3)

    def test_general_formula(self):
        """lag = d * (N if the sensor output is refreshed per frame else 1) + 1, in time steps."""
        for n_frame, delay, frame_refresh in [(2, 1, True), (2, 3, True), (4, 3, False)]:
            with self.subTest(n_frame=n_frame, delay=delay, frame_refresh=frame_refresh):
                expected = delay * (n_frame if frame_refresh else 1) + 1
                self.assertEqual(self._run(n_frame, delay, frame_refresh)[0], expected)

    def test_controller_triggers_only_on_refreshed_input(self):
        """With a frame-refreshed sensor (N=4) the controller output has one sample per sensor
        frame (15 for 60 steps), not one per time step."""
        _, n_meas, n_comm = self._run(4, 1, True)
        self.assertEqual(n_meas, int(round(TOTAL_TIME / TIME_STEP)) // 4)
        self.assertEqual(n_comm, n_meas)

    def test_controller_triggers_every_step_with_step_refresh(self):
        """With a step-refreshed sensor the controller output has one sample per time step."""
        _, n_meas, n_comm = self._run(4, 1, False)
        self.assertEqual(n_meas, int(round(TOTAL_TIME / TIME_STEP)))
        self.assertEqual(n_comm, n_meas)


if __name__ == '__main__':
    unittest.main()
