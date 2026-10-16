import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.processing_objects.avc import AVC

from test.specula_testlib import cpu_and_gpu

# Synthetic single-tone loop used by the closed-loop tests: true plant at
# the vibration frequency and disturbance phasor
P_STAR = 0.555 * np.exp(1j * np.radians(119.0))
LAM_STAR = 120.0 * np.exp(0.7j)


def _make_avc(target_device_idx, p_hat=P_STAR, gx=30.0, gomega=0.0, **kwargs):
    return AVC(simul_params=SimulParams(time_step=0.001), freq=47.0,
               gx=gx, gomega=gomega, k=1.0,
               plant_re=p_hat.real, plant_im=p_hat.imag,
               target_device_idx=target_device_idx, **kwargs)


def _run_closed_loop(obj, n_steps, p_star, lam_star, target_device_idx, xp):
    """Close a synthetic single-tone loop around ``obj`` through the true
    plant ``p_star`` and return the residual history.

    The residual is the phasor ``Z = theta * conj(p_star) + lam_star``
    projected on the true phase, with ``theta`` the command of the previous
    step.
    """
    measurement = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
    obj.inputs['in_measurement'].set(measurement)
    obj.setup()

    ts = float(obj.T)
    omega_true = 2 * np.pi * 47.0
    phase = 0.0
    residuals = np.zeros(n_steps)
    comm = 0.0

    for i in range(n_steps):
        z = comm * np.conj(p_star) + lam_star
        y = z.real * np.cos(phase) + z.imag * np.sin(phase)
        residuals[i] = y

        t = obj.seconds_to_t((i + 1) * ts)
        measurement.value[:] = xp.asarray([y])
        measurement.generation_time = t
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()

        # the emitted command, as a phasor in the object's own frame
        comm = -_plant(obj) * _estimate(obj) / abs(_plant(obj)) ** 2

        phase += omega_true * ts

    return residuals


def _plant(obj):
    x1, x2 = cpuArray(obj.outputs['out_state'].value)[0, :2]
    return complex(x1, x2)


def _estimate(obj):
    x3, x4 = cpuArray(obj.outputs['out_state'].value)[0, 2:]
    return complex(x3, x4)


class TestAVC(unittest.TestCase):

    @cpu_and_gpu
    def test_construction_and_shapes(self, target_device_idx, xp):
        """One output per channel for out_comm, one per instance for
        out_freq and out_state."""
        simul_params = SimulParams(time_step=0.001)
        for freq, n_channels, n_instances in [(47.0, 1, 1),
                                              ([10.0, 20.0, 30.0], 3, 3),
                                              ([[47.0, 113.0]], 1, 2),
                                              ([[47.0, 113.0], [30.0]], 2, 3)]:
            with self.subTest(freq=freq):
                avc = AVC(simul_params=simul_params, freq=freq, gx=0.05, gomega=0.001, k=1.0,
                          target_device_idx=target_device_idx)

                self.assertEqual(avc.outputs['out_comm'].value.shape, (n_channels,))
                self.assertEqual(avc.outputs['out_freq'].value.shape, (n_instances,))
                self.assertEqual(avc.outputs['out_state'].value.shape, (n_instances, 4))
                np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_channels))

    @cpu_and_gpu
    def test_empty_channel_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        for freq in ([], [[47.0], []]):
            with self.subTest(freq=freq):
                with self.assertRaises(ValueError):
                    AVC(simul_params=simul_params, freq=freq, gx=0.05, gomega=0.001, k=1.0,
                        target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_parameter_forms_agree(self, target_device_idx, xp):
        """A per-instance parameter can be a flat list in the order of freq
        or a list of lists shaped like freq; a scalar applies to all."""
        simul_params = SimulParams(time_step=0.001)
        freq = [[47.0, 113.0], [30.0]]
        for gx, expected in [([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
                             ([[1.0, 2.0], [3.0]], [1.0, 2.0, 3.0]),
                             (2.0, [2.0, 2.0, 2.0])]:
            with self.subTest(gx=gx):
                avc = AVC(simul_params=simul_params, freq=freq, gx=gx, gomega=0.001, k=1.0,
                          target_device_idx=target_device_idx)
                np.testing.assert_allclose(cpuArray(avc.gx), expected)

    @cpu_and_gpu
    def test_mismatched_parameter_shape_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        for gx in ([1.0, 2.0], [[1.0], [2.0, 3.0]]):
            with self.subTest(gx=gx):
                with self.assertRaises(ValueError):
                    AVC(simul_params=simul_params, freq=[[47.0, 113.0], [30.0]],
                        gx=gx, gomega=0.001, k=1.0,
                        target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_zero_plant_raises(self, target_device_idx, xp):
        """The calibration sits in the denominator of the fixed map."""
        with self.assertRaises(ValueError):
            _make_avc(target_device_idx, p_hat=0j)

    @cpu_and_gpu
    def test_both_calibration_forms_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVC(simul_params=simul_params, freq=47.0,
                gx=3.0, gomega=0.2, k=1.0,
                plant_re=1.0, plant_gain=1.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_wrong_input_length_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        avc = AVC(simul_params=simul_params, freq=[25.0, 55.0],
                  gx=0.05, gomega=0.001, k=1.0,
                  target_device_idx=target_device_idx)
        measurement = BaseValue(value=xp.zeros(3), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()

        t = avc.seconds_to_t(0.001)
        measurement.generation_time = t
        with self.assertRaises(ValueError):
            avc.check_ready(t)

    @cpu_and_gpu
    def test_channels_feed_and_sum_their_instances(self, target_device_idx, xp):
        """freq = [[f1, f2], [f3]]: instances 1 and 2 are fed channel 0,
        instance 3 channel 1, and out_comm sums the corrections per channel.

        With the default plant 1 + 0j and gomega = 0, the first step gives
        x3 = gx * T * y, x4 = 0, alpha = 2 pi f T, and the correction
        u = -cos(alpha) * x3, emitted at the second step."""
        T = 0.001
        freq = np.array([47.0, 113.0, 30.0])
        gx = np.array([1.0, 2.0, 3.0])
        y = np.array([0.3, -0.2])
        avc = AVC(simul_params=SimulParams(time_step=T), freq=[[47.0, 113.0], [30.0]],
                  gx=gx.tolist(), gomega=0.0, k=1.0,
                  target_device_idx=target_device_idx)
        measurement = BaseValue(value=xp.asarray(y), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()

        for step in (1, 2):
            t = avc.seconds_to_t(step * T)
            measurement.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()
            if step == 1:
                x3_first = cpuArray(avc.outputs['out_state'].value)[:, 2].copy()

        y_instances = y[[0, 0, 1]]
        np.testing.assert_allclose(x3_first, gx * T * y_instances, rtol=1e-6)
        u = -np.cos(2 * np.pi * freq * T) * x3_first
        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value),
                                   [u[0] + u[1], u[2]], rtol=1e-5)

    @cpu_and_gpu
    def test_cartesian_and_polar_calibration_give_the_same_map(self, target_device_idx, xp):
        """theta = -Lambda_hat / conj(P_hat): with P_hat = rho exp(i psi), the
        map (x3, x4) -> (thetac, thetas) is a rotation by psi scaled by -1/rho."""
        simul_params = SimulParams(time_step=0.001)
        rho, psi = 0.555, np.radians(119.0)
        expected = {'m11': -np.cos(psi) / rho, 'm12': np.sin(psi) / rho,
                    'm21': -np.sin(psi) / rho, 'm22': -np.cos(psi) / rho}

        cartesian = AVC(simul_params=simul_params, freq=47.0,
                        gx=3.0, gomega=0.2, k=1.0,
                        plant_re=rho * np.cos(psi), plant_im=rho * np.sin(psi),
                        target_device_idx=target_device_idx)
        polar = AVC(simul_params=simul_params, freq=47.0,
                    gx=3.0, gomega=0.2, k=1.0,
                    plant_gain=rho, plant_phase=np.degrees(psi),
                    target_device_idx=target_device_idx)

        for avc in (cartesian, polar):
            for name, value in expected.items():
                np.testing.assert_allclose(cpuArray(getattr(avc, name)), [value], rtol=1e-6)

    @cpu_and_gpu
    def test_first_trigger_outputs_zero_correction(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        n_channels = 2
        avc = AVC(simul_params=simul_params, freq=[25.0, 55.0],
                  gx=0.05, gomega=0.001, k=1.0,
                  target_device_idx=target_device_idx)

        measurement = BaseValue(value=xp.array([0.3, -0.2]), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(measurement)
        avc.setup()

        t = avc.seconds_to_t(0.001)
        measurement.generation_time = t
        avc.check_ready(t)
        avc.trigger()
        avc.post_trigger()

        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_channels))

    @cpu_and_gpu
    def test_output_precision(self, target_device_idx, xp):
        """Outputs follow the requested precision (0=double, 1=single),
        regardless of the input dtype."""
        for precision in (0, 1):
            expected_dtype = np.float64 if precision == 0 else np.float32
            with self.subTest(precision=precision):
                avc = _make_avc(target_device_idx, precision=precision)
                measurement = BaseValue(value=xp.asarray([0.3], dtype=xp.float64),
                                        target_device_idx=target_device_idx, precision=0)
                avc.inputs['in_measurement'].set(measurement)
                avc.setup()
                t = avc.seconds_to_t(0.001)
                measurement.generation_time = t
                avc.check_ready(t)
                avc.trigger()
                avc.post_trigger()

                for name in ('out_comm', 'out_freq', 'out_state'):
                    self.assertEqual(cpuArray(avc.outputs[name].value).dtype, expected_dtype)

    @cpu_and_gpu
    def test_plant_state_never_changes(self, target_device_idx, xp):
        """plant_re, plant_im are a calibration, not a state."""
        avc = _make_avc(target_device_idx, gx=3.0, gomega=0.2)
        _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

        state = cpuArray(avc.outputs['out_state'].value)[0]
        self.assertEqual(state[0], cpuArray(avc.plant_re)[0])
        self.assertEqual(state[1], cpuArray(avc.plant_im)[0])

    @cpu_and_gpu
    def test_cancels_tone_with_exact_calibration(self, target_device_idx, xp):
        avc = _make_avc(target_device_idx)
        res = _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

        self.assertLess(res[-200:].std(), 0.01 * res[:200].std())

    @cpu_and_gpu
    def test_converged_estimate_matches_analytic(self, target_device_idx, xp):
        """At equilibrium the residual is zero: theta = -Lambda / conj(P_star)
        with theta = -Lambda_hat / conj(P_hat), so
        Lambda_hat = conj(P_hat / P_star) * Lambda (Lambda itself if the
        calibration is exact)."""
        for p_hat in (P_STAR, 2.0 * P_STAR * np.exp(1j * np.radians(40.0))):
            with self.subTest(p_hat=p_hat):
                avc = _make_avc(target_device_idx, p_hat=p_hat)
                _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

                expected = np.conj(p_hat / P_STAR) * LAM_STAR
                np.testing.assert_allclose(_estimate(avc), expected, rtol=1e-4)

    @cpu_and_gpu
    def test_estimate_scales_linearly_with_disturbance(self, target_device_idx, xp):
        estimates = []
        for scale in (1.0, 2.0, 5.0):
            avc = _make_avc(target_device_idx)
            _run_closed_loop(avc, 2000, P_STAR, scale * LAM_STAR, target_device_idx, xp)
            estimates.append(_estimate(avc))

        np.testing.assert_allclose(estimates, np.array([1.0, 2.0, 5.0]) * estimates[0], rtol=1e-4)

    @cpu_and_gpu
    def test_correction_amplitude_independent_of_disturbance_phase(self, target_device_idx, xp):
        """The converged command has amplitude |Lambda| / |P_star| whatever
        the phase of the disturbance."""
        for phase in (0.0, 0.7, 2.5, -2.0):
            with self.subTest(phase=phase):
                avc = _make_avc(target_device_idx)
                lam = abs(LAM_STAR) * np.exp(1j * phase)
                _run_closed_loop(avc, 2000, P_STAR, lam, target_device_idx, xp)

                command = -_plant(avc) * _estimate(avc) / abs(_plant(avc)) ** 2
                self.assertAlmostEqual(abs(command), abs(LAM_STAR) / abs(P_STAR), delta=1e-2)

    @cpu_and_gpu
    def test_convergence_rate_follows_calibration_phase_error(self, target_device_idx, xp):
        """The residual envelope evolves as exp(-rate t), with
        rate = 0.5 * gx * Re(kappa) and kappa = conj(P_star / P_hat). With
        the exact magnitude and a phase error dphi, Re(kappa) = cos(dphi):
        convergence below 90 degrees, divergence above, and neither at
        90 degrees, the stability limit.

        The rate is measured between the residual rms around 0.5 s and
        around 1.5 s (40 ms windows, about two periods of the tone)."""
        gx = 30.0
        t1, t2, half_window = 0.5, 1.5, 20
        i1, i2 = int(t1 / 0.001), int(t2 / 0.001)
        for dphi in (0.0, 60.0, 85.0, 90.0, 95.0, 120.0):
            with self.subTest(dphi=dphi):
                p_hat = P_STAR * np.exp(1j * np.radians(dphi))
                avc = _make_avc(target_device_idx, p_hat=p_hat, gx=gx)
                res = _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

                rms1 = res[i1 - half_window:i1 + half_window].std()
                rms2 = res[i2 - half_window:i2 + half_window].std()
                rate = -np.log(rms2 / rms1) / (t2 - t1)
                expected = 0.5 * gx * np.cos(np.radians(dphi))
                if dphi == 90.0:
                    # the residual stays within +-5% over 1 s
                    self.assertLess(abs(rate), 0.05)
                else:
                    self.assertAlmostEqual(rate, expected, delta=0.05 * abs(expected))

    @cpu_and_gpu
    def test_magnitude_error_only_slows_convergence(self, target_device_idx, xp):
        """A calibration 5 times too large reduces the convergence rate by
        5 but keeps it positive."""
        ratios = []
        for scale in (1.0, 5.0):
            avc = _make_avc(target_device_idx, p_hat=scale * P_STAR)
            res = _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)
            ratios.append(res[-200:].std() / res[:200].std())

        self.assertLess(ratios[0], 0.01)
        self.assertLess(ratios[1], 0.5)
        self.assertGreater(ratios[1], ratios[0])

    @cpu_and_gpu
    def test_frequency_tracking_converges_on_pure_tone(self, target_device_idx, xp):
        """Open-loop PLL check: the estimate is pulled onto a pure tone
        from a wrong initial guess at the analytic rate.

        For y = A sin(phi), sin(alpha) * y averages to (A / 2) cos(alpha - phi),
        and near lock the phase error delta obeys
        delta'' + k a delta' + a delta = 0, with a = gomega * A. The
        frequency error decays as exp(lambda t), lambda being the slow root
        of lambda**2 + k a lambda + a = 0: -1.08 1/s here, so it shrinks
        by exp(1.5 lambda) = 0.197 between 1.0 s and 2.5 s.
        """
        T = 0.001
        true_freq, amp, gomega, k = 50.0, 0.7, 20.0, 1.0
        a = gomega * amp
        slow_root = (-k * a + np.sqrt((k * a)**2 - 4 * a)) / 2
        t1, t2 = 1.0, 2.5
        expected_ratio = np.exp(slow_root * (t2 - t1))
        # error averaged over 40 ms, i.e. 4 periods of the ripple at 2 * true_freq
        half_window = 20
        windows = {int(t / T): [] for t in (t1, t2)}

        simul_params = SimulParams(time_step=T)
        avc = AVC(simul_params=simul_params, freq=48.0,
                  gx=0.5, gomega=gomega, k=k,
                  target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        for it in range(5000):
            t_sec = (it + 1) * T
            t = avc.seconds_to_t(t_sec)
            meas.value[:] = xp.asarray([amp * np.sin(2 * np.pi * true_freq * t_sec)])
            meas.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()
            for center, errors in windows.items():
                if abs(it - center) < half_window:
                    freq = float(cpuArray(avc.outputs['out_freq'].value)[0])
                    errors.append(abs(freq - true_freq))

        err1, err2 = (np.mean(errors) for errors in windows.values())
        self.assertAlmostEqual(err2 / err1, expected_ratio, delta=0.02 * expected_ratio)

        final_freq = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(final_freq, true_freq, delta=0.1)

    @cpu_and_gpu
    def test_gomega_zero_holds_frequency(self, target_device_idx, xp):
        avc = _make_avc(target_device_idx)
        _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

        f_est = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(f_est, 47.0, places=4)


if __name__ == '__main__':
    unittest.main()
