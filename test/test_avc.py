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
    return AVC(simul_params=SimulParams(time_step=0.001), n_avc=1, freq=47.0,
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
        simul_params = SimulParams(time_step=0.001)
        n_avc = 3
        avc = AVC(simul_params=simul_params, n_avc=n_avc,
                  freq=[10.0, 20.0, 30.0], gx=0.05, gomega=0.001, k=1.0,
                  plant_re=[1.0, 2.0, 3.0], plant_im=[0.0, 1.0, -1.0],
                  target_device_idx=target_device_idx)

        self.assertEqual(avc.outputs['out_comm'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_freq'].value.shape, (n_avc,))
        self.assertEqual(avc.outputs['out_state'].value.shape, (n_avc, 4))
        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_avc))

    @cpu_and_gpu
    def test_invalid_n_avc_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVC(simul_params=simul_params, n_avc=0,
                freq=10.0, gx=0.05, gomega=0.001, k=1.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_mismatched_parameter_length_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        with self.assertRaises(ValueError):
            AVC(simul_params=simul_params, n_avc=3,
                freq=[10.0, 20.0], gx=0.05, gomega=0.001, k=1.0,
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
            AVC(simul_params=simul_params, n_avc=1, freq=47.0,
                gx=3.0, gomega=0.2, k=1.0,
                plant_re=1.0, plant_gain=1.0,
                target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_wrong_input_length_raises(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        avc = AVC(simul_params=simul_params, n_avc=2, freq=[25.0, 55.0],
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
    def test_cartesian_and_polar_calibration_give_the_same_map(self, target_device_idx, xp):
        """theta = -Lambda_hat / conj(P_hat): with P_hat = rho exp(i psi), the
        map (x3, x4) -> (thetac, thetas) is a rotation by psi scaled by -1/rho."""
        simul_params = SimulParams(time_step=0.001)
        rho, psi = 0.555, np.radians(119.0)
        expected = {'m11': -np.cos(psi) / rho, 'm12': np.sin(psi) / rho,
                    'm21': -np.sin(psi) / rho, 'm22': -np.cos(psi) / rho}

        cartesian = AVC(simul_params=simul_params, n_avc=1, freq=47.0,
                        gx=3.0, gomega=0.2, k=1.0,
                        plant_re=rho * np.cos(psi), plant_im=rho * np.sin(psi),
                        target_device_idx=target_device_idx)
        polar = AVC(simul_params=simul_params, n_avc=1, freq=47.0,
                    gx=3.0, gomega=0.2, k=1.0,
                    plant_gain=rho, plant_phase=np.degrees(psi),
                    target_device_idx=target_device_idx)

        for avc in (cartesian, polar):
            for name, value in expected.items():
                np.testing.assert_allclose(cpuArray(getattr(avc, name)), [value], rtol=1e-6)

    @cpu_and_gpu
    def test_first_trigger_outputs_zero_correction(self, target_device_idx, xp):
        simul_params = SimulParams(time_step=0.001)
        n_avc = 2
        avc = AVC(simul_params=simul_params, n_avc=n_avc, freq=[25.0, 55.0],
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

        np.testing.assert_allclose(cpuArray(avc.outputs['out_comm'].value), np.zeros(n_avc))

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
    def test_converges_within_90_degrees_of_calibration_error(self, target_device_idx, xp):
        """The convergence rate is 0.5 * gx * |P_star / P_hat| * cos(phase
        error): positive up to 90 degrees."""
        for dphi, max_ratio in ((0.0, 0.01), (60.0, 0.05), (85.0, 1.0)):
            with self.subTest(dphi=dphi):
                p_hat = P_STAR * np.exp(1j * np.radians(dphi))
                avc = _make_avc(target_device_idx, p_hat=p_hat)
                res = _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)
                self.assertLess(res[-200:].std() / res[:200].std(), max_ratio)

    @cpu_and_gpu
    def test_diverges_beyond_90_degrees_of_calibration_error(self, target_device_idx, xp):
        for dphi, min_ratio in ((95.0, 1.0), (120.0, 100.0)):
            with self.subTest(dphi=dphi):
                p_hat = P_STAR * np.exp(1j * np.radians(dphi))
                avc = _make_avc(target_device_idx, p_hat=p_hat)
                res = _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)
                self.assertGreater(res[-200:].std() / res[:200].std(), min_ratio)

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
        from a wrong initial guess."""
        T = 0.001
        true_freq = 50.0
        simul_params = SimulParams(time_step=T)
        avc = AVC(simul_params=simul_params, n_avc=1, freq=48.0,
                  gx=0.5, gomega=5.0, k=1.0,
                  target_device_idx=target_device_idx)

        meas = BaseValue(value=xp.zeros(1), target_device_idx=target_device_idx)
        avc.inputs['in_measurement'].set(meas)
        avc.setup()

        for it in range(20000):
            t_sec = (it + 1) * T
            t = avc.seconds_to_t(t_sec)
            meas.value[:] = xp.asarray([0.7 * np.sin(2 * np.pi * true_freq * t_sec)])
            meas.generation_time = t
            avc.check_ready(t)
            avc.trigger()
            avc.post_trigger()

        final_freq = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(final_freq, true_freq, delta=1.0)

    @cpu_and_gpu
    def test_gomega_zero_holds_frequency(self, target_device_idx, xp):
        avc = _make_avc(target_device_idx)
        _run_closed_loop(avc, 2000, P_STAR, LAM_STAR, target_device_idx, xp)

        f_est = float(cpuArray(avc.outputs['out_freq'].value)[0])
        self.assertAlmostEqual(f_est, 47.0, places=4)


if __name__ == '__main__':
    unittest.main()
