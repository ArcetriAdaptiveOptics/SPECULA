"""
Tests of the Gauss-Newton fit in BaseSprintEstimator._iterative_estimation.

The real ``_iterative_estimation``, ``_compute_sensitivity_matrices`` (central
differences), ``_misfit`` and ``_check_observability`` are exercised. Only
``_compute_nominal_im`` is replaced on the instance by a synthetic model that is
LINEAR in the 4 SprintShSynim parameters (shift_x, shift_y, rotation,
magnification):

    IM(p) = IM0 + sum_i p_i * S_i

Central differences are exact on a linear model, so the Gauss-Newton step is
exact and the physically predictable result is: the fit recovers the true
parameters, whatever the (unknown) optical gain.
"""
import logging
import unittest

import numpy as np

import specula
specula.init(0, log_level=logging.INFO)

from specula.processing_objects.sprint_sh_synim import SprintShSynim

from test.specula_testlib import cpu_and_gpu
from test.test_sprint import create_test_system

NSLOPES = 2000
N_PARAMS = 4
# Perturbation sizes of SprintShSynim: used to express the errors in "steps"
PERTURBATIONS = np.array([1.0, 1.0, 0.1, 0.01])
SENS_SCALE = 0.01
P_TRUE = np.array([0.7, -1.3, 0.25, 0.03])


def _sensitivities(nmodes, seed=0, correlated=True, scale=SENS_SCALE):
    """Fixed random IM0 (nslopes, nmodes) and S (nslopes, nmodes, 4).

    With ``correlated`` S_2 = 0.9 S_1 + 0.44 noise (correlation ~0.9).

    S is small with respect to IM0 (S p_true ~ 1% of the IM rms, as for a
    real mis-registration). This matters: the Gauss-Newton step uses the
    optical gain estimated at the current point, G(p), and does not account for
    dG/dp, so the model is exactly linear only when the parameters do not change
    the scale of the IM. The bias of G scales as |S p|^2 / |IM0|^2 and sets a
    linear convergence rate of the same order after the first (exact) step.
    """
    rng = np.random.default_rng(seed)
    im0 = rng.standard_normal((NSLOPES, nmodes))
    # Scaled like the physical sensitivities (IM change per unit of parameter)
    s = rng.standard_normal((NSLOPES, nmodes, N_PARAMS)) * PERTURBATIONS * scale
    if correlated:
        s[:, :, 2] = 0.9 * s[:, :, 1] * (PERTURBATIONS[2] / PERTURBATIONS[1]) \
            + 0.44 * s[:, :, 2]
    return im0, s


def _make_estimator(target_device_idx, im0, s, nmodes, **kwargs):
    """SprintShSynim whose nominal IM is the linear synthetic model (float64)."""
    simul_params, source, dm, wfs, ccd, slopec = create_test_system()
    params = dict(max_iterations=20, convergence_threshold=1e-12,
                  integration_gain=1.0)
    params.update(kwargs)
    sprint = SprintShSynim(
        simul_params=simul_params, dm=dm, slopec=slopec, source=source, wfs=wfs,
        modes_index=list(range(nmodes)),
        carrier_frequencies=[100.0 + 10 * i for i in range(nmodes)],
        target_device_idx=target_device_idx,
        precision=0,
        **params,
    )
    xp = sprint.xp
    im0_x = xp.asarray(im0, dtype=sprint.dtype)
    s_x = xp.asarray(s, dtype=sprint.dtype)
    sprint.misreg_params = xp.zeros(N_PARAMS, dtype=sprint.dtype)
    # Normally done in setup() from the input slopes
    sprint.estimated_intmat.set_nslopes(NSLOPES)

    def nominal():
        return im0_x + xp.einsum('smp,p->sm', s_x, sprint.misreg_params)

    sprint._compute_nominal_im = nominal
    return sprint


def _measured(sprint, im0, s, gains, p_true=P_TRUE):
    """Measured IM = G * IM(p_true), one gain per mode."""
    im = im0 + np.einsum('smp,p->sm', s, p_true)
    return sprint.xp.asarray(im * np.asarray(gains)[None, :], dtype=sprint.dtype)


def _count_calls(sprint, name):
    """Wrap sprint.<name> on the instance; return the list of recorded results."""
    orig = getattr(sprint, name)
    calls = []

    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        calls.append(out)
        return out

    setattr(sprint, name, wrapper)
    return calls


def _estimate(sprint):
    return np.asarray(sprint.xp.asnumpy(sprint.misreg_params)
                      if hasattr(sprint.xp, 'asnumpy') else sprint.misreg_params)


class TestSprintFit(unittest.TestCase):

    def _check_truth(self, sprint, tol_steps):
        err = np.abs(_estimate(sprint) - P_TRUE) / PERTURBATIONS
        self.assertLess(err.max(), tol_steps, msg=f"errors in perturbation units: {err}")

    @cpu_and_gpu
    def test_linear_model_converges_to_the_truth(self, target_device_idx, xp):
        """Gain 1: recovery in at most 4 iterations, also for 2 modes.

        S_2 is strongly correlated with S_1 (correlation ~0.9). The previous
        per-parameter projection delta_p = s_p.d / s_p.s_p ignores the
        correlation, so it would not converge to the truth in this case; the
        joint Gauss-Newton step does.
        """
        for nmodes, gains in ((1, [-7.3]), (2, [-7.3, 2.1])):
            with self.subTest(nmodes=nmodes):
                im0, s = _sensitivities(nmodes)
                sprint = _make_estimator(target_device_idx, im0, s, nmodes)
                n_sens = _count_calls(sprint, '_compute_sensitivity_matrices')
                sprint._iterative_estimation(_measured(sprint, im0, s, gains))

                self._check_truth(sprint, 1e-6)
                self.assertLess(sprint.current_error, 1e-10)
                # 1e-12 residual: exact step, plus a few refinements of the gain bias
                self.assertLessEqual(len(n_sens), 4)

    @cpu_and_gpu
    def test_variable_projection_converges_with_a_large_effect(self, target_device_idx, xp):
        """Large mis-registration effect (|S p_true| / |IM0| ~ 1.5): fast convergence.

        The optical gain G is re-estimated at every point, so the quantity
        fitted is the residual meas - G(p) IM(p), not meas - G IM(p) with a
        fixed G. Its Jacobian is obtained by projecting the sensitivities of
        each mode on the complement of the current nominal IM (Kaufman's
        variable projection). Without the projection the step ignores dG/dp
        and the convergence is only linear when the effect is large.

        Measured (1 and 2 modes): with the projection the fit reaches P_TRUE
        (error < 1e-6 perturbation units) in 4 iterations; with the projection
        disabled it needs ~57.
        """
        for nmodes, gains in ((1, [-7.3]), (2, [-7.3, 2.1])):
            with self.subTest(nmodes=nmodes):
                im0, s = _sensitivities(nmodes, scale=1.0)
                ratio = np.linalg.norm(np.einsum('smp,p->sm', s, P_TRUE)) \
                    / np.linalg.norm(im0)
                # Test design: the effect on the IM is large
                self.assertGreater(ratio, 1.2)

                def run(projection):
                    sprint = _make_estimator(
                        target_device_idx, im0, s, nmodes, max_iterations=100,
                        convergence_threshold=1e-12, param_tolerance=1e-9)
                    if not projection:
                        orig = sprint._estimate_misreg_correction
                        sprint._estimate_misreg_correction = \
                            lambda im_diff, sens, im_nom=None: orig(im_diff, sens)
                    n_sens = _count_calls(sprint, '_compute_sensitivity_matrices')
                    sprint._iterative_estimation(_measured(sprint, im0, s, gains))
                    return sprint, len(n_sens)

                sprint, n_proj = run(True)
                self._check_truth(sprint, 1e-6)
                self.assertLessEqual(n_proj, 5)

                # Control: no projection reaches the same tolerance, but slowly
                sprint, n_plain = run(False)
                self._check_truth(sprint, 1e-6)
                self.assertGreater(n_plain, 20)

    @cpu_and_gpu
    def test_integration_gain_reaches_the_truth(self, target_device_idx, xp):
        """integration_gain=0.5 halves the error at each iteration."""
        im0, s = _sensitivities(1)
        # The default param_tolerance (0.01 of the perturbations) would stop the
        # fit long before 1e-6
        sprint = _make_estimator(target_device_idx, im0, s, 1,
                                 integration_gain=0.5, max_iterations=60,
                                 param_tolerance=1e-8)
        n_sens = _count_calls(sprint, '_compute_sensitivity_matrices')
        sprint._iterative_estimation(_measured(sprint, im0, s, [-7.3]))

        self._check_truth(sprint, 1e-6)
        self.assertGreaterEqual(len(n_sens), 15)
        # Stopped on the parameter step (or residual), not on max_iterations
        self.assertLess(len(n_sens), 60)

    @cpu_and_gpu
    def test_best_point_is_returned(self, target_device_idx, xp):
        """Steps are not rejected: the lowest-residual point is the one returned.

        The first step is halved (residual drops to a moderate value) and the
        second one is 5x the Gauss-Newton step (overshoot: the residual grows
        again). max_iterations=2 stops the fit right after the bad step.
        """
        im0, s = _sensitivities(1)
        sprint = _make_estimator(target_device_idx, im0, s, 1, max_iterations=2)

        orig_step = sprint._estimate_misreg_correction
        factors = iter([0.5, 5.0])
        sprint._estimate_misreg_correction = lambda *args: next(factors) * orig_step(*args)

        orig_misfit = sprint._misfit
        history = []  # (residual, params) at the initial point and after each step

        def misfit(*args, **kwargs):
            out = orig_misfit(*args, **kwargs)
            history.append((out[1], _estimate(sprint).copy()))
            return out

        sprint._misfit = misfit
        sprint._iterative_estimation(_measured(sprint, im0, s, [-7.3]))

        self.assertEqual(len(history), 3)
        residuals = [h[0] for h in history]
        # Test design: iteration 1 improves, iteration 2 (overshoot) is worse
        self.assertLess(residuals[1], residuals[0])
        self.assertGreater(residuals[2], residuals[1])

        best = int(np.argmin(residuals))
        self.assertEqual(best, 1)
        self.assertAlmostEqual(sprint.current_error, residuals[best], delta=1e-12)
        np.testing.assert_allclose(_estimate(sprint), history[best][1], atol=1e-12)
        self.assertGreater(np.abs(_estimate(sprint) - history[2][1]).max(), 1e-3)

    @cpu_and_gpu
    def test_unobservable_parameter_warns(self, target_device_idx, xp):
        gains = [-7.3]
        # Magnification with zero sensitivity
        im0, s = _sensitivities(1)
        s[:, :, 3] = 0.0
        sprint = _make_estimator(target_device_idx, im0, s, 1, max_iterations=3)
        with self.assertLogs(sprint.logger.name, level=logging.WARNING) as cm:
            sprint._iterative_estimation(_measured(sprint, im0, s, gains))
        text = "\n".join(cm.output)
        self.assertIn("no sensitivity", text)
        self.assertIn("magnification", text)

        # shift_y identical to shift_x: degenerate pair
        im0, s = _sensitivities(1, correlated=False)
        s[:, :, 1] = s[:, :, 0]
        sprint = _make_estimator(target_device_idx, im0, s, 1, max_iterations=3)
        with self.assertLogs(sprint.logger.name, level=logging.WARNING) as cm:
            sprint._iterative_estimation(_measured(sprint, im0, s, gains))
        self.assertIn("poorly constrained", "\n".join(cm.output))

    @cpu_and_gpu
    def test_well_conditioned_model_does_not_warn(self, target_device_idx, xp):
        """Control for the warnings: independent sensitivities log no WARNING."""
        im0, s = _sensitivities(1, correlated=False)
        sprint = _make_estimator(target_device_idx, im0, s, 1, max_iterations=3)
        with self.assertNoLogs(sprint.logger.name, level=logging.WARNING):
            sprint._iterative_estimation(_measured(sprint, im0, s, [-7.3]))

    @cpu_and_gpu
    def test_zero_sensitivity_gives_no_step(self, target_device_idx, xp):
        """S = 0: normal equations are all zero, parameters stay at the start."""
        im0, s = _sensitivities(1)
        s[:] = 0.0
        sprint = _make_estimator(target_device_idx, im0, s, 1, max_iterations=3)
        # Measurement not explained by the model: the fit cannot converge
        measured = _measured(sprint, im0, _sensitivities(1)[1], [-7.3])
        sprint._iterative_estimation(measured)

        np.testing.assert_array_equal(_estimate(sprint), np.zeros(N_PARAMS))
        self.assertTrue(np.isfinite(sprint.current_error))


if __name__ == '__main__':
    unittest.main()
