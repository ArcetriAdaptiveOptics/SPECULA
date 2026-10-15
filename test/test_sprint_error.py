"""
Tests of the error reported by BaseSprintEstimator._iterative_estimation.

The reported error (``current_error``) is the least-squares residual relative
to the measurement, rms(meas - nominal * G) / rms(meas), so it must not depend
on the probe amplitude (scale and sign of the measured IM).

The real ``_iterative_estimation`` is exercised. Only the two model methods are
replaced on the instance: ``_compute_nominal_im`` returns a fixed synthetic IM
and ``_compute_sensitivity_matrices`` returns zeros, so the misregistration
does not move and the "model at the true registration" condition holds exactly.
"""
import logging
import unittest

import numpy as np

import specula
specula.init(0, log_level=logging.INFO)

from specula.processing_objects.sprint_sh_synim import SprintShSynim

from test.specula_testlib import cpu_and_gpu
from test.test_sprint import create_test_system

NSLOPES = 4000


def _make_estimator(target_device_idx, im_nominal, nmodes=1):
    """SprintShSynim with a frozen synthetic nominal IM (float64)."""
    simul_params, source, dm, wfs, ccd, slopec = create_test_system()
    sprint = SprintShSynim(
        simul_params=simul_params, dm=dm, slopec=slopec, source=source, wfs=wfs,
        modes_index=list(range(nmodes)),
        carrier_frequencies=[100.0 + 10 * i for i in range(nmodes)],
        max_iterations=1,
        convergence_threshold=1e-12,
        target_device_idx=target_device_idx,
        precision=0,
    )
    xp = sprint.xp
    nom = xp.asarray(im_nominal, dtype=sprint.dtype)
    sprint._compute_nominal_im = lambda: nom
    sprint._compute_sensitivity_matrices = lambda: xp.zeros(
        (nom.shape[0], nom.shape[1], sprint.n_params), dtype=sprint.dtype)
    return sprint


def _run(sprint, im_measured):
    xp = sprint.xp
    sprint._iterative_estimation(xp.asarray(im_measured, dtype=sprint.dtype))
    return sprint.current_error


def _old_error(im_nominal, im_measured):
    """Previous definition: rms(meas / G - nominal) / rms(meas), single mode."""
    g = (im_nominal.T @ im_measured) / (im_nominal.T @ im_nominal)
    return (np.sqrt(np.mean((im_measured / g - im_nominal) ** 2))
            / np.sqrt(np.mean(im_measured ** 2))).item()


def _synthetic(sigma_rel, amplitude, seed=0):
    """Unit-RMS signal; noise RMS = sigma_rel * |amplitude| (fixed SNR)."""
    rng = np.random.default_rng(seed)
    nom = rng.standard_normal((NSLOPES, 1))
    nom /= np.sqrt(np.mean(nom ** 2))
    noise = rng.standard_normal((NSLOPES, 1)) * sigma_rel * abs(amplitude)
    return nom, amplitude * nom + noise


class TestSprintReportedError(unittest.TestCase):

    @cpu_and_gpu
    def test_error_is_amplitude_invariant(self, target_device_idx, xp):
        """Same SNR, amplitudes 1, 10, 100 and -10: same reported error."""
        errors = {}
        for amp in (1.0, 10.0, 100.0, -10.0):
            nom, meas = _synthetic(0.3, amp)
            errors[amp] = _run(_make_estimator(target_device_idx, nom), meas)
        ref = errors[1.0]
        for amp, err in errors.items():
            self.assertAlmostEqual(err / ref, 1.0, delta=0.01, msg=f"A={amp}")

    def test_old_expression_is_not_amplitude_invariant(self):
        """Sanity check of the test design: the old formula scales as 1/|A|."""
        nom1, meas1 = _synthetic(0.3, 1.0)
        nom100, meas100 = _synthetic(0.3, 100.0)
        ratio = _old_error(nom1, meas1) / _old_error(nom100, meas100)
        self.assertGreater(ratio, 50.0)

    @cpu_and_gpu
    def test_expected_value_white_noise(self, target_device_idx, xp):
        """err ~ sigma / sqrt(s^2 + sigma^2) for white noise on a single mode."""
        for sigma_rel in (0.1, 0.5):
            nom, meas = _synthetic(sigma_rel, 7.0, seed=1)
            err = _run(_make_estimator(target_device_idx, nom), meas)
            expected = sigma_rel / np.sqrt(1.0 + sigma_rel ** 2)
            self.assertAlmostEqual(err / expected, 1.0, delta=0.03,
                                   msg=f"sigma/s={sigma_rel}")

    @cpu_and_gpu
    def test_noise_free_gives_zero(self, target_device_idx, xp):
        nom, _ = _synthetic(0.0, 1.0)
        err = _run(_make_estimator(target_device_idx, nom), -25.0 * nom)
        self.assertLess(err, 1e-9)

    @cpu_and_gpu
    def test_pure_noise_is_bounded_by_one(self, target_device_idx, xp):
        """Measurement uncorrelated with the model: error <= 1 and close to 1."""
        rng = np.random.default_rng(2)
        nom = rng.standard_normal((NSLOPES, 1))
        meas = 5.0 * rng.standard_normal((NSLOPES, 1))
        err = _run(_make_estimator(target_device_idx, nom), meas)
        self.assertLessEqual(err, 1.0)
        self.assertGreater(err, 0.99)

    @cpu_and_gpu
    def test_single_mode_equals_convergence_metric(self, target_device_idx, xp):
        """For one mode current_error == rms(im_diff)/rms(meas) * mean|G|."""
        nom, meas = _synthetic(0.4, -13.0, seed=3)
        err = _run(_make_estimator(target_device_idx, nom), meas)
        g = (nom.T @ meas) / (nom.T @ nom)
        im_diff = meas / g - nom
        error_weighted = (np.sqrt(np.mean(im_diff ** 2))
                          / np.sqrt(np.mean(meas ** 2)) * np.abs(g)).item()
        self.assertAlmostEqual(err, error_weighted, delta=1e-9)

    @cpu_and_gpu
    def test_multimode_amplitude_invariant_and_bounded(self, target_device_idx, xp):
        """Two modes with different gains: invariant to a global scale, in [0, 1]."""
        rng = np.random.default_rng(5)
        nom = rng.standard_normal((NSLOPES, 2))
        noise = 0.3 * rng.standard_normal((NSLOPES, 2))
        gains = np.array([2.0, -0.5])
        errs = []
        for amp in (1.0, 50.0):
            meas = amp * (nom * gains + noise * np.abs(gains))
            errs.append(_run(_make_estimator(target_device_idx, nom, nmodes=2), meas))
        self.assertAlmostEqual(errs[1] / errs[0], 1.0, delta=0.01)
        self.assertTrue(0.0 <= errs[0] <= 1.0)


if __name__ == '__main__':
    unittest.main()
