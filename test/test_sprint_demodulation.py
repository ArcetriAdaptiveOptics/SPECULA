"""
Tests of BaseSprintEstimator._demodulate_slopes (coherent demodulation).

Slopes are synthetic: s_i(t) = a_i cos(2 pi f t + phi_c) + n_i(t), with signed
a_i and a carrier phase phi_c common to all slopes. The demodulated IM column
must be linear in the slopes: a_i up to one global sign (phi0 is defined
modulo pi), whatever phi_c, and without the positive bias that rectifying the
noise (|z|) produces at low SNR.
"""
import logging
import unittest

import numpy as np

import specula
specula.init(0, log_level=logging.INFO)

from specula.lib.demodulate_signal import demodulate_signal
from specula.processing_objects.base_sprint_estimator import BaseSprintEstimator

from test.specula_testlib import cpu_and_gpu

FS = 1000.0
FREQ = 40.0
NT = 2000  # 80 carrier periods: exact lock-in


def _make_estimator(slopes, xp):
    """Bare BaseSprintEstimator (no __init__) with only the attributes used."""
    est = BaseSprintEstimator.__new__(BaseSprintEstimator)
    est.xp = xp
    est.dtype = xp.float64
    est.dt = 1.0 / FS
    est.nmodes = 1
    est.carrier_frequencies = [FREQ]
    est.apply_absolute_slopes = False
    est.logger = logging.getLogger('test_sprint_demodulation')
    est.slopes_history = [xp.asarray(row) for row in slopes]
    est.time_history = list(range(len(slopes)))
    return est


def _slopes(a, phi_c, sigma, seed):
    """(nt, nslopes) history of a_i cos(w t + phi_c) + white noise."""
    t = np.arange(NT) / FS
    carrier = np.cos(2 * np.pi * FREQ * t + phi_c)
    noise = np.random.default_rng(seed).normal(0.0, sigma, (NT, len(a)))
    return carrier[:, None] * a[None, :] + noise


def _demodulate(slopes, xp):
    est = _make_estimator(slopes, xp)
    im = est._demodulate_slopes()
    return np.asarray(im.get() if hasattr(im, 'get') else im)[:, 0]


def _spread_amplitudes(n, seed):
    """Random signs, magnitudes in [0.5, 1.5]."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)


class TestSprintDemodulation(unittest.TestCase):

    @cpu_and_gpu
    def test_high_snr_recovers_amplitudes(self, target_device_idx, xp):
        """Noiseless: column = +-a (one global sign) for phi_c = 0 and 90 deg."""
        a = _spread_amplitudes(50, seed=1)
        for phi_c in (0.0, np.pi / 2):
            d = _demodulate(_slopes(a, phi_c, 0.0, seed=0), xp)
            s = np.sign(d @ a)
            np.testing.assert_allclose(s * d, a, rtol=1e-3, atol=1e-3)
            # All slopes share the same sign relation (old method: random at 90 deg)
            self.assertTrue(np.all(np.sign(d) == s * np.sign(a)))

    @cpu_and_gpu
    def test_low_snr_unbiased(self, target_device_idx, xp):
        """phi_c = 90 deg, lock-in SNR ~1.5: mean error ~0, regression gain ~1,
        while the old amp*sign(cos(phase)) estimator is biased low in gain."""
        a = _spread_amplitudes(2000, seed=2)
        # lock-in noise per quadrature = sigma*sqrt(2/NT) = 0.67 for |a| = 1
        sigma = 0.67 * np.sqrt(NT / 2)
        slopes = _slopes(a, np.pi / 2, sigma, seed=3)

        d = _demodulate(slopes, xp)
        s = np.sign(d @ a)
        err = s * d - a
        sem = err.std() / np.sqrt(len(a))
        self.assertLess(abs(err.mean()), 4 * sem)
        gain_new = s * (d @ a) / (a @ a)
        self.assertAlmostEqual(gain_new, 1.0, delta=0.05)

        # Old estimator, inline (global sign fixed the same way)
        amp, phase = demodulate_signal(xp.asarray(slopes), FREQ, FS,
                                       cumulated=True, xp=xp, dtype=xp.float64)
        d_old = amp * xp.sign(xp.cos(phase))
        d_old = d_old.get() if hasattr(d_old, 'get') else d_old
        gain_old = abs(d_old @ a) / (a @ a)
        self.assertLess(gain_old, gain_new - 0.1)

    @cpu_and_gpu
    def test_pure_noise_has_zero_mean(self, target_device_idx, xp):
        """a = 0: the column is linear in the noise, so its mean over slopes
        is ~0 (|z| would give a positive mean)."""
        slopes = _slopes(np.zeros(2000), 0.0, 20.0, seed=4)
        d = _demodulate(slopes, xp)
        sem = d.std() / np.sqrt(len(d))
        self.assertLess(abs(d.mean()), 4 * sem)
        self.assertGreater(d.std(), 0.0)


if __name__ == '__main__':
    unittest.main()
