import specula
specula.init(0)  # Default target device

import unittest

import numpy as np
from scipy.signal import lfilter, welch

from specula.lib.find_vib_peaks import find_vib_peaks, fit_ar2, VibPeaks

# PSD frequency grid of 10 s of data at 1 kHz: 0.1 ... 500 Hz, one bin = 0.1 Hz
FS = 1000.0
FREQ = np.fft.rfftfreq(10000, d=1 / FS)[1:]
DF = FREQ[1] - FREQ[0]

# find_vib_peaks searches the damping in steps of 0.0005
DAMPING_STEP = 0.0005


def _resonance_poles(f0, zeta, fs=FS):
    """Discrete poles of a damped resonance at f0 [Hz] with damping ratio
    zeta: z = exp(s / fs), s = -zeta w0 +/- 1j w0 sqrt(1 - zeta**2)."""
    w0 = 2 * np.pi * f0
    p = np.exp((-zeta * w0 + 1j * w0 * np.sqrt(1 - zeta**2)) / fs)
    return p, np.conj(p)


def _ar2_psd(freq, f0, zeta, power, fs=FS):
    """PSD of an AR2 process with the poles of the resonance, normalized to
    the given integrated power."""
    p1, p2 = _resonance_poles(f0, zeta, fs)
    z_inv = np.exp(-2j * np.pi * freq / fs)
    psd = 1 / np.abs((1 - p1 * z_inv) * (1 - p2 * z_inv))**2
    return psd / (np.sum(psd) * (freq[1] - freq[0])) * power


class TestFindVibPeaks(unittest.TestCase):

    def test_single_peak_within_one_bin(self):
        for f0 in (47.3, 47.33, 123.45):
            with self.subTest(f0=f0):
                zeta, power = 0.005, 30.0
                psd = 0.05 + _ar2_psd(FREQ, f0, zeta, power)

                result = find_vib_peaks(psd, FREQ, fmin=1.0, fmax=400.0)

                self.assertIsInstance(result, VibPeaks)
                self.assertEqual(result.freq.size, 1)
                self.assertLessEqual(abs(result.freq[0] - f0), DF)
                self.assertLessEqual(abs(result.damping[0] - zeta), DAMPING_STEP)
                # the fit integrates the peak region only
                self.assertGreater(result.power[0], 0.5 * power)
                self.assertLess(result.power[0], 1.2 * power)

    def test_two_peaks_within_one_bin(self):
        peaks = [(23.4, 0.004, 50.0), (87.1, 0.006, 20.0)]
        psd = 0.05 + sum(_ar2_psd(FREQ, f0, zeta, power) for f0, zeta, power in peaks)

        result = find_vib_peaks(psd, FREQ, fmin=1.0, fmax=400.0)

        self.assertEqual(result.freq.size, 2)
        np.testing.assert_array_equal(result.freq, np.sort(result.freq))
        for (f0, zeta, _), f, k in zip(peaks, result.freq, result.damping):
            self.assertLessEqual(abs(f - f0), DF)
            self.assertLessEqual(abs(k - zeta), DAMPING_STEP)

    def test_vibration_in_noise_from_time_series(self):
        """AR2 vibration plus white noise, PSD estimated with Welch's
        method as for real telemetry."""
        rng = np.random.default_rng(42)
        n = 100000
        p1, p2 = _resonance_poles(47.3, 0.005)
        ar2 = np.real(np.poly([p1, p2]))
        x = lfilter([1.0], ar2, rng.normal(size=n)) + 3.0 * rng.normal(size=n)
        # 4 s segments: 0.25 Hz bins resolve the 0.47 Hz wide peak
        # (2 * zeta * f0), and ~49 averages keep the PSD noise (~15%) far
        # below the 3x detection threshold. The window broadens the peak,
        # so the damping is not checked here.
        freq, psd = welch(x, fs=FS, nperseg=4000)

        result = find_vib_peaks(psd[1:], freq[1:], fmin=1.0, fmax=400.0)

        self.assertEqual(result.freq.size, 1)
        self.assertLessEqual(abs(result.freq[0] - 47.3), freq[1] - freq[0])

    def test_white_noise_gives_no_peaks(self):
        rng = np.random.default_rng(3)
        # same Welch setup as test_vibration_in_noise_from_time_series
        freq, psd = welch(rng.normal(size=100000), fs=FS, nperseg=4000)

        result = find_vib_peaks(psd[1:], freq[1:], fmin=1.0, fmax=400.0)

        self.assertEqual(result.freq.size, 0)
        self.assertEqual(result.damping.size, 0)
        self.assertEqual(result.power.size, 0)

    def test_low_power_peak_is_rejected(self):
        """A peak with a negligible fraction of the in-band power is not
        reported (power_ratio_threshold)."""
        psd = 5.0 + _ar2_psd(FREQ, 100.0, 0.005, 0.01)

        result = find_vib_peaks(psd, FREQ, fmin=1.0, fmax=400.0, power_ratio_threshold=0.01)

        self.assertEqual(result.freq.size, 0)

    def test_band_too_narrow_returns_no_peaks(self):
        psd = np.full_like(FREQ, 0.1)

        result = find_vib_peaks(psd, FREQ, fmin=1.0, fmax=2.0, num_sampl_init=50)

        self.assertEqual(result.freq.size, 0)

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            find_vib_peaks(np.ones(10), FREQ, fmin=1.0, fmax=400.0)


class TestFitAr2(unittest.TestCase):

    def test_recovers_known_single_peak(self):
        band = (FREQ > 40.0) & (FREQ < 60.0)
        f0, zeta = 50.0, 0.005
        power = _ar2_psd(FREQ, f0, zeta, 10.0)[band]

        sigma, k, f_vib = fit_ar2(power, FREQ[band], 1 / FS, threshold=0.5, df=DF)

        self.assertEqual(f_vib.size, 1)
        self.assertLessEqual(abs(f_vib[0] - f0), DF)
        self.assertLessEqual(abs(k[0] - zeta), DAMPING_STEP)
        self.assertGreater(sigma[0], 0)

    def test_two_close_peaks_in_one_segment(self):
        """Two close peaks in one segment: the first fit, on the strongest,
        leaves a contiguous region of large error around the second one,
        which is then fitted too (the secondary-peak branch). Both are
        found within one bin and returned sorted by frequency."""
        band = (FREQ > 45.0) & (FREQ < 56.0)
        peaks = [(50.0, 0.003, 20.0), (51.0, 0.003, 8.0)]
        psd = sum(_ar2_psd(FREQ, f0, zeta, power) for f0, zeta, power in peaks)

        sigma, k, f_vib = fit_ar2(psd[band], FREQ[band], 1 / FS, threshold=0.5, df=DF)

        self.assertEqual(f_vib.size, 2)
        np.testing.assert_array_equal(f_vib, np.sort(f_vib))
        for (f0, _, _), f in zip(peaks, f_vib):
            self.assertLessEqual(abs(f - f0), DF)


if __name__ == '__main__':
    unittest.main()
