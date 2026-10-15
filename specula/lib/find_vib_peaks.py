"""
Vibration peak detection from a residual PSD.

Finds the narrowband peaks standing above the smoothed noise floor and fits
each with a second order (AR2) model, giving the frequencies to initialize
:py:class:`~specula.processing_objects.avc.AVC` with.

Ported from PASSATA's ``find_vib_peaks.pro`` / ``fit_ar2.pro`` (G. Agapito,
2022), derived from the Matlab ``optAVC`` library (P. Haguenauer). Where the
two disagree, the Matlab source is followed.
"""
from collections import namedtuple

import numpy as np


#: Result of :func:`find_vib_peaks`: frequency [Hz], AR2 damping ratio and
#: integrated power of each detected peak.
VibPeaks = namedtuple('VibPeaks', ['freq', 'damping', 'power'])

_EMPTY_VIB_PEAKS = VibPeaks(freq=np.array([]), damping=np.array([]), power=np.array([]))


def _ar2_psd_model(freq, t, f_vib, k):
    """PSD of an AR2 resonance at ``f_vib`` [Hz] with damping ``k``, for a
    sampling period ``t`` [s]."""
    a1 = 2 * np.exp(-2 * np.pi * k * f_vib * t) * np.cos(2 * np.pi * f_vib * t * np.sqrt(1 - k ** 2))
    a2 = -np.exp(-4 * np.pi * k * f_vib * t)
    denom = np.abs(1 - a1 * np.exp(-2j * np.pi * freq * t) - a2 * np.exp(-4j * np.pi * freq * t)) ** 2
    psd = np.zeros_like(denom)
    nonzero = denom != 0
    psd[nonzero] = denom[nonzero] ** (-1)
    return psd


def _ar2_least_squares(power, freq, t):
    """Fit a single AR2 resonance to a PSD segment.

    The frequency is the one of the maximum sample; the damping is
    increased in fixed steps as long as the fit residual decreases.

    Returns the fitted PSD, the frequency [Hz] and the damping ratio.
    """
    power = np.asarray(power, dtype=float)
    freq = np.asarray(freq, dtype=float)

    peak_loc = int(np.argmax(power))
    peak_pwr = power[peak_loc]
    f_vib = freq[peak_loc]

    k = 0.0001
    k_step = 0.0005
    cost = 1e10
    num_iter = 0
    converging = True

    while converging and num_iter < 100:
        psd = _ar2_psd_model(freq, t, f_vib, k)
        if not np.any(psd):
            # AR2 denominator vanished
            return psd, f_vib, k
        sigma0 = peak_pwr / np.max(psd)
        psd = sigma0 * psd
        new_cost = float(np.sum((psd - power) ** 2))
        if cost - new_cost > 0:
            cost = new_cost
            k += k_step
            num_iter += 1
        else:
            converging = False

    k = max(k - k_step, 0.0001)
    psd = _ar2_psd_model(freq, t, f_vib, k)
    sigma0 = np.max(power) / np.max(psd)
    psd = sigma0 * psd

    return psd, f_vib, k


def fit_ar2(power, freq, t, threshold, df):
    """Fit up to two AR2 resonances to a PSD segment.

    A second fit is attempted where the relative error of the first one
    exceeds ``threshold`` over a contiguous range. It is kept if it carries
    at least 25% of the segment power and is more than ``2 * df`` away from
    the first peak.

    Parameters
    ----------
    power, freq : ndarray
        PSD samples and frequencies [Hz] of the segment.
    t : float
        Sampling period [s].
    threshold : float
        Relative error above which a second peak is searched.
    df : float
        Frequency bin spacing [Hz].

    Returns
    -------
    sigma, damping, freq_vib : ndarray, one or two elements
        Integrated power, damping ratio and frequency [Hz] of the fitted
        peaks, sorted by ascending frequency.
    """
    power = np.asarray(power, dtype=float)
    freq = np.asarray(freq, dtype=float)

    sigma_t = [0.0]
    f_vib_t = [0.0]
    k_t = [0.0]

    cum_pow = float(np.sum(power) * df)

    psd, f_vib, k = _ar2_least_squares(power, freq, t)
    cum_psd = float(np.sum(psd) * df)
    sigma_t[0] = cum_psd
    f_vib_t[0] = f_vib
    k_t[0] = k

    # A contiguous region with a large relative error hints at a second peak
    with np.errstate(divide='ignore', invalid='ignore'):
        err = np.abs(psd - power) / psd
    err = np.nan_to_num(err, nan=0.0, posinf=np.inf)
    bad = np.where(err > threshold)[0]

    is_compact = True
    if bad.size > 1:
        gaps = np.diff(bad)
        if np.any(gaps > 1):
            is_compact = False

    if bad.size > 0 and is_compact:
        init_ind = bad[0] - 1 if bad[0] > 0 else bad[0]
        end_ind = bad[-1] + 1 if bad[-1] < power.size - 1 else bad[-1]
        power_ref = power[init_ind:end_ind + 1]
        freq_ref = freq[init_ind:end_ind + 1]

        psd2, f_vib2, k2 = _ar2_least_squares(power_ref, freq_ref, t)
        cum_psd2 = float(np.sum(psd2) * df)

        if cum_psd2 > 0.25 * cum_pow and abs(f_vib_t[0] - f_vib2) > 2 * df:
            sigma_t.append(cum_psd2)
            f_vib_t.append(f_vib2)
            k_t.append(k2)

    sigma_t = np.asarray(sigma_t)
    f_vib_t = np.asarray(f_vib_t)
    k_t = np.asarray(k_t)

    order = np.argsort(f_vib_t)
    return sigma_t[order], k_t[order], f_vib_t[order]


def find_vib_peaks(psd, freq, fmin, fmax,
                   num_sampl_init=50,
                   smooth_alpha=0.99,
                   detection_threshold=3.0,
                   power_ratio_threshold=0.01,
                   fit_threshold=0.5):
    """Detect vibration peaks in a residual PSD.

    Samples of the ``[fmin, fmax]`` band exceeding ``detection_threshold``
    times the smoothed noise floor are grouped by contiguity; the groups
    with enough power are fitted with :func:`fit_ar2`.

    Parameters
    ----------
    psd, freq : array-like
        PSD of the residual and its frequencies [Hz], uniformly spaced.
    fmin, fmax : float
        Frequency band [Hz] to search.
    num_sampl_init : int, optional
        Samples used to initialize the backward smoothing filter, and
        minimum number of in-band samples. Default: 50.
    smooth_alpha : float, optional
        Pole of the noise floor smoothing filter, in (0, 1). Default: 0.99.
    detection_threshold : float, optional
        Detection level, in units of the smoothed floor. Default: 3.0.
    power_ratio_threshold : float, optional
        Minimum fraction of the in-band power for a group to be kept.
        Default: 0.01.
    fit_threshold : float, optional
        ``threshold`` of :func:`fit_ar2`. Default: 0.5.

    Returns
    -------
    VibPeaks
        ``freq`` [Hz], ``damping`` and ``power`` arrays, one entry per
        vibration. Empty if none is found.
    """
    psd = np.asarray(psd, dtype=float)
    freq = np.asarray(freq, dtype=float)

    if psd.shape != freq.shape:
        raise ValueError(f'psd and freq must have the same shape, got {psd.shape} and {freq.shape}')

    fs = freq[-1] * 2
    dfreq = freq[1] - freq[0]
    expand_freq = dfreq * 1.1

    band = (freq >= fmin) & (freq <= fmax)
    n_red = int(np.count_nonzero(band))
    if n_red == 0 or n_red <= num_sampl_init:
        return _EMPTY_VIB_PEAKS

    freq_red = freq[band]
    psd_red = psd[band]
    cum_psd_total = float(np.sum(psd_red) * dfreq)

    # Noise floor: average of a forward and a backward exponential
    # smoothing, to avoid a directional lag near the peaks.
    psd_filt_fwd = np.empty(n_red)
    x = psd_red[0]
    for kk in range(n_red):
        x = smooth_alpha * x + (1 - smooth_alpha) * psd_red[kk]
        psd_filt_fwd[kk] = x

    psd_filt_bwd = np.empty(n_red)
    x = float(np.mean(psd_red[-num_sampl_init:]))
    for kk in range(n_red):
        y = psd_red[n_red - 1 - kk]
        x = smooth_alpha * x + (1 - smooth_alpha) * y
        psd_filt_bwd[kk] = x

    psd_smooth = (psd_filt_fwd + psd_filt_bwd[::-1]) / 2

    above = psd_red > detection_threshold * psd_smooth
    if not np.any(above):
        return _EMPTY_VIB_PEAKS

    samples = psd_red[above]
    samp_freq = freq_red[above]

    # Group contiguous (within expand_freq) detections together.
    if samp_freq.size == 1:
        group_bounds = np.array([0, 1])
    else:
        gaps = np.diff(samp_freq)
        breaks = np.where(gaps > expand_freq)[0]
        group_bounds = np.concatenate(([0], breaks + 1, [samp_freq.size]))

    power_content = np.array([
        np.sum(samples[group_bounds[ii]:group_bounds[ii + 1]]) * dfreq
        for ii in range(len(group_bounds) - 1)
    ])

    freq_list, k_list, sigma_list = [], [], []

    for ii in range(power_content.size):
        if power_content[ii] <= cum_psd_total * power_ratio_threshold:
            continue

        f_lo = samp_freq[group_bounds[ii]] - expand_freq
        f_hi = samp_freq[group_bounds[ii + 1] - 1] + expand_freq
        fit_mask = (freq_red > f_lo) & (freq_red < f_hi)
        freq_fit = freq_red[fit_mask]
        power_fit = psd_red[fit_mask]

        if freq_fit.size == 0:
            continue

        sigma_t, k_t, f_vib_t = fit_ar2(power_fit, freq_fit, 1.0 / fs, fit_threshold, dfreq)

        if np.sum(sigma_t) < 2 * power_content[ii] and np.max(f_vib_t) < fmax:
            for jj in range(f_vib_t.size):
                if f_vib_t[jj] > 0:
                    freq_list.append(f_vib_t[jj])
                    k_list.append(k_t[jj])
                    sigma_list.append(sigma_t[jj])

    if not freq_list:
        return _EMPTY_VIB_PEAKS

    return VibPeaks(freq=np.array(freq_list), damping=np.array(k_list), power=np.array(sigma_list))
