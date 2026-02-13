"""
Signal demodulation utilities.
Based on demodulate_passata.pro from PASSATA/LBT-SOUL.
"""
from specula import np


def demodulate_signal(signal_data, carrier_freq, sampling_freq,
                     cumulated=True, verbose=False, xp=np, dtype=np.float32):
    """
    Demodulate signal(s) using a carrier frequency.
    
    Vectorized version that can process multiple signals simultaneously.
    
    parameters
    ----------
    signal_data : array_like
        Input signal time series. Can be:
        - 1D array: single signal, shape (nt,)
        - 2D array: multiple signals, shape (nt, nsignals)
    carrier_freq : float
        Carrier frequency in Hz
    sampling_freq : float
        Sampling frequency in Hz
    cumulated : bool, optional
        If True, use cumulative demodulation averaging. Default: True
    verbose : bool, optional
        Print debug information. Default: False
    xp : module, optional
        Array module (numpy or cupy). Default: numpy (np)
    dtype : data-type, optional
        Data type for computations. Default: np.float32
    
    Returns
    -------
    value : float or ndarray
        Demodulated amplitude(s). Shape (nsignals,) if 2D input, scalar if 1D.
    phase : float or ndarray
        Demodulated phase(s) in radians. Shape (nsignals,) if 2D input, scalar if 1D.
    
    Examples
    --------
    # Single signal
    >>> amp, phase = demodulate_signal(signal_1d, 5.0, 1000.0)
    
    # Multiple signals (vectorized)
    >>> amps, phases = demodulate_signal(signals_2d, 5.0, 1000.0)
    >>> # signals_2d.shape = (nt, nslopes)
    >>> # amps.shape = (nslopes,)
    >>> # phases.shape = (nslopes,)
    
    Notes
    -----
    The cumulated method computes demodulation over increasing time windows,
    providing more stable estimates as more data accumulates.
    
    References
    ----------
    - PASSATA demodulate_passata.pro
    - LBT-SOUL calibration software (2020)
    """
    # Convert to array
    data = xp.asarray(signal_data, dtype=dtype)

    # Handle 1D vs 2D input
    is_1d = (data.ndim == 1)
    if is_1d:
        data = data[:, xp.newaxis]  # Shape: (nt, 1)

    nt, nsignals = data.shape

    # Time parameters
    dt = 1.0 / sampling_freq
    t = xp.arange(nt, dtype=dtype) * dt
    w = 2 * xp.pi * carrier_freq

    # Calculate n4mean (averaging window at end of signal)
    periods = int(xp.floor(xp.max(t) * carrier_freq))
    if periods > 0:
        test_vect = (xp.arange(periods) + 1) * sampling_freq / carrier_freq
        errors = xp.abs(test_vect - xp.round(test_vect))
        idx = xp.where(errors <= 1e-3)[0]
        if len(idx) > 0:
            n4mean = int(test_vect[xp.max(idx)])
        else:
            n4mean = int(test_vect[xp.argmin(errors)])
    else:
        n4mean = max(1, nt // 4)

    # Linear detrend (vectorized across all signals)
    data_mean = xp.mean(data, axis=0, keepdims=True)  # Shape: (1, nsignals)
    cur_data = data - data_mean

    # Tilt per signal
    tilt = (cur_data[-1:, :] - cur_data[0:1, :]) / nt  # Shape: (1, nsignals)
    t_ramp = xp.arange(nt, dtype=dtype)[:, xp.newaxis]  # Shape: (nt, 1)
    cur_data = cur_data - tilt * t_ramp - cur_data[0:1, :]

    # Find phase with reference carrier (vectorized)
    sin_carrier = xp.sin(w * t)[:, xp.newaxis]  # Shape: (nt, 1)
    cos_carrier = xp.cos(w * t)[:, xp.newaxis]  # Shape: (nt, 1)

    qa_ref = xp.mean(cur_data * sin_carrier, axis=0)  # Shape: (nsignals,)
    pa_ref = xp.mean(cur_data * cos_carrier, axis=0)  # Shape: (nsignals,)
    pphi0 = xp.arctan2(qa_ref, pa_ref)  # Shape: (nsignals,)

    # Generate phased carriers (broadcast to all signals)
    dem_sin = xp.sin(w * t[:, xp.newaxis] \
              - pphi0[xp.newaxis, :])  # Shape: (nt, nsignals)
    dem_cos = xp.cos(w * t[:, xp.newaxis] \
              - pphi0[xp.newaxis, :])  # Shape: (nt, nsignals)

    if cumulated:
        # Cumulated demodulation with progressive windows
        qa = xp.zeros((nt, nsignals), dtype=xp.float32)
        pa = xp.zeros((nt, nsignals), dtype=xp.float32)

        for j in range(2, nt):
            # Window from start to j (all signals at once)
            window_data = data[:j+1, :] - xp.mean(data[:j+1, :], axis=0, keepdims=True)
            window_tilt = (window_data[j:j+1, :] - window_data[0:1, :]) / j
            t_window = xp.arange(j+1, dtype=dtype)[:, xp.newaxis]
            window_data = window_data - window_tilt * t_window - window_data[0:1, :]

            qa[j, :] = xp.sum(window_data * dem_sin[:j+1, :], axis=0) / (j + 1)
            pa[j, :] = xp.sum(window_data * dem_cos[:j+1, :], axis=0) / (j + 1)

        # Compute amplitude and phase time series
        data_dem_temp = 2.0 * xp.sqrt(qa[2:, :]**2 + pa[2:, :]**2)
        pphi_temp = xp.arctan2(qa[2:, :], pa[2:, :])

        # Average over last n4mean samples
        start_idx = max(0, nt - 2 - n4mean)
        end_idx = nt - 2

        if end_idx > start_idx:
            value = xp.mean(data_dem_temp[start_idx:end_idx, :],
                                   axis=0)  # Shape: (nsignals,)
            pphi = xp.mean(pphi_temp[start_idx:end_idx, :],
                                  axis=0)  # Shape: (nsignals,)
        else:
            value = data_dem_temp[-1, :] if len(data_dem_temp) > 0 else xp.zeros(nsignals)
            pphi = pphi_temp[-1, :] if len(pphi_temp) > 0 else xp.zeros(nsignals)
    else:
        # Simple demodulation (single pass, vectorized)
        qa = xp.mean(cur_data * dem_sin, axis=0)  # Shape: (nsignals,)
        pa = xp.mean(cur_data * dem_cos, axis=0)  # Shape: (nsignals,)
        pphi = xp.arctan2(qa, pa)  # Shape: (nsignals,)
        value = 2.0 * xp.sqrt(qa**2 + pa**2)  # Shape: (nsignals,)

    # Add reference phase
    pphi = pphi + pphi0

    if verbose:
        print(f"Demodulation results:")
        print(f"  Number of signals: {nsignals}")
        print(f"  Amplitude range: [{float(xp.min(value)):.6e},"
              f" {float(xp.max(value)):.6e}]")
        print(f"  Phase range: [{float(xp.min(pphi)):.6f},"
              f" {float(xp.max(pphi)):.6f}] rad")
        print(f"  Carrier freq: {carrier_freq} Hz")
        print(f"  Sampling freq: {sampling_freq} Hz")
        print(f"  n4mean: {n4mean}")
        print(f"  Data points: {nt}")

    # Convert to CPU arrays and return scalar if input was 1D
    if is_1d:
        return float(value[0]), float(pphi[0])
    else:
        return value, pphi
