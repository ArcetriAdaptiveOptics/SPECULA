"""
Signal demodulation utilities.
Based on demodulate_passata.pro from PASSATA/LBT-SOUL.
"""
from specula import xp


def demodulate_signal(signal_data, carrier_freq, sampling_freq, 
                     cumulated=True, verbose=False, xp_module=None):
    """
    Demodulate a signal using a carrier frequency.
    
    This implements the demodulation algorithm from demodulate_passata.pro,
    with cumulative averaging and linear detrending.
    
    Parameters
    ----------
    signal_data : array_like
        Input signal time series
    carrier_freq : float
        Carrier frequency in Hz
    sampling_freq : float
        Sampling frequency in Hz
    cumulated : bool, optional
        If True, use cumulative demodulation averaging. Default: True
    verbose : bool, optional
        Print debug information. Default: False
    xp_module : module, optional
        Array module (numpy or cupy). If None, uses specula.xp
    
    Returns
    -------
    value : float
        Demodulated amplitude
    phase : float
        Demodulated phase (radians)
    
    Notes
    -----
    The cumulated method computes demodulation over increasing time windows,
    providing more stable estimates as more data accumulates.
    
    References
    ----------
    - PASSATA demodulate_passata.pro
    - LBT-SOUL calibration software (2020)
    """
    if xp_module is None:
        xp_module = xp

    # Convert to array
    data = xp_module.asarray(signal_data, dtype=xp_module.float32)

    # Time parameters
    dt = 1.0 / sampling_freq
    nt = len(data)
    t = xp_module.arange(nt, dtype=xp_module.float32) * dt

    w = 2 * xp_module.pi * carrier_freq

    # Calculate N4mean (averaging window at end of signal)
    periods = int(xp_module.floor(xp_module.max(t) * carrier_freq))
    if periods > 0:
        testVect = (xp_module.arange(periods) + 1) * sampling_freq / carrier_freq
        errors = xp_module.abs(testVect - xp_module.round(testVect))
        idx = xp_module.where(errors <= 1e-3)[0]
        if len(idx) > 0:
            N4mean = int(testVect[xp_module.max(idx)])
        else:
            N4mean = int(testVect[xp_module.argmin(errors)])
    else:
        N4mean = max(1, nt // 4)

    # Linear detrend
    cur_data = data - xp_module.mean(data)
    tilt = (cur_data[-1] - cur_data[0]) / nt
    cur_data = cur_data - tilt * xp_module.arange(nt, dtype=xp_module.float32) - cur_data[0]

    # Find phase with reference carrier
    Qa_ref = xp_module.mean(cur_data * xp_module.sin(w * t))
    Pa_ref = xp_module.mean(cur_data * xp_module.cos(w * t))
    pphi0 = xp_module.arctan2(Qa_ref, Pa_ref)

    # Generate phased carriers
    dem_sin = xp_module.sin(w * t - pphi0)
    dem_cos = xp_module.cos(w * t - pphi0)

    if cumulated:
        # Cumulated demodulation with progressive windows
        Qa = xp_module.zeros(nt, dtype=xp_module.float32)
        Pa = xp_module.zeros(nt, dtype=xp_module.float32)

        for j in range(2, nt):
            # Window from start to j
            window_data = data[:j+1] - xp_module.mean(data[:j+1])
            window_tilt = (window_data[j] - window_data[0]) / j
            window_data = (window_data 
                          - window_tilt * xp_module.arange(j+1, dtype=xp_module.float32) 
                          - window_data[0])

            Qa[j] = xp_module.sum(window_data * dem_sin[:j+1]) / (j + 1)
            Pa[j] = xp_module.sum(window_data * dem_cos[:j+1]) / (j + 1)

        # Compute amplitude and phase time series
        data_dem_temp = 2.0 * xp_module.sqrt(Qa[2:]**2 + Pa[2:]**2)
        pphi_temp = xp_module.arctan2(Qa[2:], Pa[2:])

        # Average over last N4mean samples
        start_idx = max(0, nt - 2 - N4mean)
        end_idx = nt - 2

        if end_idx > start_idx:
            value = float(xp_module.mean(data_dem_temp[start_idx:end_idx]))
            pphi = float(xp_module.mean(pphi_temp[start_idx:end_idx]))
        else:
            value = float(data_dem_temp[-1]) if len(data_dem_temp) > 0 else 0.0
            pphi = float(pphi_temp[-1]) if len(pphi_temp) > 0 else 0.0
    else:
        # Simple demodulation (single pass)
        Qa = xp_module.mean(cur_data * dem_sin)
        Pa = xp_module.mean(cur_data * dem_cos)
        pphi = xp_module.arctan2(Qa, Pa)
        value = 2.0 * xp_module.sqrt(Qa**2 + Pa**2)

        value = float(value)
        pphi = float(pphi)

    # Add reference phase
    pphi += float(pphi0)

    if verbose:
        print(f"Demodulation results:")
        print(f"  Amplitude: {value:.6e}")
        print(f"  Phase: {pphi:.6f} rad")
        print(f"  Carrier freq: {carrier_freq} Hz")
        print(f"  Sampling freq: {sampling_freq} Hz")
        print(f"  N4mean: {N4mean}")
        print(f"  Data points: {nt}")

    return value, pphi
