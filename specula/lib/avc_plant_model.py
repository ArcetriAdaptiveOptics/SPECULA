"""
Analytic closed-loop plant model for AVC calibration.

Single-mode AO loop, evaluated at ``z = exp(2j*pi*f*T)``::

    HOL   = WFS * RTC * CTR * DAC * DM * LPF
    S     = 1 / (1 + HOL)
    P_avc = -WFS * RTC * delAVC * DAC * DM * LPF * S

``P_avc`` is the response from an AVC command to the measured residual,
i.e. the plant calibration (``plant_re``, ``plant_im``) of
:py:class:`~specula.processing_objects.avc.AVC`. The minus sign is the one
of the loop's negative feedback: the AVC command is summed to the
integrator one, which the DM subtracts from the disturbance.

Ported from ESO's ``generateAVCInitConditions.m``, which has the opposite
sign for ``P_avc``.
"""
import numpy as np


def _as_freq_array(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    if np.any(f <= 0):
        raise ValueError("frequencies must be > 0 Hz")
    return f


def _zoh(f, T):
    """Zero-order hold over one sample period T (WFS integration, DAC)."""
    w = 2 * np.pi * f
    return (1 - np.exp(-1j * w * T)) / (1j * w * T)


def _dm_response(f, dm_dc_gain, dm_f0, dm_csi):
    if dm_f0 is None:
        return np.full(f.shape, dm_dc_gain, dtype=complex)
    w = 2 * np.pi * f
    w0 = 2 * np.pi * dm_f0
    return dm_dc_gain / ((1j * w)**2 / w0**2 + 2 * (1j * w) * dm_csi / w0 + 1)


def _lpf_response(f, fc, T):
    """First-order low-pass, matching
    ``IirFilterData.lpf_from_fc(fc, fs=1/T, n_ord=1)``."""
    w = 2 * np.pi * f
    omega = np.tan(np.pi * fc * T)
    a0 = omega / (1 + omega)
    z_inv = np.exp(-1j * w * T)
    return a0 / (1 - (1 - a0) * z_inv)


def open_loop_transfer_function(f, T, loop_gain, loop_delay,
                                dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                                lpf_fc=None):
    """Open-loop transfer function ``HOL(f)`` of a single-mode AO loop.

    Parameters
    ----------
    f : float or array-like [Hz]
        Frequencies, must be > 0.
    T : float [s]
        Loop sampling period.
    loop_gain : float
        Integrator gain of the mode.
    loop_delay : float [s]
        RTC delay (``Integrator.delay * T``).
    dm_dc_gain : float, optional
        DM DC gain. Default 1.0.
    dm_f0, dm_csi : float, optional
        Natural frequency [Hz] and damping ratio of a 2nd-order DM
        resonance. Default: no DM dynamics.
    lpf_fc : float [Hz], optional
        Cutoff of a first-order low-pass in the command path. Default: none.

    Returns
    -------
    HOL : ndarray, complex
    """
    f = _as_freq_array(f)
    w = 2 * np.pi * f
    wfs = _zoh(f, T)
    dac = _zoh(f, T)
    ctr = loop_gain / (1 - np.exp(-1j * w * T))
    rtc = np.exp(-1j * w * loop_delay)
    dm = _dm_response(f, dm_dc_gain, dm_f0, dm_csi)
    lpf = 1.0 if lpf_fc is None else _lpf_response(f, lpf_fc, T)
    return wfs * rtc * ctr * dac * dm * lpf


def closed_loop_sensitivity(f, T, loop_gain, loop_delay,
                            dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                            lpf_fc=None):
    """Closed-loop sensitivity ``S(f) = 1 / (1 + HOL(f))``. Parameters as
    in :py:func:`open_loop_transfer_function`."""
    hol = open_loop_transfer_function(f, T, loop_gain, loop_delay,
                                      dm_dc_gain, dm_f0, dm_csi, lpf_fc)
    return 1.0 / (1.0 + hol)


def avc_command_to_residual_tf(f, T, loop_gain, loop_delay,
                               avc_extra_delay=0.0,
                               dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                               lpf_fc=None):
    """Transfer function ``P_avc(f)`` from an AVC command, summed to the
    main loop's DM command, to the measured residual.

    Parameters
    ----------
    avc_extra_delay : float [s], optional
        Delay of the AVC command path in excess of ``loop_delay``.
        Negative if the AVC command skips part of it, e.g. when it is
        summed downstream of a delayed integrator. Default 0.0.

    Other parameters as in :py:func:`open_loop_transfer_function`.

    Returns
    -------
    P_avc : ndarray, complex
    """
    f = _as_freq_array(f)
    w = 2 * np.pi * f
    wfs = _zoh(f, T)
    dac = _zoh(f, T)
    rtc = np.exp(-1j * w * loop_delay)
    del_avc = np.exp(-1j * w * avc_extra_delay)
    dm = _dm_response(f, dm_dc_gain, dm_f0, dm_csi)
    lpf = 1.0 if lpf_fc is None else _lpf_response(f, lpf_fc, T)
    s = closed_loop_sensitivity(f, T, loop_gain, loop_delay,
                                dm_dc_gain, dm_f0, dm_csi, lpf_fc)
    return -wfs * rtc * del_avc * dac * dm * lpf * s


def generate_avc_init_conditions(freq_avc, T, loop_gain, loop_delay,
                                 avc_extra_delay=0.0,
                                 dm_dc_gain=1.0, dm_f0=None, dm_csi=None,
                                 lpf_fc=None):
    """Plant calibration for :py:class:`~specula.processing_objects.avc.AVC`:
    :py:func:`avc_command_to_residual_tf` evaluated at ``freq_avc`` [Hz].

    Returns
    -------
    plant_re, plant_im : float
        Real and imaginary part of ``P_avc(freq_avc)``, or ``(1.0, 0.0)``
        if it is zero.
    """
    p_avc = avc_command_to_residual_tf(freq_avc, T, loop_gain, loop_delay,
                                       avc_extra_delay, dm_dc_gain,
                                       dm_f0, dm_csi, lpf_fc)
    value = complex(p_avc[0])
    if abs(value) == 0:
        return 1.0, 0.0
    return float(value.real), float(value.imag)
