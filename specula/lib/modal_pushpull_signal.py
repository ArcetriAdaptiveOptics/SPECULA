import numpy as np
from typing import Optional, Sequence

from specula.lib.zernike_generator import ZernikeGenerator

def modal_pushpull_signal(
    n_modes: int,
    first_mode: Optional[int] = 0,
    amplitude: Optional[float] = None,
    vect_amplitude: Optional[Sequence[float]] = None,
    linear: bool = False,
    constant: bool = False,
    min_amplitude: Optional[float] = None,
    only_push: bool = False,
    pattern: Sequence[float] = [1, -1],
    ncycles: int = 1,
    repeat_ncycles: bool = False,
    nsamples: int = 1,
    xp=np,
) -> np.ndarray:
    """
    Generate a modal push-pull time history for calibration purposes.

    This function generates a modal push-pull time series to be used in
    adaptive optics (AO) simulations. The output represents a modal time
    history that can be used for calibration and testing.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    first_mode: int, optional
        First mode to actuate. Default to zero.
    amplitude : float, optional
        Amplitude of mode 0. By default it will be rescaled as 1/sqrt(rad_order)
    vect_amplitude : sequence of float, optional
        Vector of modal amplitudes. 
    linear : bool, optional
        If True, `vect_amplitude` changes as ``1/rad_order`` instead of
        ``1/sqrt(rad_order)``. Default is False.
    constant : bool, optional
        If True, `vect_amplitude` is constant across all modes. Default is False.
    min_amplitude : float, optional
        Minimum value for `vect_amplitude`. Default is None.
    only_push : bool, optional
        If True, generates a signal with only positive pushes. Default is False.
    pattern : sequence of float, optional
        Push-pull pattern. Default is ``[-1, 1]``, but can be any sequence of numbers.
    ncycles : int, optional
        Number of push-pull cycles. Default is 1.
    repeat_ncycles : bool, optional
        If True, generates `ncycles` of push followed by `ncycles` of pull.
        Default is False.
    nsamples : int, optional
        Number of samples to hold in each position. Default is 1.

    Returns
    -------
    time_hist : np.ndarray
        Modal time history signal.

    Notes
    -----
    This function is primarily used for calibration in AO simulations.

    Examples
    --------
    >>> time_hist = modal_pushpull_signal(
    ...     n_modes=5,
    ...     amplitude=0.1,
    ...     pattern=[-1, 1],
    ...     ncycles=3,
    ...     nsamples=2
    ... )

    History
    -------
    Created on 12-SEP-2014 by Guido Agapito (guido.agapito@inaf.it)
    2025-08-29 by Alfio Puglisi (alfio.puglisi@inaf.it): added "pattern", "first mode" and "constant" parameters
    """
    if only_push:
        pattern = [1]

    if vect_amplitude is None:
        radorder = xp.array([ZernikeGenerator.degree(x)[0] for x in xp.arange(first_mode, n_modes) + 2])
        if linear:
            vect_amplitude = amplitude/radorder
        elif constant:
            vect_amplitude = xp.repeat(amplitude, len(radorder))
        else:
            vect_amplitude = amplitude/xp.sqrt(radorder)
        if min_amplitude is not None:
            vect_amplitude = xp.minimum(vect_amplitude, min_amplitude)

    # Prepend zero values equal to the number of skipped modes
    vect_amplitude = xp.hstack((
        xp.repeat(0, first_mode), vect_amplitude
    ))

    n_pokes = len(pattern)

    real_n_modes = n_modes - first_mode
    time_hist = xp.zeros((n_pokes * real_n_modes * ncycles, n_modes))
    for mode in range(first_mode, n_modes):
        hist_idx = mode - first_mode
        poke_pattern = vect_amplitude[mode] * xp.array(pattern)
        if repeat_ncycles:
            time_hist[n_pokes*hist_idx*ncycles:n_pokes*(hist_idx+1)*ncycles, mode] = \
                xp.repeat(poke_pattern, ncycles)
        else:
            for j in range(ncycles):
                time_hist[n_pokes*(ncycles*hist_idx+j):n_pokes*(ncycles*hist_idx+j+1), mode] = poke_pattern

    return xp.repeat(time_hist, nsamples, axis=0)
