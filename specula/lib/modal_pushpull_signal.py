import numpy as np
from typing import Optional, Sequence

from specula.lib.zernike_generator import ZernikeGenerator

def modal_pushpull_amplitudes(
    n_modes: int,
    first_mode: Optional[int] = 0,
    amplitude: Optional[float] = None,
    vect_amplitude: Optional[Sequence[float]] = None,
    linear: bool = False,
    constant: bool = False,
    min_amplitude: Optional[float] = None,
    xp=np,
) -> np.ndarray:
    """
    Compute the per-mode push-pull amplitudes used by `PushPullGenerator`.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    first_mode : int, optional
        First mode to actuate. Default to zero.
    amplitude : float, optional
        Amplitude of mode 0. By default it will be rescaled as 1/sqrt(rad_order)
    vect_amplitude : sequence of float, optional
        Vector of modal amplitudes for modes `first_mode` to `n_modes`-1.
        If given, `amplitude`, `linear`, `constant` and `min_amplitude` are ignored.
    linear : bool, optional
        If True, `vect_amplitude` changes as ``1/rad_order`` instead of
        ``1/sqrt(rad_order)``. Default is False.
    constant : bool, optional
        If True, `vect_amplitude` is constant across all modes. Default is False.
    min_amplitude : float, optional
        Minimum value for `vect_amplitude`. Default is None.
    xp : module, optional
        Array module to use (e.g., numpy or cupy). Default is numpy.

    Returns
    -------
    vect_amplitude : np.ndarray
        Amplitudes of length `n_modes`, zero for the first `first_mode` modes.

    History
    -------
    Created on 12-SEP-2014 by Guido Agapito (guido.agapito@inaf.it), as part of modal_pushpull_signal
    2025-08-29 by Alfio Puglisi (alfio.puglisi@inaf.it): added "first mode" and "constant" parameters
    """
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
    return xp.hstack((
        xp.repeat(0, first_mode), vect_amplitude
    ))
