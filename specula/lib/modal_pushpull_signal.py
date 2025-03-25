#+
# NAME:
#   modal_pushpull_signal
# PURPOSE:
#   generate a modal push-pull time history to be use for calibration purpose
# CATEGORY:
#   AO simulation.
# CALLING SEQUENCE:
# function modal_pushpull_signal, n_modes, amplitude=amplitude, vect_amplitude=vect_amplitude
# INPUTS:
#   n_modes         number of modes
# KEYWORD:
#   amplitude	    amplitude of mode 0
#   vect_amplitude  modal amplitude vector
#   linear          vect_amplitude change as 1/rad_order
#   min_amplitude   min value for vect_amplitude
#   only_push       makes an only push signal
#   ncycles         number of cycle of push-pull
#   repeat_ncycles   set it to have ncycles of push and then ncycles of pull
# OUTPUTS:
#   time_hist       modal time history
# COMMON BLOCKS:
#   None.
# SIDE EFFECTS:
#   None.
# RESTRICTIONS:
#   None
# MODIFICATION HISTORY:
#   Created 12-SEP-2014 by Guido Agapito guido.agapito@inaf.it
#-
import numpy as np
from specula.lib.zernike_generator import ZernikeGenerator

def modal_pushpull_signal(n_modes, amplitude=None, vect_amplitude=None,
                                linear=None, min_amplitude=None, only_push=False,
                                ncycles=1, repeat_ncycles=False, xp=np):

    if vect_amplitude is None:
        radorder = xp.array([ZernikeGenerator.degree(x)[0] for x in xp.arange(n_modes)+2])
        if linear:
            vect_amplitude = amplitude/radorder
        else:
            vect_amplitude = amplitude/np.sqrt(radorder)
        if min_amplitude is not None:
            vect_amplitude = xp.minimum(vect_amplitude, min_amplitude)

    if only_push:
        time_hist = xp.zeros((n_modes * ncycles, n_modes))
        for i in range(n_modes):
            for j in range(ncycles):
                time_hist[ncycles*i+j, i] = vect_amplitude[i]     
    else:
        if repeat_ncycles:
            time_hist = xp.zeros((2*n_modes*ncycles, n_modes))
            for i in range(n_modes):
                time_hist[2*i*ncycles:2*(i+1)*ncycles, i] = \
                    xp.concatenate((xp.repeat(vect_amplitude[i], ncycles), xp.repeat(-vect_amplitude[i], ncycles)))
        else:
            time_hist =xp.zeros((2*n_modes*ncycles,n_modes))
            for i in range(n_modes):
                for j in range(ncycles):
                    time_hist[2*(ncycles*i+j):2*(ncycles*i+j)+2,i] = xp.array([vect_amplitude[i],-vect_amplitude[i]])

    return time_hist
