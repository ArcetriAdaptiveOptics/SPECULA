from specula import fuse


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)


def pyr_compute_slopes(frame, ind_pup, SHLIKE=False, INTENSITY_BASED=False, norm_fact=None, threshold=None, xp=None, float_dtype=None):
    """
    Computes the pyramid signals from a CCD frame.
    
    Parameters:
        frame (numpy.ndarray): CCD frame containing the four sub-pupils.
        ind_pup (numpy.ndarray): 2D array with pupil indices.
        SHLIKE (bool): If True, normalize signals to each subaperture's flux.
        INTENSITY_BASED (bool): If True, compute slopes based on intensity.
        norm_fact (float): Normalize signals to this constant value if provided.
        threshold (float): Subtract this threshold from all pixel values.
    
    Returns:
        sx (numpy.ndarray): x-signal vector.
        sy (numpy.ndarray): y-signal vector.
        flux (float): Total intensity.
    """
    
    if INTENSITY_BASED and SHLIKE:
        raise ValueError('INTENSITY_BASED and SHLIKE keywords cannot be set together.')

    if INTENSITY_BASED and SHLIKE:
        raise ValueError('INTENSITY_BASED and SHLIKE keywords cannot be set together.')

    A = frame.flatten()[ind_pup[:, 0]]
    B = frame.flatten()[ind_pup[:, 1]]
    C = frame.flatten()[ind_pup[:, 2]]
    D = frame.flatten()[ind_pup[:, 3]]
    # Extract intensity arrays for each sub-pupil
    
    # Compute total intensity
    flux = xp.sum(A+B+C+D)

    if threshold is not None:
        A -= threshold
        clamp_generic_less(0,0,A, xp=xp)
        B -= threshold
        clamp_generic_less(0,0,B, xp=xp)
        C -= threshold
        clamp_generic_less(0,0,C, xp=xp)
        D -= threshold
        clamp_generic_less(0,0,D, xp=xp)

    per_subap_sum = A+B+C+D
    total_intensity = xp.sum(per_subap_sum)
    clamp_generic_less(0, 0, total_intensity, xp=xp)
 
    if norm_fact is not None:
        factor = 1.0 / norm_fact
    elif INTENSITY_BASED:
        n_subap = ind_pup.shape[0]
        factor = 4 * n_subap / total_intensity
        sx = factor * xp.concatenate([A, B])
        sy = factor * xp.concatenate([C, D])
    else:
        if not SHLIKE:
            n_subap = ind_pup.shape[0]
            factor = n_subap / total_intensity
        else:
            inv_factor = per_subap_sum                
            clamp_generic_less(0, 1e-6, inv_factor)
            factor = 1.0 / inv_factor
            clamp_generic_less(0,0, factor)
            
        sx = (A+B-C-D).astype(float_dtype) * factor
        sy = (B+C-A-D).astype(float_dtype) * factor

    clamp_generic_more(0, 1, total_intensity, xp=xp)

    return sx*total_intensity, sy*total_intensity, flux
