# -*- coding: utf-8 -*-

#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-20  Created
#
#########################################################

def rebin2d(a, shape, sample=False, xp=None):
    '''
    Replacement of IDL's rebin() function for 2d arrays.

    Parameters
    ----------
    a : array-like
        The input array to be rebinned.
    shape : tuple of int
        The desired shape of the output array.
    sample : bool, optional
        If True, the output array is sampled at the new shape.
    xp : module, optional
        Array module to use (default: numpy).

    Returns
    -------
    array-like
        The rebinned array.

    Raises
    ------
    ValueError
        If the resampling factors are not integers.
        If upsampling with sample=False is attempted.
        If upsampling and downsampling in different axes is attempted.
    '''
    if a.shape == shape:
        return a

    if sample:
        slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,shape) ]
        idx = xp.mgrid[slices].astype(xp.int32)
        return a[tuple(idx)]

    else:
        M, N = a.shape
        m, n = shape

        if m<=M and n<=N:
            if (M//m != M/m) or (N//n != N/n):
                raise ValueError('Resampling by non-integer factors is not supported')
            # Same as a.reshape((m,M//m,n,N//n)).mean(3).mean(1), but summing
            # strided slices: cupy is very slow at reducing a short last axis
            fy, fx = M//m, N//n
            dtype = a.dtype if xp.issubdtype(a.dtype, xp.inexact) else xp.float64
            cols = a[:, 0::fx].astype(dtype)
            for j in range(1, fx):
                cols += a[:, j::fx]
            if fx > 1:
                cols /= fx
            out = cols[0::fy].copy()
            for i in range(1, fy):
                out += cols[i::fy]
            if fy > 1:
                out /= fy
            return out
        elif m>=M and n>=N:
            raise ValueError('Upsampling with sample=False is not supported')
        else:
            raise ValueError('Upsampling and downsampling in different axes it not supported')
