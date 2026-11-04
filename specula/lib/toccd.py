# -*- coding: utf-8 -*-
#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-28  Created
#
#########################################################

import functools

import numpy as np
import scipy.sparse

import specula
from specula import cpuArray


def toccd(a, newshape, set_total=None, xp=None, out=None):
    '''
    Clone of oaalib's toccd() function: rebin an array by area weighting,
    similar to openvc's INTER_AREA interpolation.

    Each output pixel is the mean of the input over its area, with the input
    pixels partially covered weighted by the covered fraction (see
    overlap_weights()). The rebinning is separable: on CPU, it is computed with
    two sparse matrix products, and on GPU with a single elementwise kernel.

    Parameters
    ----------
    a : array
        array to be resized
    newshape : tuple, list or array
        shape of resized array
    set total : float, optional
        if set and strictly positive, normalize the resized array to this total count.
        if set and zero or negative, the resized array is not renormalized.
        if not set, the same total count as the input array is used.
    xp : module
        numpy or cupy module
    out : array, optional
        output array (or view) with shape newshape. On GPU, it must have the same
        dtype as a, and it is written directly by the kernel.

    Returns
    -------
    array
        resized array (out, if given)
    '''
    newshape = tuple(int(n) for n in cpuArray(newshape))  # Works for lists, tuples and any cupy/numpy array

    if out is not None and out.shape != newshape:
        raise ValueError(f'out has shape {out.shape} instead of {newshape}')

    if a.shape == newshape:
        if out is None:
            return a
        out[:] = a
        return out

    if len(a.shape) != 2:
        raise ValueError('Input array has shape %s, cannot continue' % str(a.shape))

    if len(newshape) != 2:
        raise ValueError('Output shape is %s, cannot continue' % str(newshape))

    cp = specula.cp
    if cp is not None and xp is cp:  # pragma: gpu
        if a.dtype not in (cp.float32, cp.float64):
            raise TypeError(f'toccd(): unsupported dtype {a.dtype} on GPU.'
                            f' Valid dtypes are float32 and float64')
        if out is None:
            out = cp.empty(newshape, dtype=a.dtype)
        elif out.dtype != a.dtype:
            raise TypeError(f'toccd(): out has dtype {out.dtype} instead of {a.dtype}')
        ix, wx, iy, wy = _device_weights(a.shape, newshape, a.dtype, cp.cuda.Device().id)
        _toccd_kernel()(a, ix, wx, iy, wy, a.shape[1], newshape[1], ix.shape[1], iy.shape[1], out)
    else:
        dtype = np.result_type(a.dtype, np.float32)
        wy = _sparse_weights(a.shape[0], newshape[0], dtype)
        wx = _sparse_weights(a.shape[1], newshape[1], dtype)
        rebinned = (wx @ (wy @ a).T).T
        if out is None:
            out = np.ascontiguousarray(rebinned)
        else:
            out[:] = rebinned

    eps = xp.finfo(out.dtype).eps
    if set_total is None:
        out *= a.sum() / xp.maximum(out.sum(), eps)
    elif set_total > 0:
        out *= set_total / xp.maximum(out.sum(), eps)
    return out


@functools.lru_cache(maxsize=None)
def overlap_weights(n_in, n_out):
    '''
    Area weights to rebin an axis from n_in to n_out pixels.

    Output pixel o covers the input interval [o * n_in / n_out, (o + 1) * n_in / n_out)
    and its value is sum_k in[idx[o, k]] * w[o, k], where w is the covered
    length of each input pixel divided by n_in. This is the mean of the L.C.M.
    upsampled input used by oaalib's toccd(), without building it: the interval
    boundaries are integers in units of 1 / n_out input pixels, so the weights are exact.

    Returns
    -------
    idx : int ndarray (n_out, K)
        input pixel indices (padded with weight zero)
    w : float64 ndarray (n_out, K)
        weights
    '''
    start = np.arange(n_out) * n_in
    end = start + n_in
    first = start // n_out
    last = (end - 1) // n_out
    k = int((last - first).max()) + 1
    idx = first[:, None] + np.arange(k)[None, :]
    covered = np.minimum(end[:, None], (idx + 1) * n_out) - np.maximum(start[:, None], idx * n_out)
    w = np.maximum(covered, 0) / n_in
    return np.minimum(idx, n_in - 1), w


@functools.lru_cache(maxsize=None)
def _sparse_weights(n_in, n_out, dtype):
    '''Weights of overlap_weights() as a (n_out, n_in) sparse matrix'''
    idx, w = overlap_weights(n_in, n_out)
    rows = np.repeat(np.arange(n_out), idx.shape[1])
    return scipy.sparse.csr_matrix((w.ravel().astype(dtype), (rows, idx.ravel())),
                                   shape=(n_out, n_in))


@functools.lru_cache(maxsize=None)
def _device_weights(in_shape, out_shape, dtype, device_id):  # pragma: gpu
    '''Weights of overlap_weights() for both axes on the current GPU'''
    cp = specula.cp
    ix, wx = overlap_weights(in_shape[1], out_shape[1])
    iy, wy = overlap_weights(in_shape[0], out_shape[0])
    return (cp.asarray(ix, dtype=cp.int32), cp.asarray(wx, dtype=dtype),
            cp.asarray(iy, dtype=cp.int32), cp.asarray(wy, dtype=dtype))


@functools.cache
def _toccd_kernel():  # pragma: gpu
    '''
    The toccd elementwise kernel, created at the first GPU call,
    so that this module can be imported before init()
    '''
    # out[y, x] = sum_j sum_k a[iy[y, j], ix[x, k]] * wy[y, j] * wx[x, k]
    return specula.cp.ElementwiseKernel(
        'raw T a, raw int32 ix, raw T wx, raw int32 iy, raw T wy,'
        ' int32 inx, int32 outx, int32 kx, int32 ky',
        'T out',
        '''
        int idx = i;  // 32-bit division is much faster than the 64-bit one
        int y = idx / outx;
        int x = idx - y * outx;
        T res = 0;
        for (int j = 0; j < ky; j++) {
            int row = iy[y * ky + j] * inx;
            T rowsum = 0;
            for (int k = 0; k < kx; k++)
                rowsum += a[row + ix[x * kx + k]] * wx[x * kx + k];
            res += rowsum * wy[y * ky + j];
        }
        out = res;
        ''',
        'toccd_kernel')
