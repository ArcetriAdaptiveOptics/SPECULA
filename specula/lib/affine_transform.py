import inspect

from specula import cp, np

if cp:  # pragma: no cover
    from cupyx.scipy.ndimage import affine_transform as _cupy_affine_transform
    # cupy < 14.1 always uses float64 coordinates and has no float64_coords argument
    _COORDS_KW = ({'float64_coords': True}
                if 'float64_coords' in inspect.signature(_cupy_affine_transform).parameters else {})

# Number of output elements computed at once by the generic CPU interpolation:
# small temporaries stay in cache, and large memory allocations are avoided.
_CHUNK_SIZE = 16384


def affine_transform(data, matrix, output, xp):
    """Linear (order=1) affine transformation of a 2D array, zero outside the input.

    Same result as ndimage.affine_transform(data, matrix, output=output, order=1,
    mode='constant'): output[o] = data[matrix[:, :2] @ o + matrix[:, 2]] with
    bilinear interpolation, and zero where that coordinate is outside [0, n-1]
    on either axis of data.

    On GPU, calls cupyx.scipy.ndimage.affine_transform with double precision
    coordinates. matrix is read from device memory, so the call can be captured
    in a CUDA graph.
    On CPU, if matrix[:, :2] is a signed permutation (shifts, rot90(), flips) and
    all samples are inside data, the result is a weighted sum of up to four
    strided views of data. Otherwise, a bilinear interpolation is computed in
    chunks of output elements. Both are faster than scipy.ndimage.affine_transform.

    Parameters
    ----------
    data : 2D array
        Input array
    matrix : array (2, 3)
        [A | offset]: the data coordinates of output element o are A @ o + offset
    output : 2D array
        Output array (written in place), with the same dtype as data
    xp : module
        numpy or cupy
    """
    if xp is not np:
        _cupy_affine_transform(data, matrix, output=output, output_shape=output.shape,
                               order=1, **_COORDS_KW)
        return
    linear, offset = matrix[:, :2], matrix[:, 2]
    if not (_is_signed_permutation(linear) and _permuted(data, linear, offset, output)):
        _bilinear(data, linear, offset, output)


def _is_signed_permutation(linear):
    """True if the 2x2 matrix has exactly one element equal to +1 or -1 in each
    row and column, and zeros elsewhere"""
    mag = np.abs(linear)
    return bool(np.all((linear == 0) | (mag == 1))
                and np.all(mag.sum(axis=0) == 1) and np.all(mag.sum(axis=1) == 1))


def _axis_slice(start, step, length):
    """Slice of length elements from start with step +1 or -1"""
    stop = start + step * length
    return slice(start, stop if stop >= 0 else None, step)


def _permuted(data, linear, offset, output):
    """Signed permutation matrix: weighted sum of up to four strided views of data.

    Returns False, without writing output, if some samples fall outside data.
    """
    swap = linear[0, 0] == 0
    axes = []
    for k in range(2):
        out_axis = 1 if linear[k, 1] != 0 else 0
        step = int(linear[k, out_axis])
        length = output.shape[out_axis]
        first = int(np.floor(offset[k]))
        frac = offset[k] - first
        last = first + step * (length - 1)
        if min(first, last) < 0 or max(first, last) + (frac != 0) > data.shape[k] - 1:
            return False
        axes.append((first, step, length, frac))

    (first0, step0, length0, frac0), (first1, step1, length1, frac1) = axes
    terms = [(d0, d1, w0 * w1)
             for d0, w0 in ((0, 1 - frac0), (1, frac0)) if w0 != 0
             for d1, w1 in ((0, 1 - frac1), (1, frac1)) if w1 != 0]
    for i, (d0, d1, weight) in enumerate(terms):
        view = data[_axis_slice(first0 + d0, step0, length0),
                    _axis_slice(first1 + d1, step1, length1)]
        if swap:
            view = view.T
        if i > 0:
            output += weight * view
        elif weight == 1:
            output[...] = view
        else:
            np.multiply(view, weight, out=output)
    return True


def _bilinear(data, linear, offset, output):
    """Bilinear interpolation, zero outside [0, n-1] as ndimage 'constant' mode"""
    ny, nx = data.shape
    linear = linear.astype(np.float64)
    offset = offset.astype(np.float64)
    flat = np.ascontiguousarray(data).ravel()
    i1 = np.arange(output.shape[1], dtype=np.float64)[None, :]
    rows = max(1, _CHUNK_SIZE // output.shape[1])
    for start in range(0, output.shape[0], rows):
        i0 = np.arange(start, min(start + rows, output.shape[0]), dtype=np.float64)[:, None]
        y = (linear[0, 0] * i0 + offset[0]) + linear[0, 1] * i1
        x = (linear[1, 0] * i0 + offset[1]) + linear[1, 1] * i1
        valid = (y >= 0) & (y <= ny - 1) & (x >= 0) & (x <= nx - 1)
        y0 = np.floor(y)
        x0 = np.floor(x)
        wy = (y - y0).astype(output.dtype)
        wx = (x - x0).astype(output.dtype)
        # Flat index of the (y0, x0) corner. Neighbours out of data only happen
        # with zero weight or outside the valid region, so clipping is enough.
        i00 = (y0 * nx + x0).astype(np.intp)
        v00 = np.take(flat, i00, mode='clip')
        v01 = np.take(flat, i00 + 1, mode='clip')
        i00 += nx
        v10 = np.take(flat, i00, mode='clip')
        v11 = np.take(flat, i00 + 1, mode='clip')
        # v00 + wx * (v01 - v00) for both rows, then between rows
        v01 -= v00
        v01 *= wx
        v00 += v01
        v11 -= v10
        v11 *= wx
        v10 += v11
        v10 -= v00
        v10 *= wy
        v00 += v10
        np.multiply(v00, valid, out=output[start:start + rows])
