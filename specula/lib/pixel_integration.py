"""
Exact integration of a band-limited periodic intensity over detector pixels.

The intensity |DTFT(E)|^2 of a field E with n samples is band limited: it has
frequencies up to n - 1 per period T (T = lambda / pixel pitch of the field).
Its samples on an FFT grid with N >= 2n - 1 points per period therefore give
its Fourier coefficients exactly, and its integral over any interval is a
weighted sum of the samples. This replaces cutting and rebinning a finer FFT
when a detector has pixels that are not a multiple of the FFT sampling.
"""
import numpy as np


def pixel_integration_matrix(n_fft, period, pxscale, npx, stop=None):
    """
    Real (npx, n_fft) matrix G that integrates a band-limited periodic
    intensity over the detector pixels, from its samples on the FFT grid.

    I(theta) has period `period` [arcsec] and frequencies up to
    n_fft / 2 - 1 per period (true for |FFT|^2 of a field with
    n_fft / 2 pixels zero-padded to n_fft). Its samples I_j at
    theta_j = j * period / n_fft (FFT order, axis at j = 0) give its
    Fourier coefficients exactly, c = ifft(I), with
    I(theta) = sum_k c_k exp(-2 pi i k theta / period). The integral over
    [a, b] is then sum_k c_k B_k, with B_k the integral of the exponential,
    and, along one axis, sum_j G_j I_j with G = ifft(B). In 2D the detector
    image is G @ I @ G.T. Pixel i spans [(i - npx/2), (i + 1 - npx/2)] * pxscale,
    clipped to [-stop/2, stop/2] if a square field stop is given.
    """
    k = np.fft.fftfreq(n_fft) * n_fft
    a = (np.arange(npx) - npx / 2) * pxscale
    b = a + pxscale
    if stop is not None:
        a = np.clip(a, -stop / 2, stop / 2)
        b = np.clip(b, -stop / 2, stop / 2)
    w = -2 * np.pi * k / period
    bm = np.empty((npx, n_fft), dtype=complex)
    nz = k != 0
    bm[:, nz] = (np.exp(1j * np.outer(b, w[nz])) - np.exp(1j * np.outer(a, w[nz]))) / (1j * w[nz])
    bm[:, ~nz] = (b - a)[:, None]
    if n_fft % 2 == 0:
        # Zero by the band limit: dropping it makes G real
        bm[:, n_fft // 2] = 0
    g = np.fft.ifft(bm, axis=1)
    return g.real


_LEFT_MUL_SRC = r'''
// Y[b, r, m] = sum_n A[r, n] X[b, n, m], i.e. Y[b] = A @ X[b] for each b.
// X can be a strided view (sxb, sxn, sxm: strides in elements).
//
// Standard tiled matrix product. A block of TILE x TILE threads computes a
// TILE x TILE tile of Y[b], one element per thread. The sum over n is done in
// chunks of TILE: at each step the block copies a TILE x TILE piece of A and
// one of X[b] to shared memory (one element per thread), then every thread
// adds the products for its element from there. Each element of A and X is
// thus read from global memory once per block instead of once per thread,
// which matters when n is large (e.g. N = 1920 for the MORFEO NGS SH).
#define TILE 16
template<typename T>
__global__ void left_mul(const T* A, const T* X, T* Y, int nb, int nr, int nn, int nm,
                         long long sxb, long long sxn, long long sxm) {
    __shared__ T As[TILE][TILE];
    __shared__ T Xs[TILE][TILE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int b = blockIdx.z;
    int r = blockIdx.y * TILE + ty;     // row of Y (and of A) of this thread
    int m = blockIdx.x * TILE + tx;     // column of Y (and of X) of this thread
    const T* xb = X + (long long)b * sxb;
    T acc = 0;
    for (int n0 = 0; n0 < nn; n0 += TILE) {
        // Load A[r, n0 + tx] and X[b, n0 + ty, m]; zeros outside the matrices
        int na = n0 + tx;
        As[ty][tx] = (r < nr && na < nn) ? A[(long long)r * nn + na] : (T)0;
        int nx = n0 + ty;
        Xs[ty][tx] = (nx < nn && m < nm) ? xb[(long long)nx * sxn + (long long)m * sxm] : (T)0;
        __syncthreads();                // tiles complete before they are used
        for (int k = 0; k < TILE; k++)
            acc += As[ty][k] * Xs[k][tx];
        __syncthreads();                // tiles used before they are overwritten
    }
    if (r < nr && m < nm)
        Y[((long long)b * nr + r) * nm + m] = acc;
}
'''
_left_mul_kernels = {}


def left_mul(a, x, out, xp):
    """
    out[b] = a @ x[b] for each b, with a (r, n), x (nb, n, m) possibly a
    strided view, out (nb, r, m) contiguous. On GPU a raw kernel is used
    instead of cuBLAS, which cannot be captured in a CUDA graph.
    """
    if xp is np:
        np.matmul(a, x, out=out)
        return
    # The kernel reads a and writes out as row-major arrays of out.dtype
    if not (a.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError('left_mul: a and out must be C-contiguous')
    if not a.dtype == x.dtype == out.dtype:
        raise ValueError(f'left_mul: a, x and out must have the same dtype, '
                         f'got {a.dtype}, {x.dtype}, {out.dtype}')
    key = (out.dtype.str, xp.cuda.Device().id)
    if key not in _left_mul_kernels:
        ctype = 'float' if out.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_LEFT_MUL_SRC, name_expressions=[f'left_mul<{ctype}>'])
        _left_mul_kernels[key] = mod.get_function(f'left_mul<{ctype}>')
    item = x.itemsize
    nb, n, m = x.shape
    r = a.shape[0]
    # Grid: tiles of 16 x 16 (TILE in the kernel) along (m, r), one subaperture per z
    _left_mul_kernels[key](((m + 15) // 16, (r + 15) // 16, nb), (16, 16),
                           (a, x, out, np.int32(nb), np.int32(r), np.int32(n), np.int32(m),
                            np.int64(x.strides[0] // item), np.int64(x.strides[1] // item),
                            np.int64(x.strides[2] // item)))


def integrate_pixels(g, intensity, tmp, work, xp):
    """
    Integrate a batch of 2D intensity samples over square detector pixels,
    g @ intensity[b] @ g.T for each b, with g from pixel_integration_matrix().

    Parameters
    ----------
    g : array (npx, N)
        Pixel integration matrix (the same along both axes).
    intensity : array (nb, N, N)
        Intensity samples on the FFT grid (FFT order along both axes).
    tmp : array (nb, npx, N)
        Work buffer, contiguous.
    work : array (nb, npx, npx)
        Work buffer, contiguous.
    xp : module
        numpy or cupy.

    Returns
    -------
    image : array (nb, npx, npx)
        A view on `work` (transposed): copy it to keep it beyond the next call.
    """
    # tmp[b] = g @ intensity[b]; work[b] = g @ tmp[b].T = (tmp[b] @ g.T).T
    left_mul(g, intensity, tmp, xp)
    left_mul(g, tmp.swapaxes(1, 2), work, xp)
    return work.swapaxes(1, 2)
