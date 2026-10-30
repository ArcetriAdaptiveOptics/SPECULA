"""
Batched matrix product that can be captured in a CUDA graph.

CuPy does not allow cuBLAS calls during stream capture ("calling cuBLAS API
during stream capture is currently unsupported", also through its low-level
cuBLAS binding), so `cp.matmul` cannot be used in a `trigger_code()` that is
captured in a CUDA graph. For matrix-vector products, use `sum_product()` in
`specula.processing_objects.slopec`, which is close to optimal there. For
matrix-matrix products it is several times slower, because it does not reuse
the data: use `left_mul()` below, a standard tiled matrix product.

CuPy has its own GEMM kernel, which can be captured, but only for integer
types; compiled for floats it is 2-5x slower than this one on the SH sizes.
When CuPy allows cuBLAS during stream capture, `left_mul()` can be replaced
by `cp.matmul`.
"""
import numpy as np

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
    out[b] = a @ x[b] for each b. On GPU a raw kernel is used instead of
    cuBLAS, so that the product can be captured in a CUDA graph.

    Parameters
    ----------
    a : array (r, n)
        Left matrix, C-contiguous.
    x : array (nb, n, m)
        Batch of right matrices; can be a strided view (e.g. a transposed one).
    out : array (nb, r, m)
        Output, C-contiguous, same dtype as `a` and `x` (float32 or float64).
    xp : module
        numpy or cupy.
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
    # Grid: tiles of 16 x 16 (TILE in the kernel) along (m, r), one matrix of the batch per z
    _left_mul_kernels[key](((m + 15) // 16, (r + 15) // 16, nb), (16, 16),
                           (a, x, out, np.int32(nb), np.int32(r), np.int32(n), np.int32(m),
                            np.int64(x.strides[0] // item), np.int64(x.strides[1] // item),
                            np.int64(x.strides[2] // item)))


_VECMAT_SRC = r'''
// Partial sums of Y[k, j] = sum_i V[k, i] M[i, j] over the chunk s of i:
// part[s, k, j]. The threads of a block read consecutive j, so the rows of M
// are read with coalesced accesses, each element of M once. The partial sums
// are then added in a fixed order (deterministic, unlike atomic additions).
template<typename T>
__global__ void vecmat_part(const T* V, const T* M, T* part, int nk, int n, int m, int chunk) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int s = blockIdx.y;
    if (j >= m) return;
    int i0 = s * chunk;
    int i1 = min(n, i0 + chunk);
    for (int k = 0; k < nk; k++) {
        const T* v = V + (long long)k * n;
        T acc = 0;
        for (int i = i0; i < i1; i++)
            acc += v[i] * M[(long long)i * m + j];
        part[((long long)s * nk + k) * m + j] = acc;
    }
}
'''
_vecmat_kernels = {}


def vecmat(v, m, xp):
    """
    v @ m for a vector v (n,) or a few vectors (k, n), and a C-contiguous
    matrix m (n, m), with a raw kernel that can be captured in a CUDA graph.
    As fast as cuBLAS on large matrices (e.g. influence functions), without
    copying the transposed matrix.
    """
    if xp is np:
        return v @ m
    if not m.flags.c_contiguous or v.dtype != m.dtype:
        raise ValueError('vecmat: the matrix must be C-contiguous, with the dtype of the vectors')
    key = (m.dtype.str, xp.cuda.Device().id)
    if key not in _vecmat_kernels:
        ctype = 'float' if m.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_VECMAT_SRC, name_expressions=[f'vecmat_part<{ctype}>'])
        _vecmat_kernels[key] = mod.get_function(f'vecmat_part<{ctype}>')
    v2 = xp.ascontiguousarray(v if v.ndim == 2 else v[None])
    nk, n = v2.shape
    ncol = m.shape[1]
    # Chunks of rows: enough threads to fill the GPU when there are few columns
    nsplit = max(1, min(64, n, (4 * 170 * 1024) // max(ncol, 1)))
    chunk = -(-n // nsplit)
    nsplit = -(-n // chunk)
    part = xp.empty((nsplit, nk, ncol), dtype=m.dtype)
    _vecmat_kernels[key](((ncol + 255) // 256, nsplit), (256,),
                         (v2, m, part, np.int32(nk), np.int32(n), np.int32(ncol), np.int32(chunk)))
    y = part.sum(axis=0)
    return y if v.ndim == 2 else y[0]


_MATVEC_SRC = r'''
// Y[k, r] = sum_c A[r, c] V[k, c], A C-contiguous (R x C): one warp per row r.
// The threads of the warp read consecutive elements of the row, then add their
// sums with warp shuffles, in a fixed order (deterministic).
template<typename T>
__global__ void matvec_rows(const T* A, const T* V, T* Y, int nk, int R, int C) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (row >= R) return;
    const T* a = A + (long long)row * C;
    for (int k = 0; k < nk; k++) {
        const T* v = V + (long long)k * C;
        T acc = 0;
        for (int c = lane; c < C; c += 32)
            acc += a[c] * v[c];
        for (int off = 16; off > 0; off /= 2)
            acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0)
            Y[(long long)k * R + row] = acc;
    }
}
'''
_matvec_kernels = {}


def matvec(a, v, xp):
    """
    a @ v for a C-contiguous matrix a (r, n) and a vector v (n,), or a few
    vectors v (k, n) giving (k, r), with a raw kernel that can be captured in a
    CUDA graph. As fast as cuBLAS, also for many short rows (e.g. a transposed
    influence function).
    """
    if xp is np:
        return a @ v if v.ndim == 1 else v @ a.T
    if not a.flags.c_contiguous or v.dtype != a.dtype:
        raise ValueError('matvec: the matrix must be C-contiguous, with the dtype of the vectors')
    key = (a.dtype.str, xp.cuda.Device().id)
    if key not in _matvec_kernels:
        ctype = 'float' if a.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_MATVEC_SRC, name_expressions=[f'matvec_rows<{ctype}>'])
        _matvec_kernels[key] = mod.get_function(f'matvec_rows<{ctype}>')
    v2 = xp.ascontiguousarray(v if v.ndim == 2 else v[None])
    nk, n = v2.shape
    r = a.shape[0]
    y = xp.empty((nk, r), dtype=a.dtype)
    _matvec_kernels[key](((r * 32 + 255) // 256,), (256,),
                         (a, v2, y, np.int32(nk), np.int32(r), np.int32(n)))
    return y if v.ndim == 2 else y[0]


def matmul(a, b, xp):
    """
    a @ b, also inside a CUDA graph capture (see iteras.accelerator), where cuBLAS
    cannot be used: then matrix @ vector and vector(s) @ matrix use matvec() or
    vecmat(), depending on the memory layout of the matrix. Outside a capture,
    or for other shapes, it is a @ b (cuBLAS).
    """
    from specula import capturing
    if xp is np or not capturing():
        return a @ b
    if a.ndim == 2 and b.ndim == 1:
        if a.flags.c_contiguous:
            return matvec(a, b, xp)
        if a.T.flags.c_contiguous:
            return vecmat(b, a.T, xp)
    if a.ndim in (1, 2) and b.ndim == 2:
        if b.flags.c_contiguous:
            return vecmat(a, b, xp)
        if b.T.flags.c_contiguous:
            return matvec(b.T, a, xp)
    return a @ b
