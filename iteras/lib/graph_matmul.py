"""
Matrix products that can be captured in a CUDA graph.

CuPy does not allow cuBLAS calls during stream capture ("calling cuBLAS API
during stream capture is currently unsupported", also through its low-level
cuBLAS binding), so `cp.matmul` cannot be used in a `trigger_code()` that is
captured in a CUDA graph (see iteras.accelerator and build_stream()).

Use `matmul(a, b, xp)`: it is `a @ b` (cuBLAS) outside a capture, and inside one
it uses the raw kernels of this module, chosen by the shapes and memory layout:

* `matvec()`: matrix @ vector(s), one warp per row of the matrix;
* `vecmat()`: a few vectors @ matrix, partial sums over chunks of rows;
* `left_mul()`: matrix @ matrix (or a batch of them), a standard tiled product.

The first two are deterministic and as fast as cuBLAS on the matrices of the
simulations (influence functions, reconstructors), without copying a
transposed matrix. CuPy has its own GEMM kernel, which can be captured, but
only for integer types; compiled for floats it is 2-5x slower than `left_mul()`
on the SH sizes. When CuPy allows cuBLAS during stream capture, `matmul()` can
simply return `a @ b`.
"""
import numpy as np

import iteras

# CuPy 12 requires an explicit C++ standard for template kernels (name_expressions);
# CuPy >= 13 uses this one by default
_CXX_STD = ('-std=c++17',)

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
    _left_mul_gpu(a, x, out, xp)  # pragma: gpu


def _left_mul_gpu(a, x, out, xp):  # pragma: gpu
    # The kernel reads a and writes out as row-major arrays of out.dtype
    if not (a.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError('left_mul: a and out must be C-contiguous')
    if not a.dtype == x.dtype == out.dtype:
        raise ValueError(f'left_mul: a, x and out must have the same dtype, '
                         f'got {a.dtype}, {x.dtype}, {out.dtype}')
    key = (out.dtype.str, xp.cuda.Device().id)
    if key not in _left_mul_kernels:
        ctype = 'float' if out.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_LEFT_MUL_SRC, options=_CXX_STD, name_expressions=[f'left_mul<{ctype}>'])
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
# About 4 waves of resident threads on a large GPU (170 multiprocessors of 1024
# threads): with fewer threads, most of the GPU would be idle. Fixed (not read
# from the device), so that the result does not depend on the GPU model.
_VECMAT_THREADS = 4 * 170 * 1024


def vecmat(v, m, xp):
    """
    v @ m for a vector v (n,) or a few vectors (k, n), and a C-contiguous
    matrix m (n, m), with a raw kernel that can be captured in a CUDA graph.
    As fast as cuBLAS on large matrices (e.g. influence functions), without
    copying the transposed matrix.
    """
    if xp is np:
        return v @ m
    return _vecmat_gpu(v, m, xp)  # pragma: gpu


def _vecmat_gpu(v, m, xp):  # pragma: gpu
    if not m.flags.c_contiguous or v.dtype != m.dtype:
        raise ValueError('vecmat: the matrix must be C-contiguous, with the dtype of the vectors')
    key = (m.dtype.str, xp.cuda.Device().id)
    if key not in _vecmat_kernels:
        ctype = 'float' if m.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_VECMAT_SRC, options=_CXX_STD, name_expressions=[f'vecmat_part<{ctype}>'])
        _vecmat_kernels[key] = mod.get_function(f'vecmat_part<{ctype}>')
    v2 = xp.ascontiguousarray(v if v.ndim == 2 else v[None])
    nk, n = v2.shape
    ncol = m.shape[1]
    # One thread per column and chunk of rows: with few columns, the rows are split
    # in up to 64 chunks, to have about _VECMAT_THREADS threads, enough to fill a
    # large GPU
    nsplit = max(1, min(64, n, _VECMAT_THREADS // max(ncol, 1)))
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
    return _matvec_gpu(a, v, xp)  # pragma: gpu


def _matvec_gpu(a, v, xp):  # pragma: gpu
    if not a.flags.c_contiguous or v.dtype != a.dtype:
        raise ValueError('matvec: the matrix must be C-contiguous, with the dtype of the vectors')
    key = (a.dtype.str, xp.cuda.Device().id)
    if key not in _matvec_kernels:
        ctype = 'float' if a.dtype == np.float32 else 'double'
        mod = xp.RawModule(code=_MATVEC_SRC, options=_CXX_STD, name_expressions=[f'matvec_rows<{ctype}>'])
        _matvec_kernels[key] = mod.get_function(f'matvec_rows<{ctype}>')
    v2 = xp.ascontiguousarray(v if v.ndim == 2 else v[None])
    nk, n = v2.shape
    r = a.shape[0]
    y = xp.empty((nk, r), dtype=a.dtype)
    _matvec_kernels[key](((r * 32 + 255) // 256,), (256,),
                         (a, v2, y, np.int32(nk), np.int32(r), np.int32(n)))
    return y if v.ndim == 2 else y[0]


# Up to this number of rows, a @ b with a 2-d "a" is computed as vectors @ matrix
_FEW_VECTORS = 8


def matmul(a, b, xp):
    """
    a @ b, also inside a CUDA graph capture (see iteras.accelerator), where cuBLAS
    cannot be used: then matrix @ vector and vector(s) @ matrix use matvec() or
    vecmat(), depending on the memory layout of the matrix, and matrix @ matrix
    uses left_mul(). Outside a capture, or for other shapes, it is a @ b (cuBLAS).
    """
    from iteras import capturing
    if xp is np or not capturing():
        return a @ b
    return _matmul_captured(a, b, xp)  # pragma: gpu


def _matmul_captured(a, b, xp):  # pragma: gpu
    '''a @ b inside a CUDA graph capture (see matmul())'''
    # With a transposed (Fortran-ordered) matrix, the other kernel is used on its
    # C-contiguous transpose, since a @ b = (b.T @ a.T).T:
    # A @ v = v @ A.T, and v @ B = B.T @ v (for vectors, .T does nothing)
    if a.ndim == 2 and b.ndim == 1:
        if a.flags.c_contiguous:
            return matvec(a, b, xp)
        if a.T.flags.c_contiguous:
            return vecmat(b, a.T, xp)
    few = a.ndim == 1 or (a.ndim == 2 and a.shape[0] <= _FEW_VECTORS)
    if few and b.ndim == 2:
        if b.flags.c_contiguous:
            return vecmat(a, b, xp)
        if b.T.flags.c_contiguous:
            return matvec(b.T, a, xp)
    if a.ndim == 2 and b.ndim == 2 and a.dtype == b.dtype:
        out = xp.empty((1, a.shape[0], b.shape[1]), dtype=a.dtype)
        left_mul(xp.ascontiguousarray(a), b[None], out, xp)
        return out[0]
    return a @ b


_SPARSE_COLUMNS_SRC = '''
T acc = 0;
for (int k = 0; k < kmax; k++) {
    long long j = (long long)k * n + i;
    acc += w[j] * v[idx[j]];
}
y = acc;
'''
_sparse_columns_kernels = {}


class SparseColumns:
    """
    A matrix M (r, n) with few non-zero elements in each column, for v @ M.

    For each column j, the k-th non-zero element (in row order) is
    w[k, j] = M[idx[k, j], j], with zero padding up to kmax elements
    (the ELLPACK format, column-major so that the GPU reads are coalesced).
    v @ M is then y[j] = sum_k w[k, j] v[idx[k, j]]: on GPU, one thread per
    column, adding in a fixed order (deterministic), in a kernel that can be
    captured in a CUDA graph. Only the non-zero elements are read, which
    matters for zonal or segmented influence functions: e.g. the M4 ones
    have about 2% of non-zeros, and v @ M is limited by the memory bandwidth.
    """

    def __init__(self, idx, w, xp):
        self.idx = idx
        self.w = w
        self.kmax, self.n = w.shape
        self.xp = xp

    @classmethod
    def from_dense(cls, m, xp, max_fraction=0.125, chunk_rows=None):
        """
        The SparseColumns form of the dense matrix m (r, n), or None if a
        column has more than max_fraction * r non-zero elements: the dense
        product is then faster. The rows are scanned in chunks of up to 4M
        elements by default, so that the temporaries stay small (they would
        otherwise remain in the memory pool, also when a dense matrix is
        rejected).
        """
        r, n = m.shape
        limit = max_fraction * r
        if chunk_rows is None:
            chunk_rows = max(1, (1 << 22) // max(n, 1))
        # First pass: count, so that a dense matrix is rejected before extracting its elements
        counts = np.zeros(n, dtype=np.int64)
        for r0 in range(0, r, chunk_rows):
            counts += iteras.cpuArray((m[r0:r0 + chunk_rows] != 0).sum(axis=0, dtype=np.int32))
            if counts.max() > limit:
                return None
        # Second pass: extract the non-zero elements
        rows, cols, vals = [], [], []
        for r0 in range(0, r, chunk_rows):
            chunk = m[r0:r0 + chunk_rows]
            nz_r, nz_c = xp.nonzero(chunk)
            vals.append(iteras.cpuArray(chunk[nz_r, nz_c]))
            nz_r, nz_c = iteras.cpuArray(nz_r), iteras.cpuArray(nz_c)
            rows.append(nz_r + r0)
            cols.append(nz_c)
        rows = np.concatenate(rows).astype(np.int32)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        # Sort by column, then by row: position k of each element in its column
        order = np.lexsort((rows, cols))
        rows, cols, vals = rows[order], cols[order], vals[order]
        start = np.concatenate(([0], np.cumsum(counts)[:-1]))
        k = np.arange(len(cols)) - start[cols]
        kmax = max(int(counts.max()), 1)
        idx = np.zeros((kmax, n), dtype=np.int32)
        w = np.zeros((kmax, n), dtype=vals.dtype)
        idx[k, cols] = rows
        w[k, cols] = vals
        return cls(xp.asarray(idx), xp.asarray(w), xp)

    def vecmat(self, v):
        """v @ M for a vector v (r,)"""
        xp = self.xp
        if xp is np:
            # One row of w at a time, in the order of the GPU kernel:
            # the temporaries have n elements instead of kmax * n
            y = self.w[0] * v[self.idx[0]]
            for k in range(1, self.kmax):
                y += self.w[k] * v[self.idx[k]]
            return y
        return self._vecmat_gpu(v)  # pragma: gpu

    def _vecmat_gpu(self, v):  # pragma: gpu
        xp = self.xp
        if v.dtype != self.w.dtype:
            raise ValueError(f'SparseColumns.vecmat: v must have dtype {self.w.dtype}, got {v.dtype}')
        key = (self.w.dtype.str, xp.cuda.Device().id)
        if key not in _sparse_columns_kernels:
            _sparse_columns_kernels[key] = xp.ElementwiseKernel(
                'raw T v, raw int32 idx, raw T w, int32 kmax, int32 n', 'T y',
                _SPARSE_COLUMNS_SRC, 'sparse_columns_vecmat')
        y = xp.empty(self.n, dtype=v.dtype)
        _sparse_columns_kernels[key](xp.ascontiguousarray(v), self.idx, self.w,
                          np.int32(self.kmax), np.int32(self.n), y)
        return y
