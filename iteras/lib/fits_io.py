import functools

import numpy as np
from astropy.io import fits

import iteras

_BITPIX2DTYPE = {8: np.uint8, 16: np.int16, 32: np.int32, 64: np.int64,
                 -32: np.float32, -64: np.float64}

_CHUNK_BYTES = 64 << 20


@functools.cache
def _swap_kernel():  # pragma: no cover (GPU only, not run in CI)
    '''Big-endian values of type S (read as raw bytes) to native values of type O'''
    return iteras.cp.ElementwiseKernel('S x', 'O y', '''
        S v;
        const unsigned char* s = reinterpret_cast<const unsigned char*>(&x);
        unsigned char* d = reinterpret_cast<unsigned char*>(&v);
        for (int i = 0; i < sizeof(S); i++) d[i] = s[sizeof(S) - 1 - i];
        y = v;
        ''', 'iteras_fits_swap')


def load_fits_array(filename, ext=1, target_device_idx=None, precision=None, transpose=False):
    '''
    Read the data of an image extension of a FITS file into a
    numpy or cupy array allocated on *target_device_idx*
    (same convention as BaseTimeObj: None for the default device,
    -1 for the CPU).

    Floating point data is converted to the float dtype of
    *precision* (None for the global precision). Integer data is
    kept as it is if *precision* is None, otherwise a ValueError
    is raised. The result is always in native byte order.
    Returns None if the extension has no data (NAXIS=0).
    With *transpose*, a 2D array is returned transposed, in C order
    (e.g. influence functions, stored as pixels x modes and used
    as modes x pixels), without a full-size temporary copy on GPU.

    On GPU, the file is read in chunks into two pinned host buffers,
    which are uploaded asynchronously while the next chunk is read,
    and are byteswapped and converted on the GPU. This avoids
    full-size temporary copies in host memory. On the CPU, the file
    is also read in chunks, converted into the result.
    '''
    if target_device_idx is None:
        target_device_idx = iteras.default_target_device_idx

    float_dtype = np.dtype(iteras.cpu_float_dtype_list[
        iteras.global_precision if precision is None else precision])

    def target_dtype(dtype):
        if dtype.kind == 'f':
            return float_dtype
        if precision is not None:
            raise ValueError(f'{filename} extension {ext} contains integer data ({dtype}), '
                             f'but precision={precision} was requested')
        return dtype.newbyteorder('=')

    with fits.open(filename) as hdul:
        hdu = hdul[ext]
        hdr = hdu.header
        if hdr.get('NAXIS', 0) == 0:
            return None     # No data, like astropy
        info = hdu.fileinfo()
        fast = (type(hdu) in (fits.PrimaryHDU, fits.ImageHDU)
                and info['file'].compression is None
                and hdr.get('BSCALE', 1) == 1 and hdr.get('BZERO', 0) == 0
                and 'BLANK' not in hdr
                and hdr.get('BITPIX') in _BITPIX2DTYPE)
        if not fast:
            data = hdu.data
            dtype = target_dtype(data.dtype)
            if transpose:
                if data.ndim != 2:
                    raise ValueError(f'transpose needs a 2D array, {filename} extension {ext} '
                                     f'has shape {data.shape}')
                data = data.T
            if target_device_idx < 0:
                return np.array(data, dtype=dtype, order='C')
            with iteras.cp.cuda.Device(target_device_idx):  # pragma: no cover
                return iteras.cp.asarray(np.array(data, dtype=dtype, order='C'))
        src_dtype = np.dtype(_BITPIX2DTYPE[hdr['BITPIX']])
        shape = hdu.shape
        offset = info['datLoc']

    if target_device_idx < 0:
        return read_from_file(filename, offset, shape, src_dtype, target_dtype(src_dtype), byteswap=True,
                              transpose=transpose)
    return upload_from_file(filename, offset, shape, src_dtype, target_dtype(src_dtype),  # pragma: no cover
                            target_device_idx, byteswap=True, transpose=transpose)


def read_from_file(filename, offset, shape, src_dtype, out_dtype, byteswap, transpose=False):
    '''
    Read an array stored contiguously (C order) at *offset* in a file, with values
    of type *src_dtype* (big-endian if *byteswap* is True, native otherwise), into
    a new numpy array of type *out_dtype*, in chunks: the only full-size array is
    the result (reading the whole file and then converting it would need two).
    With *transpose*, a 2D array is returned transposed, in C order: each chunk of
    rows is written into the columns of the result, as in upload_from_file().
    '''
    file_dtype = np.dtype(src_dtype).newbyteorder('>' if byteswap else '=')
    size = int(np.prod(shape))
    if transpose and len(shape) != 2:
        raise ValueError(f'transpose needs a 2D array, {filename} has shape {shape}')
    out = np.empty(tuple(shape[::-1]) if transpose else shape, dtype=out_dtype)
    if size == 0:
        return out
    # With transpose, chunks of whole rows (at least one)
    row = shape[1] if transpose else 1
    target = out.T if transpose else out.reshape(-1)
    buf = np.empty(max(1, min(_CHUNK_BYTES // (file_dtype.itemsize * row), size // row)) * row,
                   dtype=file_dtype)
    with open(filename, 'rb') as f:
        f.seek(offset)
        for start in range(0, size, len(buf)):
            n = min(len(buf), size - start)
            if f.readinto(buf[:n]) != n * file_dtype.itemsize:
                raise EOFError(f'Unexpected end of file reading {filename}')
            if transpose:
                target[start // row:(start + n) // row] = buf[:n].reshape(-1, row)
            else:
                target[start:start + n] = buf[:n]
    return out


def upload_from_file(filename, offset, shape, src_dtype, out_dtype, target_device_idx,
                     byteswap, transpose=False):  # pragma: no cover (GPU only, not run in CI)
    '''
    Read an array stored contiguously (C order) at *offset* in a file, with values
    of type *src_dtype* (big-endian if *byteswap* is True, native otherwise), into
    a new cupy array of type *out_dtype* on GPU *target_device_idx*.
    With *transpose*, a 2D array is returned transposed, in C order: each chunk of
    rows is written into the columns of the result, without a temporary copy.

    The file is read in chunks into two pinned host buffers, which are uploaded
    asynchronously while the next chunk is read, and are converted (and byteswapped)
    on the GPU. This avoids full-size temporary copies in host memory.
    '''
    cp = iteras.cp
    src_dtype = np.dtype(src_dtype)
    if byteswap:
        src_dtype = src_dtype.newbyteorder('=')
    size = int(np.prod(shape))
    if transpose and len(shape) != 2:
        raise ValueError(f'transpose needs a 2D array, {filename} has shape {shape}')
    out_shape = tuple(shape[::-1]) if transpose else shape
    if size == 0:
        with cp.cuda.Device(target_device_idx):
            return cp.empty(out_shape, dtype=out_dtype)
    # With transpose, chunks of whole rows (at least one)
    row = shape[1] if transpose else 1
    chunk = max(1, min(_CHUNK_BYTES // (src_dtype.itemsize * row), size // row)) * row
    # The device chunks come from a pool used only for this upload, emptied at the end.
    # In the default pool their blocks would stay cached: results smaller than a chunk
    # are carved out of them, so that the next upload needs new chunks (1 GB of cached
    # blocks after the setup of MORFEO). A deleted pool does not free its memory.
    staging = cp.cuda.MemoryPool()
    with cp.cuda.Device(target_device_idx), cp.cuda.Stream(non_blocking=True) as stream:
        out = cp.empty(out_shape, dtype=out_dtype)
        if transpose:
            def target(start, n):
                return out.T[start // row:(start + n) // row]
        else:
            out_flat = out.reshape(-1)

            def target(start, n):
                return out_flat[start:start + n]
        # Double buffering: a chunk is read while the previous one is uploaded and converted
        host = [np.frombuffer(cp.cuda.alloc_pinned_memory(chunk * src_dtype.itemsize),
                              dtype=src_dtype, count=chunk) for _ in range(2)]
        with cp.cuda.using_allocator(staging.malloc):
            dev = [cp.empty(chunk, dtype=src_dtype) for _ in range(2)]
        done = [cp.cuda.Event() for _ in range(2)]
        try:
            with open(filename, 'rb') as f:
                f.seek(offset)
                for i, start in enumerate(range(0, size, chunk)):
                    k = i % 2
                    n = min(chunk, size - start)
                    done[k].synchronize()
                    if f.readinto(host[k][:n]) != n * src_dtype.itemsize:
                        raise EOFError(f'Unexpected end of file reading {filename}')
                    dev[k][:n].set(host[k][:n], stream=stream)
                    dst = target(start, n)
                    src = dev[k][:n].reshape(dst.shape)
                    if byteswap:
                        _swap_kernel()(src, dst)
                    else:
                        dst[...] = src
                    done[k].record(stream)
        finally:
            stream.synchronize()
            del dev
            staging.free_all_blocks()
    return out
