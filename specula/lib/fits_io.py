import functools

import numpy as np
from astropy.io import fits

import specula

_BITPIX2DTYPE = {8: np.uint8, 16: np.int16, 32: np.int32, 64: np.int64,
                 -32: np.float32, -64: np.float64}

_CHUNK_BYTES = 64 << 20


@functools.cache
def _swap_kernel():
    '''Big-endian values of type S (read as raw bytes) to native values of type O'''
    return specula.cp.ElementwiseKernel('S x', 'O y', '''
        S v;
        const unsigned char* s = reinterpret_cast<const unsigned char*>(&x);
        unsigned char* d = reinterpret_cast<unsigned char*>(&v);
        for (int i = 0; i < sizeof(S); i++) d[i] = s[sizeof(S) - 1 - i];
        y = v;
        ''', 'specula_fits_swap')


def load_fits_array(filename, exten=1, target_device_idx=None, precision=None):
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

    On GPU, the file is read in chunks into two pinned host buffers,
    which are uploaded asynchronously while the next chunk is read,
    and are byteswapped and converted on the GPU. This avoids
    full-size temporary copies in host memory.
    '''
    if target_device_idx is None:
        target_device_idx = specula.default_target_device_idx

    float_dtype = np.dtype(specula.cpu_float_dtype_list[
        specula.global_precision if precision is None else precision])

    def target_dtype(dtype):
        if dtype.kind == 'f':
            return float_dtype
        if precision is not None:
            raise ValueError(f'{filename} extension {exten} contains integer data ({dtype}), '
                             f'but precision={precision} was requested')
        return dtype.newbyteorder('=')

    with fits.open(filename) as hdul:
        hdu = hdul[exten]
        hdr = hdu.header
        if hdr.get('NAXIS', 0) == 0:
            return None     # No data, like astropy
        info = hdu.fileinfo()
        fast = (target_device_idx >= 0
                and type(hdu) in (fits.PrimaryHDU, fits.ImageHDU)
                and info['file'].compression is None
                and hdr.get('BSCALE', 1) == 1 and hdr.get('BZERO', 0) == 0
                and 'BLANK' not in hdr
                and hdr.get('BITPIX') in _BITPIX2DTYPE)
        if not fast:
            data = hdu.data
            if target_device_idx < 0:
                return np.array(data, dtype=target_dtype(data.dtype))
            with specula.cp.cuda.Device(target_device_idx):
                return specula.cp.asarray(np.asarray(data, dtype=target_dtype(data.dtype)))
        src_dtype = np.dtype(_BITPIX2DTYPE[hdr['BITPIX']])
        shape = hdu.shape
        offset = info['datLoc']

    cp = specula.cp
    size = int(np.prod(shape))
    chunk = max(1, min(_CHUNK_BYTES // src_dtype.itemsize, size))
    with cp.cuda.Device(target_device_idx), cp.cuda.Stream(non_blocking=True) as stream:
        out = cp.empty(shape, dtype=target_dtype(src_dtype))
        out_flat = out.reshape(-1)
        # Double buffering: a chunk is read while the previous one is uploaded and converted
        host = [np.frombuffer(cp.cuda.alloc_pinned_memory(chunk * src_dtype.itemsize),
                              dtype=src_dtype, count=chunk) for _ in range(2)]
        dev = [cp.empty(chunk, dtype=src_dtype) for _ in range(2)]
        done = [cp.cuda.Event() for _ in range(2)]
        with open(filename, 'rb') as f:
            f.seek(offset)
            for i, start in enumerate(range(0, size, chunk)):
                k = i % 2
                n = min(chunk, size - start)
                done[k].synchronize()
                if f.readinto(host[k][:n]) != n * src_dtype.itemsize:
                    raise EOFError(f'Unexpected end of file reading {filename}')
                dev[k][:n].set(host[k][:n], stream=stream)
                _swap_kernel()(dev[k][:n], out_flat[start:start + n])
                done[k].record(stream)
        stream.synchronize()
    return out
