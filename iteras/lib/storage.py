'''
File formats of the data saved and restored by iteras and its plugins.

Data objects (save() / restore(), calibration files) and the time series
saved by DataStore are stored as a header (keywords, values and comments)
and a set of named arrays, through write_object() and read_object().
Supported formats, chosen by the extension of the file name:

- ``fits`` (``.fits``): FITS file. The arrays are stored as image HDUs, as
  described by the *layout* argument of write_object(), so that each class
  can keep the layout of its existing files;
- ``npz`` (``.npz``): numpy archive with one entry per array, and the header
  as a JSON string in the ``__header__`` entry (read without pickle);
- ``npy`` (``.npy``): a directory containing one ``<array>.npy`` file per array
  and the header in ``header.json``. Large arrays can be read without
  copies (and streamed to the GPU in chunks, like FITS files);
- ``hdf5`` (``.h5``): HDF5 file with one dataset per array, the header in the
  ``__header__`` attribute, and the header keywords also as plain attributes.

The header of the npz, npy and HDF5 files is read as a Header, a dictionary
without the limits of FITS headers (keywords longer than 8 characters, text
that is not ASCII).

The time series of DataStore can also be saved in the ``pickle`` format
(``.pickle``), for compatibility.

The default format is set by the ITERAS_DATA_FORMAT environment variable
(or one of its aliases, see iteras.registry), and is ``fits`` if it is not set.
'''
import json
import os
import shutil

import numpy as np

import iteras
from iteras import registry

# Formats of the DataStore time series
FORMATS = ('fits', 'pickle', 'npz', 'npy', 'hdf5')

# Formats of the data objects (pickle is only for DataStore)
OBJECT_FORMATS = ('fits', 'npz', 'npy', 'hdf5')

EXTENSIONS = {'fits': '.fits', 'pickle': '.pickle', 'npz': '.npz', 'npy': '.npy', 'hdf5': '.h5'}

_OBJECT_EXTENSIONS = {'.fits': 'fits', '.fit': 'fits', '.fts': 'fits', '.npz': 'npz',
                      '.npy': 'npy', '.h5': 'hdf5', '.hdf5': 'hdf5'}

# FITS layouts of write_object()
LAYOUTS = ('extensions', 'primary', 'placeholder')

# Name of the header file in the directory of the npy format
NPY_HEADER = 'header.json'

# FITS structure keywords, recomputed by astropy from the data
_STRUCTURE_KEYWORDS = {'SIMPLE', 'BITPIX', 'EXTEND', 'XTENSION', 'PCOUNT', 'GCOUNT',
                       'BZERO', 'BSCALE', 'EXTNAME'}


# ---------------------------------------------------------------------------
# Formats and file names
# ---------------------------------------------------------------------------

def check_format(data_format, source='data_format'):
    '''Raise ValueError if *data_format* is not supported'''
    if data_format not in FORMATS:
        raise ValueError(f'Unsupported {source} {data_format!r}: must be one of {", ".join(FORMATS)}')


def default_format():
    '''Format used when none is given: ITERAS_DATA_FORMAT, or "fits" if it is not set'''
    data_format = registry.getenv('ITERAS_DATA_FORMAT', 'fits')
    check_format(data_format, source='ITERAS_DATA_FORMAT environment variable value')
    return data_format


def default_object_format():
    '''
    Format of the data object files written without an explicit extension: the
    default format (see default_format()), or 'fits' if that is not available
    for data objects (pickle).
    '''
    data_format = default_format()
    return data_format if data_format in OBJECT_FORMATS else 'fits'


def filename(directory, name, data_format):
    '''Path of the file of the DataStore input *name* in *directory*'''
    return os.path.join(directory, name + EXTENSIONS[data_format])


def find_format(directory, name):
    '''
    Format of the file of the DataStore input *name* in *directory*, looking
    for the extension of each format, or None if no file is found
    '''
    for data_format in FORMATS:
        if os.path.exists(filename(directory, name, data_format)):
            return data_format
    return None


def object_format(path):
    '''Format of a data object file, from its extension, or None if the extension is not known'''
    return _OBJECT_EXTENSIONS.get(os.path.splitext(os.fspath(path))[1].lower())


def object_filename(path):
    '''*path*, with the extension of the default object format if it has no known extension'''
    path = os.fspath(path)
    if object_format(path) is not None:
        return path
    return path + EXTENSIONS[default_object_format()]


def find_object_file(path):
    '''
    Existing file of a data object, given its path without extension: the
    file with the extension of the default object format, or of the other
    formats, or None if none exists.
    '''
    path = os.fspath(path)
    first = default_object_format()
    for data_format in [first] + [f for f in OBJECT_FORMATS if f != first]:
        candidate = path + EXTENSIONS[data_format]
        if os.path.exists(candidate):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def _plain(value):
    '''Header value as str, int, float, bool or None'''
    if isinstance(value, np.generic):
        value = value.item()
    if value is not None and not isinstance(value, (str, int, float, bool)):
        value = str(value)
    return value


class Header(dict):
    '''
    Header read from an npz, npy or HDF5 file: a dictionary of keywords and
    values, without the limits of FITS headers, used like a FITS header by
    the restore() methods. As in a FITS header, the comments of the keywords
    are in *comments*, and the COMMENT and HISTORY entries are lists of strings.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comments = {}


def header_cards(header):
    '''
    Header entries as a JSON-serializable list of [keyword, value, comment],
    from a FITS header, a Header or a dictionary. FITS structure keywords are left out.
    '''
    if header is None:
        return []
    if hasattr(header, 'cards'):
        cards = [(card.keyword, card.value, card.comment) for card in header.cards]
    elif isinstance(header, Header):
        cards = []
        for key, value in header.items():
            if key in ('COMMENT', 'HISTORY'):
                cards += [(key, text, '') for text in value]
            else:
                cards.append((key, value, header.comments.get(key, '')))
    else:
        cards = [(key, value, '') for key, value in header.items()]
    return [[key, _plain(value), comment] for key, value, comment in cards
            if key not in _STRUCTURE_KEYWORDS and not key.startswith('NAXIS')]


def header_from_cards(cards):
    '''Header built from the list of header_cards()'''
    header = Header()
    for key, value, comment in cards:
        if key in ('COMMENT', 'HISTORY'):
            header.setdefault(key, []).append(value)
        else:
            header[key] = value
            if comment:
                header.comments[key] = comment
    return header


def header_to_dict(header):
    '''
    Header entries as a dictionary of plain values (str, int, float, bool),
    from a FITS header or a dictionary. Comments and history are left out.
    '''
    return {key: (str(value) if value is None else value)
            for key, value, _ in header_cards(header)
            if key not in ('COMMENT', 'HISTORY', '')}


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_object(path, header, arrays, layout='extensions', overwrite=True):
    '''
    Write a header and named arrays to a file, in the format given by the
    extension of *path* (FITS if the extension is not known).

    Parameters
    ----------
    path : str or path-like
        File name. For the npy format, the name of a directory.
    header : astropy.io.fits.Header or dict
        Header entries.
    arrays : dict
        Arrays (numpy or cupy) by name, in order. None values are left out.
    layout : str
        How the arrays are stored in a FITS file:

        - ``'extensions'``: header in the primary HDU, without data, and each
          array in an image extension named after it (EXTNAME);
        - ``'primary'``: first array in the primary HDU, with the header, and
          the other arrays in named extensions. The primary HDU has no name:
          when reading a FITS file, its array is found by its index (0);
        - ``'placeholder'``: primary HDU with the header and a 2-element array
          of zeros, and the arrays in named extensions (the layout of old
          files written with fits.writeto() followed by fits.append()).

        The other formats store the header and the named arrays in all cases.
    overwrite : bool
        Replace an existing file. If False and the file exists, OSError is raised.
    '''
    if layout not in LAYOUTS:
        raise ValueError(f'Unknown layout {layout!r}: must be one of {", ".join(LAYOUTS)}')
    path = os.fspath(path)
    data_format = object_format(path) or 'fits'
    if os.path.exists(path) and not overwrite:
        raise OSError(f'File {path} already exists. If you mean to replace it '
                      'then use the argument "overwrite=True".')
    arrays = {name: np.asarray(iteras.cpuArray(array)) for name, array in arrays.items()
              if array is not None}
    cards = header_cards(header)

    if data_format == 'fits':
        _write_fits(path, header, arrays, layout)
    elif data_format == 'npz':
        with open(path, 'wb') as f:
            np.savez(f, __header__=np.array(json.dumps(cards)), **arrays)
    elif data_format == 'hdf5':
        import h5py
        with h5py.File(path, 'w') as f:
            f.attrs['__header__'] = json.dumps(cards)
            f.attrs['__arrays__'] = json.dumps(list(arrays))
            for key, value in header_to_dict(header).items():
                f.attrs[key] = value
            for name, array in arrays.items():
                f.create_dataset(name, data=array)
    elif data_format == 'npy':
        if os.path.exists(path):
            if not os.path.exists(os.path.join(path, NPY_HEADER)):
                raise OSError(f'{path} exists and is not a directory written in the npy format')
            shutil.rmtree(path)
        os.makedirs(path)
        for name, array in arrays.items():
            np.save(os.path.join(path, name + '.npy'), array, allow_pickle=False)
        with open(os.path.join(path, NPY_HEADER), 'w') as f:
            json.dump({'header': cards, 'arrays': list(arrays)}, f, indent=1)


def _write_fits(path, header, arrays, layout):
    from astropy.io import fits
    if header is None:
        header = fits.Header()
    elif not isinstance(header, fits.Header):
        header = fits.Header(list(header.items()))
    items = list(arrays.items())
    if layout == 'primary' and items:
        hdus = [fits.PrimaryHDU(data=items[0][1], header=header)]
        items = items[1:]
    elif layout == 'placeholder':
        hdus = [fits.PrimaryHDU(data=np.zeros(2), header=header)]
    else:
        hdus = [fits.PrimaryHDU(header=header)]
    hdus += [fits.ImageHDU(data=array, name=name.upper()) for name, array in items]
    hdul = fits.HDUList(hdus)
    try:
        hdul.writeto(path, overwrite=True)
    finally:
        hdul.close()  # Force close for Windows


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_object(path):
    '''
    Open a file written by write_object(), or a FITS file, for reading.
    Returns an ObjectReader, to be used as a context manager:

        with read_object(filename) as f:
            version = f.header['VERSION']
            data = f.read('data', 1)
    '''
    path = os.fspath(path)
    data_format = object_format(path)
    if data_format is None:
        is_npy = os.path.isdir(path) and os.path.exists(os.path.join(path, NPY_HEADER))
        data_format = 'npy' if is_npy else 'fits'
    return {'fits': _FitsReader, 'npz': _NpzReader,
            'npy': _NpyReader, 'hdf5': _Hdf5Reader}[data_format](path)


def _target_dtype(dtype, precision, path, name):
    '''
    Type of the arrays loaded with ObjectReader.load(), see load_fits_array():
    floating point and complex data have the type of the precision.
    '''
    if dtype.kind in 'fc':
        dtypes = iteras.cpu_float_dtype_list if dtype.kind == 'f' else iteras.cpu_complex_dtype_list
        return np.dtype(dtypes[iteras.global_precision if precision is None else precision])
    if precision is not None:
        raise ValueError(f'{path}: array {name} contains data that is not floating point '
                         f'or complex ({dtype}), but precision={precision} was requested')
    return dtype.newbyteorder('=')


def _to_device(array, target_device_idx, precision, path, name):
    '''Host array to the device and type of ObjectReader.load()'''
    if target_device_idx is None:
        target_device_idx = iteras.default_target_device_idx
    array = np.asarray(array, dtype=_target_dtype(array.dtype, precision, path, name))
    if target_device_idx < 0:
        return np.array(array)
    with iteras.cp.cuda.Device(target_device_idx):  # pragma: no cover
        return iteras.cp.asarray(array)


class ObjectReader:
    '''
    Header and named arrays of a file written by write_object(). Arrays are
    identified by name, and in FITS files also by the index of their HDU,
    used when no HDU has that name (files written by older versions) or
    when the name is None.
    '''

    def __init__(self, path):
        self.path = path
        self.header = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        pass

    def has(self, name, index=None):
        '''True if the array *name* (or HDU *index* in FITS files) exists'''
        raise NotImplementedError

    def names(self):
        '''
        Names of the arrays in the file. In FITS files, the names of the
        extensions (in upper case); the primary HDU, which has no name, is left out.
        '''
        raise NotImplementedError

    def read(self, name, index=None):
        '''The array *name* (or HDU *index* in FITS files), as a numpy array in native byte order'''
        raise NotImplementedError

    def load(self, name, index=None, target_device_idx=None, precision=None, transpose=False):
        '''
        The array *name* (or HDU *index*), on device *target_device_idx*
        (None for the default device, -1 for the CPU). Floating point and
        complex data are converted to the type of *precision* (None for the
        global precision); other data (integer, boolean) is kept, and a
        ValueError is raised if a precision is given.
        With *transpose*, a 2D array is returned transposed, in C order.
        Large arrays of FITS and npy files are streamed to the GPU in chunks
        (also when transposed, without a full-size temporary copy).
        '''
        array = self.read(name, index)
        if transpose:
            if array.ndim != 2:
                raise ValueError(f'{self.path}: transpose needs a 2D array, {name} has shape {array.shape}')
            array = np.ascontiguousarray(array.T)
        return _to_device(array, target_device_idx, precision, self.path, name)

    def _missing(self, name, index):
        where = name if index is None else f'{name} (HDU {index})'
        return KeyError(f'{self.path}: no array {where}')


class _FitsReader(ObjectReader):

    def __init__(self, path):
        from astropy.io import fits
        super().__init__(path)
        self._hdul = fits.open(path, memmap=False)
        self.header = self._hdul[0].header

    def close(self):
        self._hdul.close()

    def _index(self, name, index):
        if name is not None:
            for i, hdu in enumerate(self._hdul):
                if i > 0 and hdu.name == name.upper():
                    return i
        if index is not None and index < len(self._hdul):
            return index
        return None

    def has(self, name, index=None):
        i = self._index(name, index)
        # From the header: reading .data would load the whole array in host memory
        return i is not None and self._hdul[i].header.get('NAXIS', 0) > 0

    def names(self):
        return [hdu.name for hdu in self._hdul[1:] if hdu.name]

    def read(self, name, index=None):
        i = self._index(name, index)
        if i is None or self._hdul[i].data is None:
            raise self._missing(name, index)
        data = self._hdul[i].data
        return data.astype(data.dtype.newbyteorder('='))

    def load(self, name, index=None, target_device_idx=None, precision=None, transpose=False):
        from iteras.lib.fits_io import load_fits_array
        i = self._index(name, index)
        if i is None:
            raise self._missing(name, index)
        array = load_fits_array(self.path, ext=i, target_device_idx=target_device_idx,
                                precision=precision, transpose=transpose)
        if array is None:
            raise self._missing(name, index)
        return array


class _NpzReader(ObjectReader):

    def __init__(self, path):
        super().__init__(path)
        self._npz = np.load(path, allow_pickle=False)
        self.header = header_from_cards(json.loads(str(self._npz['__header__'])))

    def close(self):
        self._npz.close()

    def has(self, name, index=None):
        return name in self._npz.files

    def names(self):
        return [name for name in self._npz.files if name != '__header__']

    def read(self, name, index=None):
        if name not in self._npz.files:
            raise self._missing(name, index)
        return self._npz[name]


class _Hdf5Reader(ObjectReader):

    def __init__(self, path):
        import h5py
        super().__init__(path)
        self._file = h5py.File(path, 'r')
        self.header = header_from_cards(json.loads(self._file.attrs['__header__']))

    def close(self):
        self._file.close()

    def has(self, name, index=None):
        return name in self._file

    def names(self):
        return list(json.loads(self._file.attrs['__arrays__']))

    def read(self, name, index=None):
        if name not in self._file:
            raise self._missing(name, index)
        return self._file[name][()]


class _NpyReader(ObjectReader):

    def __init__(self, path):
        super().__init__(path)
        with open(os.path.join(path, NPY_HEADER)) as f:
            content = json.load(f)
        self.header = header_from_cards(content['header'])
        self._names = content['arrays']

    def _file(self, name):
        return os.path.join(self.path, name + '.npy')

    def has(self, name, index=None):
        return name in self._names

    def names(self):
        return list(self._names)

    def read(self, name, index=None):
        if name not in self._names:
            raise self._missing(name, index)
        return np.load(self._file(name), allow_pickle=False)

    def load(self, name, index=None, target_device_idx=None, precision=None, transpose=False):
        if name not in self._names:
            raise self._missing(name, index)
        if target_device_idx is None:
            target_device_idx = iteras.default_target_device_idx
        if target_device_idx < 0:
            return super().load(name, index, target_device_idx, precision, transpose)
        # GPU: stream the data from the file, as for FITS files
        with open(self._file(name), 'rb') as f:  # pragma: no cover
            version = np.lib.format.read_magic(f)
            if version not in ((1, 0), (2, 0)):
                return super().load(name, index, target_device_idx, precision, transpose)
            read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                           else np.lib.format.read_array_header_2_0)
            shape, fortran_order, dtype = read_header(f)
            offset = f.tell()
        if fortran_order or dtype.kind not in 'iuf' or (transpose and len(shape) != 2):  # pragma: no cover
            return super().load(name, index, target_device_idx, precision, transpose)
        from iteras.lib.fits_io import upload_from_file  # pragma: no cover
        return upload_from_file(self._file(name), offset, shape, dtype,  # pragma: no cover
                                _target_dtype(dtype, precision, self.path, name),
                                target_device_idx, byteswap=not dtype.isnative, transpose=transpose)


def read_header(path, data_format):
    '''Header of a stored file, as a dictionary'''
    if data_format == 'pickle':
        import pickle
        with open(path, 'rb') as handle:
            return header_to_dict(pickle.load(handle).get('hdr', {}))
    check_format(data_format)
    with read_object(path) as f:
        return header_to_dict(f.header)
