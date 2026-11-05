import numpy as np

from specula import cpuArray


class SlopesLayout:
    """
    Static map between a slopes vector and its 2d display frame(s).

    A layout belongs to the geometry of a wavefront sensor and does not change
    during a simulation: it is built once by the object that knows that
    geometry (:meth:`SubapData.slopes_layout() <specula.data_objects.subap_data.SubapData.slopes_layout>`,
    :meth:`PupData.slopes_layout() <specula.data_objects.pupdata.PupData.slopes_layout>`,
    :meth:`full_frame`) and given to :class:`~specula.data_objects.slopes.Slopes`,
    whose :meth:`~specula.data_objects.slopes.Slopes.get2d` uses it.

    Two kinds of layout exist:

    - ``'xy'``: the vector holds X and Y slopes, one pair per position of
      *idx*; :meth:`to_2d` returns an array of shape ``(2,) + shape``;
    - ``'scalar'``: the vector holds one value per position of *idx* (e.g.
      slopes from intensity, curvature signals, OPD maps); :meth:`to_2d`
      returns an array of shape *shape*.

    The layout is read only and stored on the CPU. Copies of the index array
    and of the derived indices on the GPU are made once per device and cached. It is not a data object
    array, so it is not copied by ``transferDataTo()`` at every step: data
    objects on other devices share the same layout.

    Parameters
    ----------
    shape : tuple of int
        Shape of the 2d frame.
    idx : array_like of int
        Flat indices into a frame of shape *shape*, one per subaperture
        (``'xy'``) or per value (``'scalar'``), in the order of the vector.
    kind : str
        ``'xy'`` or ``'scalar'``.
    """

    XY = 'xy'
    SCALAR = 'scalar'
    KINDS = (XY, SCALAR)

    # Names of the arrays of to_arrays() / from_arrays()
    IDX_NAME = 'LAYOUT_IDX'
    SHAPE_NAME = 'LAYOUT_SHAPE'
    KIND_NAME = 'LAYOUT_KIND'

    def __init__(self, shape, idx, kind):
        if kind not in self.KINDS:
            raise ValueError(f'Unknown layout kind {kind!r}: must be one of {", ".join(self.KINDS)}')
        self.shape = tuple(int(s) for s in shape)
        if len(self.shape) != 2:
            raise ValueError(f'Layout shape must be 2d, got {self.shape}')
        idx = np.asarray(cpuArray(idx))
        if idx.ndim != 1:
            raise ValueError(f'Layout indices must be a 1d array of flat indices, got shape {idx.shape}')
        idx = idx.astype(np.int64)
        npix = self.shape[0] * self.shape[1]
        if idx.size and (idx.min() < 0 or idx.max() >= npix):
            raise ValueError(f'Layout indices out of range for a frame of shape {self.shape}')
        idx.flags.writeable = False
        self.idx = idx
        self.kind = kind
        self._device_idx = {}

    # Factories -----------------------------------------------------------

    @classmethod
    def full_frame(cls, shape):
        """Layout of a vector holding a whole 2d frame of *shape*, in row-major order."""
        shape = tuple(int(s) for s in shape)
        return cls(shape, np.arange(shape[0] * shape[1]), cls.SCALAR)

    @classmethod
    def from_mask_and_map(cls, mask_shape, idx, length):
        """
        Layout from the former ``single_mask``/``display_map`` pair of
        :class:`~specula.data_objects.slopes.Slopes`: the kind is inferred
        from the length of the slopes vector, as ``Slopes.get2d()`` did.
        *idx* can be a 1d array of flat indices or a 2d array of (row, column) indices.
        """
        idx = np.asarray(cpuArray(idx))
        if idx.ndim == 2:
            idx = np.ravel_multi_index((idx[0], idx[1]), mask_shape)
        kind = cls.SCALAR if length == idx.size else cls.XY
        return cls(mask_shape, idx, kind)

    # Properties ----------------------------------------------------------

    @property
    def n_positions(self):
        """Number of positions in the frame used by the layout."""
        return self.idx.size

    @property
    def n_values(self):
        """Length of the vector described by this layout."""
        return self.idx.size * (2 if self.kind == self.XY else 1)

    def mask(self, xp=np, dtype=np.float32):
        """Frame of *shape* with ones at the positions used by the layout."""
        f = xp.zeros(self.shape[0] * self.shape[1], dtype=dtype)
        f[self.device_idx(xp)] = 1
        return f.reshape(self.shape)

    def device_idx(self, xp):
        """The index array as an *xp* array, on the current device (cached)."""
        return self._cached(xp, 'idx', lambda: self.idx)

    def value_idx(self, xp=np, interleave=False):
        """
        Flat index, into the output of :meth:`to_2d`, of each element of the
        vector (cached): ``to_2d(v).ravel()[value_idx()] == v``.
        """
        def build():
            if self.kind == self.SCALAR:
                return self.idx
            npix = self.shape[0] * self.shape[1]
            if interleave:
                return np.stack([self.idx, self.idx + npix], axis=1).ravel()
            return np.concatenate([self.idx, self.idx + npix])
        return self._cached(xp, ('value_idx', bool(interleave)), build)

    def _cached(self, xp, name, build):
        if xp is np:
            key = (name, None)
        else:
            key = (name, xp.cuda.runtime.getDevice())
        if key not in self._device_idx:
            array = build()
            self._device_idx[key] = array if xp is np else xp.asarray(array)
        return self._device_idx[key]

    @property
    def frame_shape(self):
        """Shape of the output of :meth:`to_2d`."""
        return self.shape if self.kind == self.SCALAR else (2,) + self.shape

    # Conversions ---------------------------------------------------------

    def _check_length(self, n):
        if n != self.n_values:
            raise ValueError(f'Vector of length {n} does not match a {self.kind!r} layout '
                             f'of {self.n_positions} positions ({self.n_values} values)')

    def to_2d(self, vector, xp=np, dtype=None, interleave=False):
        """
        Vector to 2d frame(s): an array of shape ``(2,) + shape`` (X, Y) for
        ``'xy'`` layouts, of shape *shape* for ``'scalar'`` layouts.
        Positions not used by the layout are zero.
        """
        self._check_length(vector.size)
        dtype = vector.dtype if dtype is None else dtype
        f = xp.zeros(self.frame_shape, dtype=dtype)
        f.reshape(-1)[self.value_idx(xp, interleave)] = vector
        return f

    def from_2d(self, frames, xp=np, interleave=False):
        """Inverse of :meth:`to_2d`: the vector read from 2d frame(s)."""
        if tuple(frames.shape) != self.frame_shape:
            raise ValueError(f'Frames shape {tuple(frames.shape)} does not match '
                             f'the layout frame shape {self.frame_shape}')
        return frames.reshape(-1)[self.value_idx(xp, interleave)]

    # Storage -------------------------------------------------------------

    def to_arrays(self):
        """The layout as named numpy arrays, for files (see :meth:`from_arrays`)."""
        return {self.IDX_NAME: self.idx.copy(),
                self.SHAPE_NAME: np.array(self.shape, dtype=np.int64),
                self.KIND_NAME: np.array([self.KINDS.index(self.kind)], dtype=np.int64)}

    @classmethod
    def from_arrays(cls, arrays):
        """
        Layout from the arrays of :meth:`to_arrays` (names are case
        insensitive, other arrays are ignored), or None if they are missing.
        """
        upper = {name.upper(): value for name, value in arrays.items()}
        if not all(name in upper for name in (cls.IDX_NAME, cls.SHAPE_NAME, cls.KIND_NAME)):
            return None
        kind = cls.KINDS[int(np.asarray(upper[cls.KIND_NAME]).ravel()[0])]
        return cls(np.asarray(upper[cls.SHAPE_NAME]).ravel(), np.asarray(upper[cls.IDX_NAME]).ravel(), kind)

    # Python protocol -----------------------------------------------------

    def __eq__(self, other):
        if not isinstance(other, SlopesLayout):
            return NotImplemented
        return (self.kind == other.kind and self.shape == other.shape
                and np.array_equal(self.idx, other.idx))

    __hash__ = None

    def __getstate__(self):
        # GPU copies are not pickled (e.g. for MPI): they are rebuilt on demand
        state = self.__dict__.copy()
        state['_device_idx'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.idx.flags.writeable = False

    def __repr__(self):
        return f'SlopesLayout(kind={self.kind!r}, shape={self.shape}, n_positions={self.n_positions})'
