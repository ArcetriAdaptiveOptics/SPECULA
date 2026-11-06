import warnings

from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue
from iteras.lib import storage
from specula.lib.slopes_layout import SlopesLayout


class Slopes(BaseDataObj):
    """
    Slopes data object.
    Holds a slopes vector, which can be interleaved (XYXYXY...) or not (XXX...YYY...).
    X and Y slopes can be accessed independently, and a 2d map is available
    through :meth:`get2d` when a :class:`~specula.lib.slopes_layout.SlopesLayout`
    is set (*layout* argument or attribute), usually by the slope computer
    from its :class:`~specula.data_objects.subap_data.SubapData` or
    :class:`~specula.data_objects.pupdata.PupData`.
    """
    def __init__(self,
                 length: int=None,
                 slopes=None,
                 interleave: bool=False,
                 target_device_idx: int=None,
                 precision: int=None,
                 layout: SlopesLayout=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if slopes is not None:
            self.slopes = self.to_xp(slopes, dtype=self.dtype)
        else:
            self.slopes = self.xp.zeros(length, dtype=self.dtype)
        self.interleave = interleave
        self.layout = layout
        self._legacy_mask_shape = None
        self._legacy_map = None
        self.pupdata_tag = None
        self.subapdata_tag = None

        if self.interleave:
            self.indices_x = self.xp.arange(0, self.size // 2) * 2
            self.indices_y = self.indices_x + 1
        else:
            self.indices_x = self.xp.arange(0, self.size // 2)
            self.indices_y = self.xp.arange(self.size//2, self.size) # ensure that no element is discarded if self.size is odd

    def get_value(self):
        '''
        Get the slopes as anumpy/cupy array
        '''
        return self.slopes

    def set_value(self, v):
        '''
        Set new slopes values.
        Arrays are not reallocated
        '''
        assert v.shape == self.slopes.shape, \
            f"Error: input array shape {v.shape} does not match slopes shape {self.slopes.shape}"

        self.slopes[:] = self.to_xp(v)

    # TODO needed to support late SlopeC-derived class initialization
    # Replace with a full initialization in base class?
    def resize(self, new_size):
        """
        Resize the slopes array to a new size.

        Parameters
        ----------
        new_size : int
            The new size for the slopes array.

        Notes
        -----
        This method reallocates the slopes array to the specified new size and updates
        the indices for x and y slopes according to the interleave setting.
        """
        self.slopes = self.xp.zeros(new_size, dtype=self.dtype)
        if self.interleave:
            self.indices_x = self.xp.arange(0, self.size // 2) * 2
            self.indices_y = self.indices_x + 1
        else:
            self.indices_x = self.xp.arange(0, self.size // 2)
            self.indices_y = self.indices_x + self.size // 2

    @property
    def size(self):
        """
        Get the size of the slopes array.
        """
        return self.slopes.size

    @property
    def xslopes(self):
        """
        Get the x slopes as a numpy/cupy array.
        """
        return self.slopes[self.indices_x]

    @xslopes.setter
    def xslopes(self, value):
        """
        Set the x slopes.
        """
        self.slopes[self.indices_x] = value

    @property
    def yslopes(self):
        """
        Get the y slopes as a numpy/cupy array.
        """
        return self.slopes[self.indices_y]

    @yslopes.setter
    def yslopes(self, value):
        """
        Set the y slopes.
        """
        self.slopes[self.indices_y] = value

    def indx(self):
        """
        Get the indices for the x slopes.
        """
        return self.indices_x

    def indy(self):
        """
        Get the indices for the y slopes.
        """
        return self.indices_y

    def sum(self, s2, factor):
        """
        Sum the slopes with another slopes object.
        """
        self.slopes += s2.slopes * factor

    def subtract(self, s2):
        """
        Subtract another slopes object.
        """
        if isinstance(s2, Slopes):
            if s2.slopes.size > 0:
                self.slopes -= s2.slopes
            else:
                self.logger.warning('s2 (slopes) is empty!')
        elif isinstance(s2, BaseValue):  # Assuming BaseValue is another class
            if s2.value.size > 0:
                self.slopes -= s2.value
            else:
                self.logger.warning('s2 (base_value) is empty!')

    def x_remap2d(self, frame, idx):
        """
        Remap the x slopes to a 2d frame.
        """
        if len(idx.shape) == 1:
            self.xp.put(frame, idx, self.slopes[self.indx()])
        elif len(idx.shape) == 2:
            frame[idx[0], idx[1]] = self.slopes[self.indx()]
        else:
            raise ValueError('Frame index must be either 1d for flattened indexes or 2d')

    def y_remap2d(self, frame, idx):
        """
        Remap the y slopes to a 2d frame.
        """
        if len(idx.shape) == 1:
            self.xp.put(frame, idx, self.slopes[self.indy()])
        elif len(idx.shape) == 2:
            frame[idx[0], idx[1]] = self.slopes[self.indy()]
        else:
            raise ValueError('Frame index must be either 1d for flattened indexes or 2d')

    def get2d(self):
        """
        Get the slopes as a 2d map, using :attr:`layout`: an array of shape
        (2, ny, nx) with the X and Y slopes, or of shape (ny, nx) for layouts
        with one value per position (e.g. slopes from intensity).
        """
        if self.layout is None:
            raise ValueError('Slopes layout has not been set')
        return self.layout.to_2d(self.slopes, self.xp, dtype=self.dtype, interleave=self.interleave)

    # Deprecated single_mask / display_map pair, replaced by the layout.
    # They are properties, so they are not copied by transferDataTo().

    @property
    def single_mask(self):
        """Deprecated: frame with ones at the positions of :attr:`layout` (None without a layout)."""
        if self.layout is None:
            return None
        return self.layout.mask(self.xp, dtype=self.dtype)

    @single_mask.setter
    def single_mask(self, mask):
        warnings.warn('Slopes.single_mask is deprecated: set Slopes.layout instead '
                      '(e.g. subapdata.slopes_layout() or pupdata.slopes_layout())',
                      DeprecationWarning, stacklevel=2)
        self._legacy_mask_shape = None if mask is None else tuple(mask.shape)
        self._update_legacy_layout()

    @property
    def display_map(self):
        """Deprecated: flat frame indices of :attr:`layout` (None without a layout)."""
        if self.layout is None:
            return None
        return self.layout.device_idx(self.xp)

    @display_map.setter
    def display_map(self, idx):
        warnings.warn('Slopes.display_map is deprecated: set Slopes.layout instead '
                      '(e.g. subapdata.slopes_layout() or pupdata.slopes_layout())',
                      DeprecationWarning, stacklevel=2)
        self._legacy_map = None if idx is None else tuple(cpuArray(idx).tolist())
        self._update_legacy_layout()

    def _update_legacy_layout(self):
        if self._legacy_mask_shape is None or self._legacy_map is None:
            self.layout = None
        else:
            self.layout = SlopesLayout.from_mask_and_map(
                self._legacy_mask_shape, self._legacy_map, self.size)

    def transferDataTo(self, destobj, force_reallocation=False):
        super().transferDataTo(destobj, force_reallocation=force_reallocation)
        # The layout is static and shared, not copied
        destobj.layout = self.layout

    def get_metadata_arrays(self):
        """The :attr:`layout`, if set, as named arrays (see SlopesLayout.to_arrays())."""
        return {} if self.layout is None else self.layout.to_arrays()

    def set_metadata_arrays(self, arrays):
        """Restore the :attr:`layout` saved by :meth:`get_metadata_arrays`, if present."""
        layout = SlopesLayout.from_arrays(arrays)
        if layout is not None:
            self.layout = layout

    def rotate(self, angle, flipx=False, flipy=False):
        """
        Rotate the x and y slopes by a given angle, and optionally flip them along the x and/or y axes.

        Parameters
        ----------
        angle : float
            The angle (in degrees) by which to rotate the slopes.
        flipx : bool, optional
            If True, flip the x slopes (default: False).
        flipy : bool, optional
            If True, flip the y slopes (default: False).

        This operation modifies the xslopes and yslopes attributes in place.
        """
        sx = self.xslopes
        sy = self.yslopes
        alpha = self.xp.arctan2(sy, sx) + self.xp.radians(angle)
        modulus = self.xp.sqrt(sx**2 + sy**2)
        signx = -1 if flipx else 1
        signy = -1 if flipy else 1
        self.xslopes = self.xp.cos(alpha) * modulus * signx
        self.yslopes = self.xp.sin(alpha) * modulus * signy

    def get_fits_header(self):
        """
        Get the FITS header for the :class:`~specula.data_objects.slopes.Slopes` object.
        """
        hdr = fits.Header()
        hdr['VERSION'] = 3
        hdr['OBJ_TYPE'] = 'Slopes'
        hdr['INTRLVD'] = int(self.interleave)
        hdr['LENGTH'] = self.size
        hdr['PUPDTAG'] = self.pupdata_tag if self.pupdata_tag is not None else ''
        hdr['SUBAPTAG'] = self.subapdata_tag if self.subapdata_tag is not None else ''
        return hdr

    def save(self, filename, overwrite=True):
        """
        Save the :class:`~specula.data_objects.slopes.Slopes` object to a FITS file.

        The method writes the slopes data and relevant metadata to a FITS file.
        The main HDU contains only the header, and the slopes data is stored
        in an image extension named 'SLOPES'. The :attr:`layout`, if set, is
        stored in the following extensions (see :meth:`get_metadata_arrays`).

        Parameters
        ----------
        filename : str
            The path to the FITS file where the :class:`~specula.data_objects.slopes.Slopes` object will be saved.
        overwrite : bool, optional
            If True, overwrite the existing file if it exists (default: True).
        """
        storage.write_object(filename, self.get_fits_header(),
                             {'slopes': self.slopes, **self.get_metadata_arrays()},
                             overwrite=overwrite)
        
    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version not in [1, 2, 3]:
            raise ValueError(f"Error: unknown version {version} in header")
        interleave = bool(hdr['INTRLVD'])
        if version == 3:
            length = hdr['LENGTH']
            slopes = Slopes(length=length, interleave=interleave, target_device_idx=target_device_idx)
        else:
            slopes = Slopes(length=1, interleave=interleave, target_device_idx=target_device_idx)
        if version >= 2:
            slopes.pupdata_tag = hdr.get('PUPDTAG', None)
            slopes.subapdata_tag = hdr.get('SUBAPTAG', None)
        return slopes
    
    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            slopesdata = f.read('slopes', 1)
            metadata = {name: f.read(name) for name in f.names() if name.lower() != 'slopes'}
        slopes = Slopes.from_header(hdr, target_device_idx=target_device_idx)
        slopes.set_metadata_arrays(metadata)
        if hdr['VERSION'] >= 3:
            slopes.set_value(slopesdata)
        else:
            slopes.resize(len(slopesdata))  # version 2 header does not have length information
            slopes.slopes[:] = slopes.to_xp(slopesdata, dtype=slopes.dtype)
        return slopes

    def array_for_display(self):
        frames = self.get2d()
        # X and Y side by side; layouts with one value per position are already 2d
        return self.xp.hstack(frames) if frames.ndim == 3 else frames
