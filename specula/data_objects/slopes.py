from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue


class Slopes(BaseDataObj):
    '''
    Slopes data object. Holds a slopes vector, which can be interleaved (XYXYXY...) or not (XXX...YYY...).
    X and Y slopes can be accessed independently, and a 2d map is available.
    '''
    def __init__(self, 
                 length: int=None,
                 slopes=None,
                 interleave: bool=False,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if slopes is not None:
            self.slopes = self.to_xp(slopes, dtype=self.dtype)
        else:
            self.slopes = self.xp.zeros(length, dtype=self.dtype)
        self.interleave = interleave
        self.single_mask = None
        self.display_map = None
        self.pupdata_tag = None
        self.subapdata_tag = None

        if self.interleave:
            self.indices_x = self.xp.arange(0, self.size // 2) * 2
            self.indices_y = self.indices_x + 1
        else:
            self.indices_x = self.xp.arange(0, self.size // 2)
            self.indices_y = self.indices_x + self.size // 2

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
                print('WARNING (slopes object): s2 (slopes) is empty!')
        elif isinstance(s2, BaseValue):  # Assuming BaseValue is another class
            if s2.value.size > 0:
                self.slopes -= s2.value
            else:
                print('WARNING (slopes object): s2 (base_value) is empty!')

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
        Get the slopes as a 2d map.
        """
        if self.single_mask is None:
            raise ValueError('Slopes single_mask has not been set')
        if self.display_map is None:
            raise ValueError('Slopes display_map has not been set')
        mask = self.single_mask
        idx = self.display_map
        if self.slopes.size == len(idx):
            # slopes from intensity case
            f = self.xp.zeros_like(mask, dtype=self.dtype)
            if len(idx.shape) == 1:
                self.xp.put(f.ravel(), idx, self.slopes)
            elif len(idx.shape) == 2:
                f[idx] = self.slopes
            return self.to_xp(f, dtype=self.dtype)
        else:
            fx = self.xp.zeros_like(mask, dtype=self.dtype)
            fy = self.xp.zeros_like(mask, dtype=self.dtype)
            self.x_remap2d(fx, idx)
            self.y_remap2d(fy, idx)
            return self.to_xp([fx, fy], dtype=self.dtype)

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
        hdr['OBJ_TYPE'] = 'Intensity'
        hdr['INTRLVD'] = int(self.interleave)
        hdr['LENGTH'] = self.size
        hdr['PUPDTAG'] = self.pupdata_tag if self.pupdata_tag is not None else ''
        hdr['SUBAPTAG'] = self.subapdata_tag if self.subapdata_tag is not None else ''
        return hdr

    def save(self, filename, overwrite=True):
        """
        Save the :class:`~specula.data_objects.slopes.Slopes` object to a FITS file.

        Parameters
        ----------
        filename : str
            The path to the FITS file where the :class:`~specula.data_objects.slopes.Slopes` object will be saved.
        overwrite : bool, optional
            If True, overwrite the existing file if it exists (default: True).

        The method writes the slopes data and relevant metadata to a FITS file.
        The main HDU contains only the header, and the slopes data is stored
        in an image extension named 'SLOPES'.
        """
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.slopes), name='SLOPES'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows
        
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
        hdr = fits.getheader(filename)
        slopes = Slopes.from_header(hdr, target_device_idx=target_device_idx)
        slopesdata = fits.getdata(filename, ext=1)
        if hdr['VERSION'] >= 3:
            slopes.set_value(slopesdata)
        else:
            slopes.resize(len(slopesdata))  # version 2 header does not have length information
            slopes.slopes = slopes.to_xp(slopesdata, dtype=slopes.dtype)
        return slopes

    def array_for_display(self):
        return self.xp.hstack(self.get2d())
