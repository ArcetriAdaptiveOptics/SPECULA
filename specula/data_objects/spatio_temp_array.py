from astropy.io import fits
import numpy as np
from specula import cpuArray

from specula.base_data_obj import BaseDataObj


class SpatioTempArray(BaseDataObj):
    """
    Spatio-temporal array data object.
    This class holds a multi-dimensional spatio-temporal array with an associated time vector.
    The temporal dimension can be on the last axis, and multiple spatial dimensions are supported.
    array[..., i] is associated with time_vector[i].
    """
    def __init__(self,
                 array,
                 time_vector,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize a SpatioTempArray object.

        Parameters
        ----------
        array : array-like
            N-dimensional array with temporal evolution on the last axis.
            Can be 1D (time only), 2D (spatial + time), 3D (spatial + spatial + time), etc.
            Typically in nm for phase screens.
        time_vector : array-like
            1D array of time values corresponding to the last axis of array.
            Must have length equal to array.shape[-1].
        target_device_idx : int, optional
            Device to be targeted for data storage. Set to -1 for CPU,
            to 0 for the first GPU device, 1 for the second GPU device, etc.
            Default is None (uses global setting).
        precision : int, optional
            Precision setting. If None will use the global_precision,
            otherwise set to 0 for double, 1 for single.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.array = self.to_xp(array)
        self.time_vector = self.to_xp(time_vector)

        if self.array.shape[-1] != self.time_vector.shape[0]:
            raise ValueError(
                f"Last dimension of array ({self.array.shape[-1]}) must match "
                f"length of time_vector ({self.time_vector.shape[0]})"
            )

    def get_value(self):
        """Get the array data."""
        return self.array

    def set_value(self, val):
        """Set the array data in-place."""
        self.array[...] = self.to_xp(val)

    def get_time_vector(self):
        """Get the time vector."""
        return self.time_vector

    def set_time_vector(self, val):
        """Set the time vector in-place."""
        self.time_vector[...] = self.to_xp(val)

    def save(self, filename):
        """
        Save the SpatioTempArray data to a FITS file.

        The array is stored as primary HDU and the time vector as an extension.
        """
        hdr = self.get_fits_header()

        # Primary HDU with array
        primary_hdu = fits.PrimaryHDU(cpuArray(self.array), header=hdr)

        # Extension HDU with time vector
        time_hdu = fits.ImageHDU(cpuArray(self.time_vector), name='TIME_VECTOR')

        hdul = fits.HDUList([primary_hdu, time_hdu])
        hdul.writeto(filename, overwrite=True)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """
        Restore a SpatioTempArray object from a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file created by save().
        target_device_idx : int, optional
            Device to be targeted for data storage.

        Returns
        -------
        SpatioTempArray
            Restored object.
        """
        hdul = fits.open(filename)

        hdr = hdul[0].header # pylint: disable=invalid-name
        version = hdr.get('VERSION')
        if version != 1:
            raise ValueError(f"Unknown version {version} in file {filename}")

        array = hdul[0].data # pylint: disable=invalid-name
        time_vector = hdul['TIME_VECTOR'].data # pylint: disable=invalid-name

        hdul.close()

        return SpatioTempArray(array, time_vector, target_device_idx=target_device_idx)

    def array_for_display(self):
        """Return the array data for display purposes."""
        return self.array

    def get_fits_header(self):
        """
        Get the FITS header for saving.
        
        Uses abbreviated keywords to comply with FITS standard (max 8 characters).
        Saves shape as space-separated string in ARSHAPE comment for readability.
        """
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'SpatioTempArray'

        # Store shape as space-separated dimensions  
        shape_str = ' '.join(str(d) for d in self.array.shape)
        hdr['ARSHAPE'] = shape_str
        hdr.add_comment(f"Array shape: {self.array.shape}", before='ARSHAPE')

        hdr['NTIME'] = self.time_vector.shape[0]
        return hdr

    @staticmethod
    def from_header(hdr, target_device_idx=None, precision=None):
        """
        Create empty SpatioTempArray from FITS header metadata.
        
        This creates an object with uninitialised arrays of the correct shape
        based on the header metadata (used for pre-allocation before loading data).

        Parameters
        ----------
        hdr : astropy.io.fits.Header
            FITS header containing ARSHAPE and NTIME metadata.
        target_device_idx : int, optional
            Device to be targeted for data storage.
        precision : int, optional
            Precision setting.

        Returns
        -------
        SpatioTempArray
            Object with uninitialised arrays of correct shape and time vector length.
        """
        version = hdr.get('VERSION')
        if version != 1:
            raise ValueError(f"Unknown version {version} in header")

        arshape_str = hdr.get('ARSHAPE')
        ntime = hdr.get('NTIME')

        if arshape_str is None or ntime is None:
            raise ValueError("Missing ARSHAPE or NTIME in header")

        # Parse shape string: "10 10 5" -> (10, 10, 5)
        array_shape = tuple(int(d) for d in str(arshape_str).split())

        # Create empty arrays with correct shape
        temp_obj = SpatioTempArray.__new__(SpatioTempArray)
        BaseDataObj.__init__(temp_obj, target_device_idx=target_device_idx, precision=precision)

        temp_obj.array = temp_obj.xp.empty(array_shape, dtype=temp_obj.dtype)
        temp_obj.time_vector = temp_obj.xp.empty(ntime, dtype=temp_obj.dtype)

        return temp_obj
