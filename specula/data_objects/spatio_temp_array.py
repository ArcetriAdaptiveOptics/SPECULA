import os
from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj


class SpatioTempArray(BaseDataObj):
    """
    Spatio-temporal array data object.
    This class holds a multi-dimensional spatio-temporal array with an associated time vector.
    Input arrays can have temporal evolution on the first or last axis.
    Internally, data are always stored in time-first layout:
    array[i, ...] is associated with time_vector[i].
    """
    def __init__(self,
                 array,
                 time_vector,
                 time_axis: int = -1,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize a SpatioTempArray object.

        Parameters
        ----------
        array : array-like
            N-dimensional array with temporal evolution on first or last axis.
            Can be 1D (time only), 2D (spatial + time), 3D (spatial + spatial + time), etc.
            Typically in nm for phase screens.
        time_vector : array-like
            1D array of time values corresponding to the selected temporal axis of array.
            Must have length equal to array.shape[time_axis].
        time_axis : int, optional
            Temporal axis of input array. Supported values are 0 (time-first) and -1 (time-last).
            Internal storage is always time-first. Default is -1.
        target_device_idx : int, optional
            Device to be targeted for data storage. Set to -1 for CPU,
            to 0 for the first GPU device, 1 for the second GPU device, etc.
            Default is None (uses global setting).
        precision : int, optional
            Precision setting. If None will use the global_precision,
            otherwise set to 0 for double, 1 for single.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.time_vector = self.to_xp(time_vector)
        input_array = self.to_xp(array)

        if time_axis not in (0, -1):
            raise ValueError(f"Unsupported time_axis={time_axis}. Supported values are 0 and -1")

        if input_array.ndim == 0:
            raise ValueError("array must have at least one dimension")

        if input_array.shape[time_axis] != self.time_vector.shape[0]:
            raise ValueError(
                f"Selected temporal dimension of array ({input_array.shape[time_axis]}) must match "
                f"length of time_vector ({self.time_vector.shape[0]})"
            )

        if time_axis == -1 and input_array.ndim > 1:
            self.array = self.xp.moveaxis(input_array, -1, 0)
        else:
            self.array = input_array

        self.array = self.xp.ascontiguousarray(self.array)

    def get_value(self):
        """Get array data in internal time-first layout: array[time, ...]."""
        return self.array

    def set_value(self, val):
        """Set array data in-place accepting time-first or time-last layout."""
        arr = self.to_xp(val)

        if arr.shape == self.array.shape:
            self.array[...] = arr
            return

        if arr.ndim == self.array.ndim and arr.ndim > 1 and arr.shape[-1] == self.array.shape[0]:
            converted = self.xp.moveaxis(arr, -1, 0)
            if converted.shape == self.array.shape:
                self.array[...] = converted
                return

        raise ValueError(
            f"Input shape {arr.shape} is incompatible with internal shape {self.array.shape}. "
            "Expected time-first shape or time-last equivalent."
        )

    def get_time_vector(self):
        """Get the time vector."""
        return self.time_vector

    def set_time_vector(self, val):
        """Set the time vector in-place."""
        self.time_vector[...] = self.to_xp(val)

    def save(self, filename):
        """
        Save the SpatioTempArray data to a FITS file.

        The array is stored in internal time-first layout as primary HDU
        and the time vector as an extension.
        """
        filename = os.fspath(filename)
        hdr = self.get_fits_header()

        # Primary HDU with array
        primary_hdu = fits.PrimaryHDU(cpuArray(self.array), header=hdr)

        # Extension HDU with time vector
        time_hdu = fits.ImageHDU(cpuArray(self.time_vector), name='TIME_VECTOR')

        hdul = fits.HDUList([primary_hdu, time_hdu])
        try:
            hdul.writeto(filename, overwrite=True)
        finally:
            hdul.close()

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
        filename = os.fspath(filename)
        with fits.open(filename, memmap=False) as hdul:
            hdr = hdul[0].header  # pylint: disable=no-member
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")

            array = hdul[0].data.copy()  # pylint: disable=no-member
            time_vector = hdul['TIME_VECTOR'].data.copy()  # pylint: disable=no-member

        return SpatioTempArray(array, time_vector, time_axis=0, target_device_idx=target_device_idx)

    def array_for_display(self):
        """Return the array data for display purposes."""
        return self.array

    def get_fits_header(self):
        """
        Get the FITS header for saving.

        Uses abbreviated keywords to comply with FITS standard (max 8 characters).
        Saves shape as space-separated string in ARSHAPE comment for readability.
        Internal temporal axis is stored in TAXIS (always 0).
        """
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'SpatioTempArray'

        # Store shape as space-separated dimensions
        shape_str = ' '.join(str(d) for d in self.array.shape)
        hdr['ARSHAPE'] = shape_str
        hdr.add_comment(f"Array shape: {self.array.shape}", before='ARSHAPE')

        hdr['NTIME'] = self.time_vector.shape[0]
        hdr['TAXIS'] = 0
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
