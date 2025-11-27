
from specula import np, array_types, cpuArray
from astropy.io import fits
from specula.base_data_obj import BaseDataObj

class BaseValue(BaseDataObj):
    def __init__(self, description='', value=None, target_device_idx=None, precision=None):
        """
        Initialize the base value object.

        Parameters:
        description (str, optional)
        value (any, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.description = description
        if value is not None:
            # if it is a scalar, convert to the appropriate scalar type
            if np.isscalar(value):
                self.value = self.dtype(value)
            else:
                self.value = self.to_xp(value, force_copy=True, dtype=self.dtype)
        else:
            self.value = None

    def get_value(self):
        return self.value

    def set_value(self, val):
        if self.value is not None:
            if np.isscalar(self.value):
                self.value = self.dtype(val)
            else:
                self.value[...] = self.to_xp(val)
        else:
            self.value = self.to_xp(val, force_copy=True, dtype=self.dtype)

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()

        if type(self.value) in array_types:
            data = cpuArray(self.value)
            hdr['NDARRAY'] = 1
        else:
            data = np.zeros(2)
            hdr['NDARRAY'] = 0
            if self.value is not None:
                hdr['VALUE'] = str(self.value)  # Store as string for simplicity
        fits.writeto(filename, data, hdr, overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        data = fits.getdata(filename)
        v = BaseValue(target_device_idx=target_device_idx)

        if hdr['NDARRAY']:
            v.value = data
        else:
            value_str = hdr.get('VALUE', None)
            if value_str is not None:
                v.value = eval(value_str)  # Convert back from string to original type
        return v

    def array_for_display(self):
        return self.value

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseValue'
        return hdr
