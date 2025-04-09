
from astropy.io import fits
from specula.base_data_obj import BaseDataObj

class BaseValue(BaseDataObj):
    def __init__(self, description='', value=None, target_device_idx=None):
        """
        Initialize the base value object.

        Parameters:
        description (str, optional)
        value (any, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(target_device_idx=target_device_idx)
        self._description = description
        self._value = value
        
    
    def set_value(self, val):
        self._value = val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val

    @property
    def ptr_value(self):
        return self._value

    @ptr_value.setter
    def ptr_value(self, val):
        self._value = val

    def save(self, filename):
        hdr = fits.Header()
        if self._value is not None:
            hdr['VALUE'] = str(self._value)  # Store as string for simplicity
        super().save(filename)
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            if self._value is not None:
                hdr['VALUE'] = str(self._value)
            hdul.flush()

    def read(self, filename):
        super().read(filename)
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            value_str = hdr.get('VALUE', None)
            if value_str is not None:
                self._value = eval(value_str)  # Convert back from string to original type

    def array_for_display(self):
        return self._value