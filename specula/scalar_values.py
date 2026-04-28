
from specula import np, array_types, cpuArray
from astropy.io import fits
from specula.base_data_obj import BaseDataObj
from types import NoneType

class _BaseScalarValue(BaseDataObj):
    '''
    Base class for scalar values.
    Internal value is guaranteed a valid instance of the requested type based on the derived class.
    '''
    def __init__(self, type_, value, description='',target_device_idx=None, precision=None):
        """

        Parameters:
        type_: scalar value type (int, float, str)
        value: data to store.
        description (str, optional)
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.description = description
        self.type = type_
        self.set_value(value)

    def get_value(self):
        return self.value

    def set_value(self, val):
        assert isinstance(val, self.type)
        self.value = val

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        data = np.zeros(2)
        hdr['VALUE'] = str(self.value)  # Store as string for simplicity
        fits.writeto(filename, data, hdr, overwrite=overwrite)

    @classmethod
    def restore(cls, filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        value_str = hdr.get('VALUE')
        if value_str is None:
            raise ValueError('FITS header does not contain a valid VALUE keyword')

        value = cls.__init__.__annotations__['value'](value_str)
        return cls(value=value, target_device_idx=target_device_idx)

    def array_for_display(self):
        return self.value

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = self.__class__.__name__
        return hdr


class IntValue(_BaseScalarValue):

    def __init__(self, description='', value: int=None, target_device_idx=None, precision=None):
        """
        Parameters:
        description (str, optional)
        value (int, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(description=description,
                         type_=int,
                         value=value,
                         target_device_idx=target_device_idx,
                         precision=precision)


class FloatValue(_BaseScalarValue):

    def __init__(self, description='', value: float=None, target_device_idx=None, precision=None):
        """
        Parameters:
        description (str, optional)
        value (float, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(description=description,
                         type_=float,
                         value=value,
                         target_device_idx=target_device_idx,
                         precision=precision)


class StringValue(_BaseScalarValue):

    def __init__(self, description='', value: str=None, target_device_idx=None, precision=None):
        """
        Parameters:
        description (str, optional)
        value (str, optional): data to store. If not set, the value is initialized to None.
        """
        super().__init__(description=description,
                         type_=str,
                         value=value,
                         target_device_idx=target_device_idx,
                         precision=precision)



