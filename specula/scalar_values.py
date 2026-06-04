
from specula import np 
from astropy.io import fits
from specula.base_data_obj import BaseDataObj


class _BaseScalarValue(BaseDataObj):
    '''
    Base class for scalar values.
    Internal value is guaranteed a valid instance of the requested type based on the derived class.
    '''
    def __init__(self, type_, value, description=''):
        """
        Base class for scalar values.

        This class ensures that the stored value is always of the expected Python
        scalar type (int, float, or str), as defined by subclasses.

        Parameters
        ----------
        type_ : type
            Expected Python type of the scalar value (e.g., int, float, str).
        value : any
            Value to store. Must match `type_`.
        description : str, optional
            Human-readable description of the value.
        """
        super().__init__()

        self.description = description
        self.type = type_
        self.set_value(value)

    def get_value(self):
        """
        Return the stored scalar value.

        Returns
        -------
        any
            The stored value.
        """
        return self.value

    def set_value(self, val):
        """
        Set the scalar value with type validation.

        Parameters
        ----------
        val : any
            New value to assign. Must match `self.type`.

        Raises
        ------
        AssertionError
            If `val` is not of the expected type.
        """
        assert isinstance(val, self.type)
        self.value = val

    def save(self, filename, overwrite=False):
        """
        Save the scalar value to a FITS file.

        The value is stored as a string in the FITS header.

        Parameters
        ----------
        filename : str
            Path to the output FITS file.
        overwrite : bool, optional
            Whether to overwrite an existing file.
        """
        hdr = self.get_fits_header()
        data = np.zeros(2)
        hdr['VALUE'] = str(self.value)      # Store as string for simplicity
        hdr['DESC'] = self.description
        fits.writeto(filename, data, hdr, overwrite=overwrite)

    @classmethod
    def restore(cls, filename):
        """
        Restore a scalar value object from a FITS file.

        Parameters
        ----------
        filename : str
            Path to the FITS file.

        Returns
        -------
        _BaseScalarValue
            Restored instance of the subclass.

        Raises
        ------
        ValueError
            If required FITS header keywords are missing.
        """
        hdr = fits.getheader(filename)
        value_str = hdr.get('VALUE', None)
        desc_str = hdr.get('DESC', None)
        if value_str is None:
            raise ValueError('FITS header does not contain a valid VALUE keyword')
        if desc_str is None:
            raise ValueError('FITS header does not contain a valid DESC keyword')

        value = cls.__init__.__annotations__['value'](value_str)
        return cls(value=value, description=desc_str)

    def array_for_display(self):
        return self.value

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = self.__class__.__name__
        return hdr


class IntValue(_BaseScalarValue):

    def __init__(self, value: int, description=''):
        """
        Integer value. Scalar container for integer values.

        This class stores a single integer value with an optional description.

        Parameters
        ----------
        value : int
            Integer value to store.
        description : str, optional
            Human-readable description of the value. Default is an empty string.
        """
        super().__init__(description=description,
                         type_=int,
                         value=value)


class FloatValue(_BaseScalarValue):

    def __init__(self, value: float, description=''):
        """
        Floatint point value. Scalar container for floating point values.

        This class stores a single floating point value with an optional description.

        Parameters
        ----------
        value : float
            Floating point value to store.
        description : str, optional
            Human-readable description of the value. Default is an empty string.
        """
        super().__init__(description=description,
                         type_=float,
                         value=value)


class StringValue(_BaseScalarValue):

    def __init__(self, value: str, description=''):
        """
        String value. Scalar container for string values.

        This class stores a single string value with an optional description.

        Parameters
        ----------
        value : str
            String value to store.
        description : str, optional
            Human-readable description of the value. Default is an empty string.
        """
        super().__init__(description=description,
                         type_=str,
                         value=value)



