
from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj


class TimeHistory(BaseDataObj):
    '''Time history'''

    def __init__(self,
                 time_history,
                 target_device_idx: int=None,
                 precision:int =None):
        """
        Initialize a :class:`~specula.data_objects.time_history.TimeHistory` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.time_history = self.to_xp(time_history)

    def get_value(self):
        return self.time_history

    def set_value(self, val):
        self.time_history[...] = self.to_xp(val)

    def save(self, filename):
        """Saves the :class:`~specula.data_objects.time_history.TimeHistory` data to a file."""
        hdr = self.get_fits_header()
        fits.writeto(filename, cpuArray(self.time_history), hdr)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the :class:`~specula.data_objects.time_history.TimeHistory` data from a file."""
        hdr = fits.getheader(filename)
        version = hdr.get('VERSION')
        if version != 1:
            raise ValueError(f"Unknown version {version} in file {filename}")
        data = fits.getdata(filename)
        return TimeHistory(data, target_device_idx=target_device_idx)

    def array_for_display(self):
        return self.time_history
    
    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'TimeHistory'
        return hdr
    
    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError
