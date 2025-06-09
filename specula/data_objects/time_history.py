
from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj


class TimeHistory(BaseDataObj):
    '''Time history'''

    def __init__(self,
                 time_history,
                 target_device_idx: int=None,
                 precision:int =None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.time_history = self.to_xp(time_history)

    def save(self, filename):
        """Saves the subaperture data to a file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        fits.writeto(filename, cpuArray(self.time_history), hdr)

    @classmethod
    def restore(cls, filename, target_device_idx=None):
        """Restores the time history data from a file."""
        hdr = fits.getheader(filename)
        version = hdr.get('VERSION')
        if version != 1:
            raise ValueError(f"Unknown version {version} in file {filename}")
        data = fits.getdata(filename)
        return TimeHistory(data, target_device_idx=target_device_idx)
