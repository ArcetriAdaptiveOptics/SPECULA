
from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class Intensity(BaseDataObj):
    '''Intensity field object'''
    def __init__(self, dimx, dimy, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
                
        self.i = self.xp.zeros((dimx, dimy), dtype=self.dtype)

    @property
    def size(self):
        return self.i.shape

    def sum(self, i2, factor=1.0):
        self.i += i2.i * factor

    def save(self, filename, hdr):
        hdr = fits.Header()
        hdr.append(('VERSION', 1))
        super().save(filename, hdr)
        fits.writeto(filename, self.i, hdr, overwrite=True)

    def read(self, filename):
        hdr = fits.getheader(filename)
        super().read(filename, hdr)
        self.i = fits.getdata(filename)
