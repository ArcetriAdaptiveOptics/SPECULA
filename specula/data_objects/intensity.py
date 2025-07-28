
from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class Intensity(BaseDataObj):
    '''Intensity field object'''
    def __init__(self, 
                 dimx: int, 
                 dimy: int, 
                 target_device_idx: int=None, 
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
                
        self.i = self.xp.zeros((dimx, dimy), dtype=self.dtype)

    def get_value(self):
        '''
        Get the intensity field as a numpy/cupy array
        '''
        return self.i

    def set_value(self, v):
        '''
        Set new values for the intensity field    
        Arrays are not reallocated
        '''
        assert v.shape == self.i.shape, \
            f"Error: input array shape {v.shape} does not match intensity field shape {self.i.shape}"
        self.i[:]= self.to_xp(v, dtype=self.dtype)

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

    def array_for_display(self):
        return self.i