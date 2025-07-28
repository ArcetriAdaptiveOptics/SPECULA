
from astropy.io import fits

from specula import cpuArray
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

    def __str__(self):
        return str(self.i)

    @property
    def size(self):
        return self.i.shape

    def sum(self, i2, factor=1.0):
        self.i += i2.i * factor

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Intensity'
        hdr['DIMX'] = self.i.shape[0]
        hdr['DIMY'] = self.i.shape[1]
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.i), name='INTENSITY'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows
        
    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        intensity = Intensity(dimx, dimy, target_device_idx=target_device_idx)
        return intensity
    
    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Intensity':
            raise ValueError(f"Error: file {filename} does not contain an Intensity object")
        intensity = Intensity.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            intensity.i = intensity.to_xp(hdul[1].data.copy())
        return intensity

    def array_for_display(self):
        return self.i