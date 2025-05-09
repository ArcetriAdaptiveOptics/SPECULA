from astropy.io import fits
import numpy as np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField

class Layer(ElectricField):
    '''Layer'''

    def __init__(self, 
                 dimx: int,
                 dimy: int,
                 pixel_pitch: float,
                 height: float,
                 shiftXYinPixel: tuple=(0.0, 0.0),
                 rotInDeg: float=0.0, 
                 magnification: float=1.0,
                 target_device_idx: int=None, 
                 precision: int=None):
        super().__init__(dimx, dimy, pixel_pitch, target_device_idx=target_device_idx, precision=precision)
        self.height = height
        self.shiftXYinPixel = cpuArray(shiftXYinPixel).astype(self.dtype)
        self.rotInDeg = rotInDeg
        self.magnification = magnification

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Layer'
        hdr['HEIGHT'] = self.height
        return hdr

    def save(self, filename):
        super().save(filename)
        fits.append(filename, np.zeros(1), self.get_fits_header())

    def read(self, filename):
        # Nothing extra to read
        super().read(filename)
 
    @staticmethod
    def restore(filename):
        ef_hdr = fits.getheader(filename, ext=0)
        layer_hdr = fits.getheader(filename, ext=2)

        version = layer_hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")

        dimx = ef_hdr['DIMX']
        dimy = ef_hdr['DIMY']
        pitch = ef_hdr['PIXPITCH']
        height = layer_hdr['HEIGHT']
        layer = Layer(dimx, dimy, pitch, height)
        layer.read(filename)
        return layer

