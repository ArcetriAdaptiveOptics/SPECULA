from astropy.io import fits
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField

class Layer(ElectricField):
    '''Layer'''

    def __init__(self, dimx, dimy, pixel_pitch, height, target_device_idx=None, precision=None, shiftXYinPixel=(0.0, 0.0), rotInDeg=0.0, magnification=1.0):
        super().__init__(dimx, dimy, pixel_pitch, target_device_idx=target_device_idx, precision=precision)
        self.height = height
        self.shiftXYinPixel = cpuArray(shiftXYinPixel).astype(self.dtype)
        self.rotInDeg = rotInDeg
        self.magnification = magnification

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['HEIGHT'] = self._height
        super().save(filename, hdr)

    def read(self, filename, hdr=None, exten=0):
        super().read(filename, hdr, exten)

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        dimx = int(hdr['DIMX'])
        dimy = int(hdr['DIMY'])
        height = float(hdr['HEIGHT'])
        pitch = float(hdr['PIXPITCH'])

        layer = Layer(dimx, dimy, pitch, height)
        layer.read(filename, hdr)
        return layer

