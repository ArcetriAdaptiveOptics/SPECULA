from astropy.io import fits
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField

class Layer(ElectricField):
    '''
    A Layer is an ElectricField with some more features: a mandatory height,
    and optional X/Y shifts, rotation and magnification
    '''

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
        self.shiftXYinPixel = self.to_xp(shiftXYinPixel).astype(self.dtype)
        self.rotInDeg = rotInDeg
        self.magnification = magnification

    # get_value and set_value are inherited from ElectricField

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Layer'
        hdr['DIMX'] = self.field[0].shape[0]
        hdr['DIMY'] = self.field[0].shape[1]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['HEIGHT'] = self.height
        hdr['SHIFTX'] = float(self.shiftXYinPixel[0])
        hdr['SHIFTY'] = float(self.shiftXYinPixel[1])
        hdr['ROTATION'] = self.rotInDeg
        hdr['MAGNIFIC'] = self.magnification
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.field[0]), name='AMPLITUDE'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.field[1]), name='PHASE'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = int(hdr['DIMX'])
        dimy = int(hdr['DIMY'])
        pitch = float(hdr['PIXPITCH'])
        height = float(hdr['HEIGHT'])
        shiftX = float(hdr['SHIFTX'])
        shiftY = float(hdr['SHIFTY'])
        rotInDeg = float(hdr['ROTATION'])
        magnification = float(hdr['MAGNIFICATION'])
        layer = Layer(dimx, dimy, pitch, height, (shiftX, shiftY), rotInDeg, magnification, target_device_idx=target_device_idx)
        return layer

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Layer':
            raise ValueError(f"Error: file {filename} does not contain a Layer object")
        layer = Layer.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            layer.field[0] = layer.to_xp(hdul[1].data.copy())
            layer.field[1] = layer.to_xp(hdul[2].data.copy())
        return layer

    # array_for_display is inherited from ElectricField