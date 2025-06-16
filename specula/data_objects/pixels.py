from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj


class Pixels(BaseDataObj):
    '''Pixels'''

    def __init__(self, 
                 dimx: int,
                 dimy: int,
                 bits: int=16,
                 signed: int=0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if bits > 64:
            raise ValueError("Cannot create pixel object with more than 64 bits per pixel")

        self.signed = signed
        self.type = self._get_type(bits, signed)
        self.pixels = self.xp.zeros((dimx, dimy), dtype=self.dtype)
        self.bpp = bits
        self.bytespp = (bits + 7) // 8  # bits self.xp.arounded to the next multiple of 8

    def _get_type(self, bits, signed):
        type_matrix = [
            [self.xp.uint8, self.xp.int8],
            [self.xp.uint16, self.xp.int16],
            [self.xp.uint32, self.xp.int32],
            [self.xp.uint32, self.xp.int32],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64],
            [self.xp.uint64, self.xp.int64]
        ]
        return type_matrix[(bits - 1) // 8][signed]

    def set_value(self, v):
        self.pixels[:] = v

    @property
    def size(self):
        return self.pixels.shape

    def multiply(self, factor):
        self.pixels *= factor

    def set_size(self, size):
        self.pixels = self.xp.zeros(size, dtype=self.dtype)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Pixels'
        hdr['TYPE'] = str(self.type)
        hdr['BPP'] = self.bpp
        hdr['BYTESPP'] = self.bytespp
        hdr['SIGNED'] = self.signed
        hdr['DIMX'] = self.size[0]
        hdr['DIMY'] = self.size[1]
        return hdr

    def save(self, filename):
        hdr = self.get_fits_header()            
        fits.writeto(filename, cpuArray(self.pixels), hdr, overwrite=True)

    def read(self, filename):
        super().read(filename)
        self.pixels = fits.getdata(filename)

    @staticmethod
    def from_header(hdr):    
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        bits = hdr['BPP']
        signed = hdr['SIGNED']

        pixels = Pixels(dimx, dimy, bits=bits, signed=signed)
        return pixels

    @staticmethod
    def restore(filename):
        hdr = fits.getheader(filename)
        pixels = Pixels.from_header(hdr)
        pixels.read(filename)
        return pixels

    def array_for_display(self):
        return self.pixels
