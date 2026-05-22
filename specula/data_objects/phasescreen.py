
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj

class Phasescreen(BaseDataObj):
    """
    Phasescreen field data object.
    """
    def __init__(self, 
                 dimx: int, 
                 dimy: int,
                 L0: float,
                 seed: int,
                 target_device_idx: int=None, 
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.phasescreen.Phasescreen` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.L0 = L0
        self.seed = seed
        self.phasescreen = self.xp.zeros((dimy, dimx), dtype=self.dtype)

    def get_value(self):
        '''
        Get the phasescreen as a numpy/cupy array
        '''
        return self.phasescreen

    def set_value(self, v):
        '''
        Set a new phasescreen.
        Arrays are not reallocated
        '''
        assert v.shape == self.phasescreen.shape, \
            f"Error: input array shape {v.shape} does not match phasescreen field shape {self.phasescreen.shape}"
        self.phasescreen[:]= self.to_xp(v)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Phasescreen'
        hdr['DIMX'] = self.phasescreen.shape[1]
        hdr['DIMY'] = self.phasescreen.shape[0]
        hdr['L0'] = self.L0
        hdr['SEED'] = self.seed
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr, data=cpuArray(self.phasescreen))
        hdul = fits.HDUList([hdu])
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        L0 = hdr['L0']
        seed = hdr['SEED']
        phasescreen = Phasescreen(dimx, dimy, L0=L0, seed=seed, target_device_idx=target_device_idx)
        return phasescreen
    
    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Phasescreen':
            raise ValueError(f"Error: file {filename} does not contain a Phasescreen object")
        phasescreen = Phasescreen.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            phasescreen.phasescreen[:] = phasescreen.to_xp(hdul[0].data)  # pylint: disable=no-member
        return phasescreen

    def array_for_display(self):
        return self.phasescreen
