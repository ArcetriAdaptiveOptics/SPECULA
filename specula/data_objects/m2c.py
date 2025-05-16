
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class M2C(BaseDataObj):
    def __init__(self,
                 m2c,
                 nmodes: int=None,
                 norm_factor: float=None,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.m2c = self.xp.array(m2c, dtype=self.dtype)
        if nmodes is not None:
            self.set_nmodes(nmodes)
        self.norm_factor = norm_factor

    @property
    def nmodes(self):
        return self.m2c.shape[1]

    def set_nmodes(self, nmodes):
        self.m2c = self.m2c[:, :nmodes]

    def save(self, filename):
        """Saves the M2C to a file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, cpuArray(self.m2c.T))

    @classmethod
    def restore(cls, filename, target_device_idx=None):
        """Restores the M2C from a file."""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            m2c = hdul[1].data.T
        return M2C(m2c=m2c, target_device_idx=target_device_idx)
