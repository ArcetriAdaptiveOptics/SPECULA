

import numpy as np
from astropy.io import fits


from specula.lib.make_xy import make_xy
from specula.base_data_obj import BaseDataObj


class Lenslet(BaseDataObj):
    def __init__(self,
                 n_lenses: int=1,
                 target_device_idx:int =None,
                 precision:int =None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.n_lenses = n_lenses
        self._lenses = []

        if n_lenses > 1:
            x, y = make_xy(n_lenses, 1.0, xp=self.xp)
        else:
            x = [0.0]
            y = [0.0]
        
        subap_size = 2.0 / n_lenses

        for i in range(n_lenses):
            row = []
            for j in range(n_lenses):
                row.append([x[i, j], y[i, j], subap_size])
            self._lenses.append(row)

    # There is no value to get/set
    def get_value(self):
        raise NotImplementedError

    def set_value(self, v, force_copy=True):
        raise NotImplementedError

    @property
    def dimx(self):
        return len(self._lenses)

    @property
    def dimy(self):
        return len(self._lenses[0]) if self._lenses else 0

    def get(self, x, y):
        """Returns the subaperture information at (x, y)"""
        return self._lenses[x][y]

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['N_LENSES'] = self.n_lenses
        return hdr

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f'Error: unknown version {version} in header')
        return Lenslet(hdr['N_LENSES'], target_device_idx=target_device_idx)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        return Lenslet.from_header(hdr, target_device_idx=target_device_idx)



