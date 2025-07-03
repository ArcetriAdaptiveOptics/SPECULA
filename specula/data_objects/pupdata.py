import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class PupData(BaseDataObj):
    '''
    TODO change to have the pupil index in the second index
    (for compatibility with existing PASSATA data)

    TODO change by passing all the initializing arguments as __init__ parameters,
    to avoid the later initialization (see test/test_slopec.py for an example),
    where things can be forgotten easily
    '''
    def __init__(self,
                 ind_pup=None,
                 radius=None,
                 cx=None,
                 cy=None,
                 framesize=None,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Initialize with provided data or defaults
        if ind_pup is not None:
            self.ind_pup = self.to_xp(ind_pup).astype(int)
        else:
            self.ind_pup = self.xp.empty((0, 4), dtype=int)

        if radius is not None:
            self.radius = self.to_xp(radius).astype(self.dtype)
        else:
            self.radius = self.xp.zeros(4, dtype=self.dtype)

        if cx is not None:
            self.cx = self.to_xp(cx).astype(self.dtype)
        else:
            self.cx = self.xp.zeros(4, dtype=self.dtype)

        if cy is not None:
            self.cy = self.to_xp(cy).astype(self.dtype)
        else:
            self.cy = self.xp.zeros(4, dtype=self.dtype)

        if framesize is not None:
            self.framesize = np.array(framesize, dtype=int)
        else:
            self.framesize = np.zeros(2, dtype=int)

    @property
    def n_subap(self):
        return self.ind_pup.shape[1] // 4

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        tmp[:, 2], tmp[:, 3] = indpup[:, 3], indpup[:, 2]
        return tmp

    @property
    def display_map(self):
        mask = self.single_mask()
        return self.xp.ravel_multi_index(self.xp.where(mask), mask.shape)

    def single_mask(self):
        f = self.xp.zeros(self.framesize[0]*self.framesize[1], dtype=self.dtype)
        self.xp.put(f, self.ind_pup[:, 0], 1)
        f2d = f.reshape(self.framesize)
        return f2d[:self.framesize[0]//2, self.framesize[1]//2:]

    def complete_mask(self):
        f = self.xp.zeros(self.framesize, dtype=self.dtype)
        for i in range(4):
            f.flat[self.ind_pup[:, i]] = 1
        return f

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['FSIZEX'] = self.framesize[0]
        hdr['FSIZEY'] = self.framesize[1]

        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, self.ind_pup.T)
        fits.append(filename, self.radius)
        fits.append(filename, self.cx)
        fits.append(filename, self.cy)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the pupil data from a file."""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version is None or version < 2:
                raise ValueError(f"Unsupported version {version} in file {filename}. Expected version >= 2")
            if version > 2:
                raise ValueError(f"Unknown version {version} in file {filename}")

            framesize = [int(hdr.get('FSIZEX')), int(hdr.get('FSIZEY'))]
            ind_pup = hdul[1].data
            radius = hdul[2].data
            cx = hdul[3].data
            cy = hdul[4].data

        return PupData(ind_pup=ind_pup, radius=radius, cx=cx, cy=cy, framesize=framesize,
                target_device_idx=target_device_idx)
