import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula import cpuArray

class PupData(BaseDataObj):
    '''
    PupData includes an ind_pup array with the pixel indexes of each pupil, of shape [index, pupil].

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
        if ind_pup is None:
            ind_pup = np.empty((0, 4))
        if radius is None:
            radius = np.zeros(4)
        if cx is None:
            cx = np.zeros(4)
        if cy is None:
            cy = np.zeros(4)
        if framesize is None:
            framesize = np.zeros(2)

        self.ind_pup = self.to_xp(ind_pup).astype(int)
        self.radius = cpuArray(radius, dtype=self.dtype)
        self.cx = cpuArray(cx, dtype=self.dtype)
        self.cy = cpuArray(cy, dtype=self.dtype)
        self.framesize = cpuArray(framesize, dtype=int)
        self.slopes_from_intensity = False

    def get_value(self):
        '''Get the pixel values as a numpy/cupy array'''
        return self.ind_pup
    
    def set_value(self, v):
        '''Set new ind_pup values.
        Arrays are not reallocated.
        '''
        assert v.shape == self.ind_pup.shape, \
            f"Error: input array shape {v.shape} does not match ind_pup shape {self.ind_pup.shape}"

        self.ind_pup[:] = self.to_xp(v)

    @property
    def n_subap(self):
        return self.ind_pup.shape[0]

    def pupil_idx(self, n):
        return self.ind_pup[:, n]

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        tmp[:, 2], tmp[:, 3] = indpup[:, 3], indpup[:, 2]
        return tmp

    def set_slopes_from_intensity(self, value: bool = True):
        self.slopes_from_intensity = value

    @property
    def display_map(self):
        if self.slopes_from_intensity:
            # Returns the indices of the pupils in the order A, B, C, D
            # where A, B, C, D are the first, second, third and fourth
            # pupils respectively. This is the order expected by PyrSlopec,
            # and it is the correct order for slopes_from_intensity.
            return self.xp.concatenate([
                self.pupil_idx(0)[self.pupil_idx(0) >= 0],  # A
                self.pupil_idx(1)[self.pupil_idx(1) >= 0],  # B  
                self.pupil_idx(2)[self.pupil_idx(2) >= 0],  # C
                self.pupil_idx(3)[self.pupil_idx(3) >= 0]   # D
            ])
        else:
            mask = self.single_mask()
            return self.xp.ravel_multi_index(self.xp.where(mask), mask.shape)

    def single_mask(self):
        f = self.xp.zeros(self.framesize[0]*self.framesize[1], dtype=self.dtype)
        self.xp.put(f, self.pupil_idx(0), 1)
        f2d = f.reshape(self.framesize)
        return f2d[:self.framesize[0]//2, self.framesize[1]//2:]

    def complete_mask(self):
        f = self.xp.zeros(self.framesize, dtype=self.dtype)
        for i in range(4):
            self.xp.put(f, self.pupil_idx(i), 1)
        return f

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['FSIZEX'] = self.framesize[0]
        hdr['FSIZEY'] = self.framesize[1]
        return hdr

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)
        fits.append(filename, cpuArray(self.ind_pup))
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
            # Workaround for ANDES pupil files missing the last HDU
            if len(hdul) >= 5:
                cy = hdul[4].data
            else:
                cy = None

        return PupData(ind_pup=ind_pup, radius=radius, cx=cx, cy=cy, framesize=framesize,
                target_device_idx=target_device_idx)

    @staticmethod
    def from_header(filename, target_device_idx=None):
        raise NotImplementedError
