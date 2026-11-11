import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula import cpuArray
from iteras.lib import storage
from specula.lib.slopes_layout import SlopesLayout

class PupData(BaseDataObj):
    """
    Pupil data object.
    This class holds the information about the pupils of a Pyramid WFS (or a Zernike WFS).
    PupData includes an ind_pup array with the pixel indexes of each pupil, of shape [index, pupil].
    """
    def __init__(self,
                 ind_pup=None,
                 radius=None,
                 cx=None,
                 cy=None,
                 framesize=None,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Notes
        -----
        TODO change by passing all the initializing arguments as __init__ parameters,
        to avoid the later initialization (see test/test_slopec.py for an example),
        where things can be forgotten easily.
        """
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
        if len(np.shape(self.ind_pup)) == 1:
            self.ind_pup = self.xp.reshape(self.ind_pup, [len(self.ind_pup),1])
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
    
    @property
    def n_pupils(self):
        return self.ind_pup.shape[1]

    def pupil_idx(self, n):
        return self.ind_pup[:, n]

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        if tmp.shape[1] == 4:
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
            #     self.pupil_idx(0)[self.pupil_idx(0) >= 0],  # A
            #     self.pupil_idx(1)[self.pupil_idx(1) >= 0],  # B
            #     self.pupil_idx(2)[self.pupil_idx(2) >= 0],  # C
            #     self.pupil_idx(3)[self.pupil_idx(3) >= 0]   # D
            return self.xp.concatenate([self.pupil_idx(i)[self.pupil_idx(i) >= 0] for i in range(self.n_pupils)])
        else:
            return self.local_display_map()

    def local_display_map(self):
        '''
        Flat index map into a single-subaperture-sized crop (the shape
        returned by :meth:`single_mask`), independent of the
        ``slopes_from_intensity`` setting.

        Since every pupil shares the same local pattern (same radius,
        only translated to a different quadrant), this same index map can
        be used to remap any subaperture-length vector - not only slopes,
        but also per-pupil pixel intensities (e.g. the A, B, C, D vectors
        computed by PyrSlopec) - into a 2d image of shape ``single_mask().shape``.
        '''
        mask = self.single_mask()
        return self.xp.ravel_multi_index(self.xp.where(mask), mask.shape)

    def single_mask(self):
        f = self.xp.zeros(self.framesize[0]*self.framesize[1], dtype=self.dtype)
        self.xp.put(f, self.pupil_idx(0), 1)
        f2d = f.reshape(self.framesize)
        return f2d[:self.framesize[0]//2, self.framesize[1]//2:]

    def slopes_layout(self, from_intensity=None):
        """
        :class:`~specula.lib.slopes_layout.SlopesLayout` of the slopes
        computed from these pupils.

        Parameters
        ----------
        from_intensity : bool, optional
            If True, one value per valid pixel of each pupil, in pupil order,
            displayed on the whole frame (shape :attr:`framesize`, as
            :meth:`complete_mask`). If False, X/Y slopes displayed on a
            single-pupil crop (shape of :meth:`single_mask`, indices of
            :meth:`local_display_map`). Default: the
            :attr:`slopes_from_intensity` setting of this object.
        """
        if from_intensity is None:
            from_intensity = self.slopes_from_intensity
        if from_intensity:
            idx = self.xp.concatenate([self.pupil_idx(i)[self.pupil_idx(i) >= 0]
                                       for i in range(self.n_pupils)])
            return SlopesLayout(tuple(self.framesize), idx, SlopesLayout.SCALAR)
        return SlopesLayout(self.single_mask().shape, self.local_display_map(), SlopesLayout.XY)

    def complete_mask(self):
        f = self.xp.zeros(self.framesize, dtype=self.dtype)
        for i in range(self.n_pupils):
            self.xp.put(f, self.pupil_idx(i), 1)
        return f

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 2
        hdr['FSIZEX'] = self.framesize[0]
        hdr['FSIZEY'] = self.framesize[1]
        return hdr

    def save(self, filename, overwrite=False):
        storage.write_object(filename, self.get_fits_header(),
                             {'ind_pup': self.ind_pup, 'radius': self.radius,
                              'cx': self.cx, 'cy': self.cy},
                             layout='placeholder', overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the pupil data from a file."""
        with storage.read_object(filename) as f:
            hdr = f.header
            version = hdr.get('VERSION')
            if version is None or version < 2:
                raise ValueError(f"Unsupported version {version} in file {filename}. Expected version >= 2")
            if version > 2:
                raise ValueError(f"Unknown version {version} in file {filename}")
            framesize = [int(hdr.get('FSIZEX')), int(hdr.get('FSIZEY'))]
            ind_pup = f.read('ind_pup', 1)
            radius = f.read('radius', 2)
            cx = f.read('cx', 3)
            # Workaround for ANDES pupil files missing the last HDU
            cy = f.read('cy', 4) if f.has('cy', 4) else None
        return PupData(ind_pup=ind_pup, radius=radius, cx=cx, cy=cy, framesize=framesize,
                target_device_idx=target_device_idx)

    @staticmethod
    def from_header(filename, target_device_idx=None):
        raise NotImplementedError
