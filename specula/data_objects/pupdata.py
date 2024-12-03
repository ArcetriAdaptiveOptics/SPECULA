from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class PupData(BaseDataObj):
    '''
    TODO change to have the pupil index in the second index
    (for compatibility with existing PASSATA data)
    '''
    def __init__(self, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.radius = self.xp.zeros(4, dtype=self.dtype)
        self.cx = self.xp.zeros(4, dtype=self.dtype)
        self.cy = self.xp.zeros(4, dtype=self.dtype)
        self.ind_pup = self.xp.empty((0, 4), dtype=int)
        self.framesize = self.xp.zeros(2, dtype=int)
        
    @property
    def n_subap(self):
        return self.ind_pup.shape[1] // 4

    def zcorrection(self, indpup):
        tmp = indpup.copy()
        tmp[:, 2], tmp[:, 3] = indpup[:, 3], indpup[:, 2]
        return tmp

    def single_mask(self):
        f = self.xp.zeros(self.framesize, dtype=self.dtype)
        f.flat[self.ind_pup[:, 0]] = 1
        return f[:self.framesize[0]//2, self.framesize[1]//2:]

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

        super().save(filename, hdr)

        fits.append(filename, self.ind_pup)
        fits.append(filename, self.radius)
        fits.append(filename, self.cx)
        fits.append(filename, self.cy)

    def read(self, filename, hdr=None, exten=1):
        #hdr, exten = super().read(filename)

        self.ind_pup = self.xp.array(fits.getdata(filename, ext=exten))
        self.radius = self.xp.array(fits.getdata(filename, ext=exten + 1))
        self.cx = self.xp.array(fits.getdata(filename, ext=exten + 2))
        self.cy = self.xp.array(fits.getdata(filename, ext=exten + 3))
        exten += 4

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version > 2:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        p = PupData(target_device_idx=target_device_idx)
        if version >= 2:
            p.framesize = [int(hdr['FSIZEX']), int(hdr['FSIZEY'])]

        p.read(filename, hdr)
        return p
