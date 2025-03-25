
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class Recmat(BaseDataObj):
    def __init__(self,
                 recmat,
                 modes2recLayer=None,
                 norm_factor: float = 0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.recmat = self.xp.array(recmat)
        self.norm_factor = norm_factor
        self.proj_list = []
        self.modes2recLayer = None
#        self.set_modes2recLayer(modes2recLayer) # TODO

    def set_modes2recLayer(self, modes2recLayer):
        if modes2recLayer is not None:
            modes2recLayer = self.xp.array(modes2recLayer)
            n = modes2recLayer.shape
            for i in range(n[0]):
                idx = self.xp.where(modes2recLayer[i, :] > 0)[0]
                proj = self.xp.zeros((n[1], len(idx)), dtype=self.dtype)
                proj[idx, :] = self.xp.identity(len(idx))
                self.proj_list.append(proj)
            self.modes2recLayer = modes2recLayer
            
    def reduce_size(self, nModesToBeDiscarded):
        nmodes = self.recmat.shape[1]
        if nModesToBeDiscarded >= nmodes:
            raise ValueError(f"nModesToBeDiscarded should be less than nmodes (<{nmodes})")
        self.recmat = self.recmat[:, :nmodes - nModesToBeDiscarded]

    def save(self, filename, hdr=None):
        
        if not filename.endswith('.fits'):
            filename += '.fits'

        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NORMFACT'] = self.norm_factor

        fits.writeto(filename, np.zeros(2), hdr)
        fits.append(filename, cpuArray(self.recmat.T))
        if self.modes2recLayer is not None:
            fits.append(filename, cpuArray(self.modes2recLayer))

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        norm_factor = float(hdr['NORMFACT'])
        recmat = fits.getdata(filename, ext=1)
        if len(fits.open(filename)) >= 3:
            mode2reLayer = fits.getdata(filename, ext=2)
        else:
            mode2reLayer = None
        return Recmat(recmat, mode2reLayer, norm_factor, target_device_idx=target_device_idx)


