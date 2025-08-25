
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class Recmat(BaseDataObj):
    '''
    Reconstruction matrix axes are [modes, slopes]
    '''
    def __init__(self,
                 recmat,
                 modes2recLayer=None,
                 norm_factor: float=0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.recmat.Recmat` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.recmat = self.to_xp(recmat)
        self.norm_factor = norm_factor
        self.proj_list = []
        self.modes2recLayer = None
#        self.set_modes2recLayer(modes2recLayer) # TODO

    def get_value(self):
        '''
        Get the recmat as a numpy/cupy array
        '''
        return self.recmat

    def set_value(self, v):
        '''
        Set new values for the recmat
        Arrays are not reallocated
        '''
        assert v.shape == self.recmat.shape, \
            f"Error: input array shape {v.shape} does not match recmat shape {self.recmat.shape}"
        self.recmat[:]= self.to_xp(v)

    def set_modes2recLayer(self, modes2recLayer):
        if modes2recLayer is not None:
            modes2recLayer = self.to_xp(modes2recLayer)
            n = modes2recLayer.shape
            for i in range(n[0]):
                idx = self.xp.where(modes2recLayer[i, :] > 0)[0]
                proj = self.xp.zeros((n[1], len(idx)), dtype=self.dtype)
                proj[idx, :] = self.xp.identity(len(idx))
                self.proj_list.append(proj)
            self.modes2recLayer = modes2recLayer
            
    def reduce_size(self, nModesToBeDiscarded):
        if nModesToBeDiscarded >= self.nmodes:
            raise ValueError(f"nModesToBeDiscarded should be less than nmodes (<{self.nmodes})")
        self.recmat = self.recmat[:self.nmodes - nModesToBeDiscarded, :]

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NORMFACT'] = self.norm_factor
        return hdr

    @property
    def nmodes(self):
        return self.recmat.shape[0]

    def save(self, filename, overwrite=False):
        if not filename.endswith('.fits'):
            filename += '.fits'
        hdr = self.get_fits_header()
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)
        fits.append(filename, cpuArray(self.recmat))
        if self.modes2recLayer is not None:
            fits.append(filename, cpuArray(self.modes2recLayer))

    @staticmethod
    def from_header(filename, target_device_idx=None):
        raise NotImplementedError

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        norm_factor = float(hdr['NORMFACT'])
        recmat = fits.getdata(filename, ext=1)
        with fits.open(filename) as hdul:
            num_ext = len(hdul)
        if num_ext >= 3:                
            mode2reLayer = fits.getdata(filename, ext=2)
        else:
            mode2reLayer = None
        return Recmat(recmat, mode2reLayer, norm_factor, target_device_idx=target_device_idx)


