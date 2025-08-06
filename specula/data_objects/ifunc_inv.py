from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from astropy.io import fits


class IFuncInv(BaseDataObj):
    def __init__(self,
                 ifunc_inv,
                 mask,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(precision=precision, target_device_idx=target_device_idx)
        self._doZeroPad = False
        
        self.ifunc_inv = self.to_xp(ifunc_inv)
        self.mask_inf_func = self.to_xp(mask)
        self.idx_inf_func = self.xp.where(self.mask_inf_func)

    @property
    def size(self):
        return self.ifunc_inv.shape

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        return hdr

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.ifunc_inv.T), name='INFLUENCE_FUNCTION_INV'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.mask_inf_func), name='MASK_INF_FUNC'))
        hdul.writeto(filename, overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None, exten=1):
        with fits.open(filename) as hdul:
            ifunc_inv = hdul[exten].data.T
            mask = hdul[exten+1].data
        return IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

    def get_value(self):
        return self.ifunc_inv
    
    def set_value(self, v, force_copy=False):
        '''Set a new influence function.
        Arrays are not reallocated.'''
        assert v.shape == self.ifunc_inv.shape, \
            f"Error: input array shape {v.shape} does not match inverse influence function shape {self.ifunc_inv.shape}"

        self.ifunc_inv[:] = self.to_xp(v, force_copy=force_copy)

    @staticmethod
    def from_header(hdr):
        raise NotImplementedError