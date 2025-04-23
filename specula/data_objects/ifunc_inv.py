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
        
        self.ifunc_inv = self.xp.array(ifunc_inv)
        self.mask_inf_func = self.xp.array(mask)
        self.idx_inf_func = self.xp.where(self.mask_inf_func)

    def save(self, filename, hdr=None):
        hdr = hdr if hdr is not None else fits.Header()
        hdr['VERSION'] = 1

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.ifunc_inv.T), name='INFLUENCE_FUNCTION_INV'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.mask_inf_func), name='MASK_INF_FUNC'))
        hdul.writeto(filename, overwrite=True)

    def restore(filename, target_device_idx=None, exten=1):
        with fits.open(filename) as hdul:
            ifunc_inv = hdul[exten].data.T
            mask = hdul[exten+1].data
        return IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)