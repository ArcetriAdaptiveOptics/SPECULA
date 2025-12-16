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
        """
        Initialize an :class:`~specula.data_objects.ifunc_inv.IFuncInv` object.
        """
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
        hdul.close()  # Force close for Windows

    @staticmethod
    def restore(filename, target_device_idx=None, exten=1):
        with fits.open(filename) as hdul:
            ifunc_inv = hdul[exten].data.T
            mask = hdul[exten+1].data
        return IFuncInv(ifunc_inv, mask, target_device_idx=target_device_idx)

    def get_value(self):
        return self.ifunc_inv
    
    def set_value(self, v):
        '''Set a new influence function.
        Arrays are not reallocated.'''
        assert v.shape == self.ifunc_inv.shape, \
            f"Error: input array shape {v.shape} does not match " \
            f"inverse influence function shape {self.ifunc_inv.shape}"

        self.ifunc_inv[:] = self.to_xp(v)

    def cut(self, start_mode=None, nmodes=None, idx_modes=None):
        """
        Cut the inverse influence function to a subset of modes.
        
        Parameters
        ----------
        start_mode : int, optional
            Starting mode index (default: 0)
        nmodes : int, optional
            Number of modes to keep (default: all remaining modes from start_mode)
        idx_modes : array-like, optional
            Explicit list of mode indices to keep. If provided, start_mode and nmodes are ignored.
        
        Notes
        -----
        The inverse influence function has shape (npixels, nmodes), so we cut along axis 1 (columns).
        This is the opposite of IFunc which has shape (nmodes, npixels) and cuts along axis 0 (rows).
        """
        if idx_modes is not None:
            if start_mode is not None:
                start_mode = None
                print('ifunc_inv.cut: start_mode cannot be set together with idx_modes. Setting to None start_mode.')
            if nmodes is not None:
                nmodes = None
                print('ifunc_inv.cut: nmodes cannot be set together with idx_modes. Setting to None nmodes.')

        nrows, ncols = self.ifunc_inv.shape

        if start_mode is None:
            start_mode = 0
        if nmodes is None:
            nmodes = ncols

        if idx_modes is not None:
            self.ifunc_inv = self.ifunc_inv[:, idx_modes]
        else:
            self.ifunc_inv = self.ifunc_inv[:, start_mode:nmodes]

    @staticmethod
    def from_header(hdr):
        raise NotImplementedError
