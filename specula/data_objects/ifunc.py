from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from specula.data_objects.ifunc_inv import IFuncInv
from astropy.io import fits

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc

from specula.lib.compute_zern_ifunc import compute_zern_ifunc

def compute_kl_ifunc(*args, **kwargs):
    raise NotImplementedError

def compute_mixed_ifunc(*args, **kwargs):
    raise NotImplementedError


class IFunc(BaseDataObj):
    '''
    Influence functions are stored as [modes, pixels]
    '''
    def __init__(self,
                 ifunc=None,
                 type_str: str=None,
                 mask=None,
                 npixels: int=None,
                 nzern: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 start_mode: int=None,
                 nmodes: int=None,
                 n_act: int=None,
                 circ_geom: bool=True,
                 angle_offset: float=0,
                 do_mech_coupling: bool=False,
                 coupling_coeffs: list=[0.31, 0.05],
                 do_slaving: bool=False,
                 slaving_thr: float=0.1,
                 idx_modes=None,
                 target_device_idx=None,
                 precision=None
                ):
        super().__init__(precision=precision, target_device_idx=target_device_idx)
        self.type_str = type_str
        self._doZeroPad = False

        if ifunc is None:
            if type_str is None:
                raise ValueError('At least one of ifunc and type must be set')
            if mask is not None:
                mask = (self.to_xp(mask) > 0).astype(self.dtype)
            if npixels is None:
                raise ValueError("If ifunc is not set, then npixels must be set!")

            type_lower = type_str.lower()
            if type_lower == 'kl':
                if nmodes is None:
                    raise ValueError('nmodes parameter is mandatory with type "kl"')
                ifunc, mask = compute_kl_ifunc(npixels, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                               xp=self.xp, dtype=self.dtype)
            elif type_lower in ['zern', 'zernike']:
                if nzern is not None:
                    raise ValueError('nzern is ignored with type "zern" or "zernike", please use nmodes instead')
                if nmodes is None:
                    raise ValueError('nmodes parameter is mandatory with type "zern" or "zernike"')
                ifunc, mask = compute_zern_ifunc(npixels, nzern=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                 xp=self.xp, dtype=self.dtype)
            elif type_lower == 'mixed':
                if nmodes is None or nzern is None:
                    raise ValueError('Both nzern and nmodes parameters are mandatory with type "mixed"')
                ifunc, mask = compute_mixed_ifunc(npixels, nzern=nzern, nmodes=nmodes, obsratio=obsratio, diaratio=diaratio, mask=mask,
                                                  xp=self.xp, dtype=self.dtype)
            elif type_lower == 'zonal':
                if n_act is None:
                    raise ValueError('nact parameter is mandatory with type "zonal"')
                ifunc, mask = compute_zonal_ifunc(npixels, n_act, circ_geom=circ_geom, angle_offset=angle_offset, do_mech_coupling=do_mech_coupling,
                                                  coupling_coeffs=coupling_coeffs, do_slaving=do_slaving, slaving_thr=slaving_thr,
                                                  obsratio=obsratio, diaratio=diaratio, mask=mask, xp=self.xp, dtype=self.dtype,
                                                  return_coordinates=False)
            else:
                raise ValueError(f'Invalid ifunc type {type_str}')

        ifunc = self.to_xp(ifunc)
        mask = self.to_xp(mask)

        self._influence_function = ifunc
        self._mask_inf_func = mask
        self._idx_inf_func = self.xp.where(self._mask_inf_func)
        self.cut(start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes)

    @property
    def influence_function(self):
        return self._influence_function

    @influence_function.setter
    def influence_function(self, ifunc):
        if self._doZeroPad:
            raise NotImplementedError("zeroPad is not implemented")
            if self._mask_inf_func is None:
                raise ValueError("if doZeroPad is set, mask_inf_func must be set before setting ifunc.")
            sIfunc = ifunc.shape

            if sIfunc[0] < sIfunc[1]:
                ifuncPad = self.xp.zeros((sIfunc[0], len(self._mask_inf_func)), dtype=ifunc.dtype)
                ifuncPad[:, self._idx_inf_func] = ifunc
            else:
                ifuncPad = self.xp.zeros((len(self._mask_inf_func), sIfunc[1]), dtype=ifunc.dtype)
                ifuncPad[self._idx_inf_func, :] = ifunc

            ifunc = ifuncPad

        self._influence_function = self.to_xp(ifunc, dtype=self.dtype)

    @property
    def mask_inf_func(self):
        return self._mask_inf_func

    @mask_inf_func.setter
    def mask_inf_func(self, mask_inf_func):
        self._mask_inf_func = self.to_xp(mask_inf_func, dtype=self.dtype)
        self._idx_inf_func = self.xp.where(self._mask_inf_func)

    @property
    def idx_inf_func(self):
        return self._idx_inf_func

    @property
    def size(self):
        return self._influence_function.shape

    @property
    def type(self):
        return self._influence_function.dtype

    def get_value(self):
        return self._influence_function
    
    def set_value(self, v):
        '''Set a new influence function.
        Arrays are not reallocated.'''
        assert v.shape == self._influence_function.shape, \
            f"Error: input array shape {v.shape} does not match influence function shape {self._influence_function.shape}"

        self._influence_function[:] = self.to_xp(v)

    @staticmethod
    def from_header(hdr):
        raise NotImplementedError

    def inverse(self):
        inv = self.xp.linalg.pinv(self._influence_function)
        return IFuncInv(inv, mask=self._mask_inf_func, precision=self.precision, target_device_idx=self.target_device_idx)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        return hdr

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self._influence_function.T), name='INFLUENCE_FUNCTION'))
        hdul.append(fits.ImageHDU(data=cpuArray(self._mask_inf_func), name='MASK_INF_FUNC'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    def cut(self, start_mode=None, nmodes=None, idx_modes=None):

        if idx_modes is not None:
            if start_mode is not None:
                start_mode = None
                print('ifunc.cut: start_mode cannot be set together with idx_modes. Setting to None start_mode.')
            if nmodes is not None:
                nmodes = None
                print('ifunc.cut: nmodes cannot be set together with idx_modes. Setting to None nmodes.')

        nrows, ncols = self.influence_function.shape

        if start_mode is None:
            start_mode = 0
        if nmodes is None:
            nmodes = nrows

        if idx_modes is not None:
            self._influence_function = self._influence_function[idx_modes, :]
        else:
            self._influence_function = self._influence_function[start_mode:nmodes, :]

    @staticmethod
    def restore(filename, target_device_idx=None, exten=1):
        with fits.open(filename) as hdul:
            ifunc = hdul[exten].data.T
            mask = hdul[exten+1].data
        return IFunc(ifunc, mask=mask, target_device_idx=target_device_idx)
