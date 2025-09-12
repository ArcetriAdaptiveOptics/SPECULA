
import numpy as np
from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj
from specula.data_objects.recmat import Recmat


class _ColsView:
    '''
    Allows numpy-like indexing for columns.
    
    This class is initialized with a reference to the main intmat object,
    and not the intmat array directly, because some Intmat methods
    re-allocate the array, making all previous references invalid.
    '''
    def __init__(self, intmat_obj): self.intmat_obj = intmat_obj
    def __getitem__(self, key): return self.intmat_obj.intmat[:, key]
    def __setitem__(self, key, value): self.intmat_obj.intmat[:, key] = self.intmat_obj.to_xp(value)
 
class _RowsView:
    '''
    Allows numpy-like indexing for rows

    This class is initialized with a reference to the main intmat object,
    and not the intmat array directly, because some Intmat methods
    re-allocate the array, making all previous references invalid.
    '''
    def __init__(self, intmat_obj): self.intmat_obj = intmat_obj
    def __getitem__(self, key): return self.intmat_obj.intmat[key, :]
    def __setitem__(self, key, value): self.intmat_obj.intmat[key, :] = self.intmat_obj.to_xp(value)

class Intmat(BaseDataObj):
    '''
    Interaction matrix axes are [slopes, modes]

    Members .modes and .slopes allow numpy-like access, for example:

    intmat_obj.modes[3:5] += 1
    '''
    def __init__(self,
                 intmat = None,
                 nmodes:  int = None,
                 nslopes: int = None,
                 slope_mm: list = None,
                 slope_rms: list = None,
                 pupdata_tag: str = '',
                 subapdata_tag: str = '',
                 norm_factor: float= 0.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        if intmat is not None:
            self.intmat = self.to_xp(intmat)
        else:
            if nmodes is None or nslopes is None:
                raise ValueError('nmode sand nslopes must set if intmat is not passed')
            self.intmat = self.xp.zeros((nslopes, nmodes), dtype=self.dtype)
        self.slope_mm = slope_mm
        self.slope_rms = slope_rms
        self.pupdata_tag = pupdata_tag
        self.subapdata_tag = subapdata_tag
        self.norm_factor = norm_factor

        self.modes = _ColsView(self)
        self.slopes = _RowsView(self)

    def get_value(self):
        '''
        Get the intmat as a numpy/cupy array
        '''
        return self.intmat

    def set_value(self, v):
        '''
        Set new values for the intmat
        Arrays are not reallocated
        '''
        assert v.shape == self.intmat.shape, \
            f"Error: input array shape {v.shape} does not match intmat shape {self.intmat.shape}"
        self.intmat[:]= self.to_xp(v)

    def set_nmodes(self, new_nmodes):
        old_nmodes = self.nmodes
        if new_nmodes > old_nmodes:
            new_intmat = self.xp.zeros((self.nslopes, new_nmodes), dtype=self.dtype)
            new_intmat[:, :old_nmodes] = self.intmat[:, :old_nmodes]
        else:
            new_intmat = self.intmat[:, :new_nmodes]
        self.intmat = new_intmat

    def set_nslopes(self, new_nslopes):
        old_nslopes = self.nslopes
        if new_nslopes > old_nslopes:
            new_intmat = self.xp.zeros((new_nslopes, self.nmodes), dtype=self.dtype)
            new_intmat[:old_nslopes, :] = self.intmat[:old_nslopes, :]
        else:
            new_intmat = self.intmat[:new_nslopes, :]
        self.intmat = new_intmat

    def reduce_size(self, n_modes_to_be_discarded):
        if n_modes_to_be_discarded >= self.nmodes:
            raise ValueError(f'nModesToBeDiscarded should be less than nmodes (<{self.nmodes})')
        self.intmat = self.modes[:self.nmodes - n_modes_to_be_discarded]

    def reduce_slopes(self, n_slopes_to_be_discarded):
        if n_slopes_to_be_discarded >= self.nslopes:
            raise ValueError(f'nSlopesToBeDiscarded should be less than nslopes (<{self.nslopes})')
        self.intmat = self.slopes[:self.nslopes - n_slopes_to_be_discarded]

    def set_start_mode(self, start_mode):
        nmodes = self.intmat.shape[1]
        if start_mode >= nmodes:
            raise ValueError(f'start_mode should be less than nmodes (<{nmodes})')
        self.intmat = self.modes[start_mode:]

    @property
    def nmodes(self):
        return self.intmat.shape[1]

    @property
    def nslopes(self):
        return self.intmat.shape[0]

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PUP_TAG'] = self.pupdata_tag
        hdr['SA_TAG'] = self.subapdata_tag
        hdr['NORMFACT'] = self.norm_factor
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.intmat), name='INTMAT'))
        if self.slope_mm is not None:
            hdul.append(fits.ImageHDU(data=cpuArray(self.slope_mm), name='SLOPEMM'))
        if self.slope_rms is not None:
            hdul.append(fits.ImageHDU(data=cpuArray(self.slope_rms), name='SLOPERMS'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError
    
    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename, ext=0)
        intmat = fits.getdata(filename, ext=1)
        norm_factor = float(hdr.get('NORMFACT', 0.0))
        pupdata_tag = hdr.get('PUP_TAG', '')
        subapdata_tag = hdr.get('SA_TAG', '')
        # Reading additional fits extensions
        with fits.open(filename) as hdul:
            num_ext = len(hdul)
        if num_ext >= 4:
            slope_mm = fits.getdata(filename, ext=2)
            slope_rms = fits.getdata(filename, ext=3)
        else:
            slope_mm = slope_rms = None
        return Intmat(intmat, slope_mm, slope_rms, pupdata_tag, subapdata_tag, norm_factor, target_device_idx=target_device_idx)

    def generate_rec(self, nmodes=None, cut_modes=0, w_vec=None, interactive=False):
        if nmodes is not None:
            intmat = self.modes[:nmodes]
        else:
            intmat = self.intmat
        recmat = self.pseudo_invert(self.to_xp(intmat), n_modes_to_drop=cut_modes, w_vec=w_vec, interactive=interactive)
        rec = Recmat(recmat, target_device_idx=self.target_device_idx)
        rec.im_tag = self.norm_factor  # TODO wrong
        return rec

    def pseudo_invert(self, matrix, n_modes_to_drop=0, w_vec=None, interactive=False):
        # TODO handle n_modes_to_drop, and w_vec
        return self.xp.linalg.pinv(matrix)

    @staticmethod
    def build_from_slopes(slopes, disturbance, target_device_idx=None):
        times = list(slopes.keys())
        nslopes = len(slopes[times[0]])
        nmodes = len(disturbance[times[0]])
        intmat = np.zeros((nslopes, nmodes))
        im = Intmat(intmat, target_device_idx=target_device_idx)
        iter_per_mode = im.xp.zeros(nmodes)

        for t in times:
            amp = disturbance[t]
            mode = np.where(amp)[0][0]
            im.modes[mode] += im.to_xp(slopes[t] / amp[mode])
            iter_per_mode[mode] += 1

        for mode in range(nmodes):
            if iter_per_mode[mode] > 0:
                im.modes[mode] /= iter_per_mode[mode]

        im.slope_mm = im.xp.zeros((nmodes, 2))
        im.slope_rms = im.xp.zeros(nmodes)
        return im
