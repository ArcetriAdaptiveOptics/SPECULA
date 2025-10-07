
import numpy as np
from astropy.io import fits
from specula import cpuArray

from specula.lib.modal_base_generator import compute_ifs_covmat
from specula.lib.mmse_reconstructor import compute_mmse_reconstructor
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
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            intmat = hdul[1].data.copy()
            norm_factor = float(hdr.get('NORMFACT', 0.0))
            pupdata_tag = hdr.get('PUP_TAG', '')
            subapdata_tag = hdr.get('SA_TAG', '')
            # Reading additional fits extensions
            if len(hdul) >= 4:
                slope_mm = hdul[2].data.copy()
                slope_rms = hdul[3].data.copy()
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

    def generate_rec_mmse(self, r0, L0, diameter, modal_base, c_noise, nmodes=None, m2c=None):
        if nmodes is not None:
            intmat = self.modes[:nmodes]
        else:
            intmat = self.intmat
        # atmosphere covariance matrix
        if m2c is not None:
            influence_function = m2c.m2c.T @ modal_base.influence_function
        else:
            influence_function = modal_base.influence_function
        c_atm = compute_ifs_covmat(
            modal_base.mask_inf_func, diameter, influence_function, r0, L0,
            oversampling=2, verbose=False, xp=self.xp, dtype=self.dtype
        )
        if c_atm.shape[0] > intmat.shape[1]:
            c_atm = c_atm[:intmat.shape[1], :intmat.shape[1]]
        # noise covariance matrix
        if isinstance(c_noise, (int, float, np.number)) \
            or (hasattr(c_noise, 'shape') and c_noise.shape == ()):
            noise_variance = [float(c_noise)]
            c_noise_mat = None
        elif isinstance(c_noise, list):
            # Handle list case
            if len(c_noise) == 1:
                noise_variance = [float(c_noise[0])]
                c_noise_mat = None
            elif len(c_noise) != intmat.shape[0]:
                raise ValueError(f'c_noise length {len(c_noise)} is not compatible with '
                                 f'intmat shape {intmat.shape}')
            else:
                noise_variance = None
                diag_elements = []
                for elem in c_noise:
                    if hasattr(elem, '__len__') and not isinstance(elem, str):
                        # array/list
                        diag_elements.extend(elem)
                    else:
                        # scalar
                        diag_elements.append(float(elem))
                # Create diagonal noise covariance matrix
                c_noise_mat = self.xp.diag(self.to_xp(diag_elements))
        elif c_noise.shape[0] == 1:
            if c_noise.ndim == 1:
                noise_variance = [float(c_noise[0])]
                c_noise_mat = None
            else:
                noise_variance = [float(c_noise[0, 0])]
                c_noise_mat = None
        elif c_noise.shape[0] != intmat.shape[0]:
            raise ValueError(f'c_noise shape {c_noise.shape} is not compatible with '
                             f'intmat shape {intmat.shape}')
        else:
            noise_variance = None
            c_noise_mat = c_noise
        # compute MMSE reconstructor
        recmat = compute_mmse_reconstructor(self.to_xp(intmat), c_atm, self.xp,
                                            self.dtype, noise_variance=noise_variance,
                                            c_noise=c_noise_mat,
                                            c_inverse=False, verbose=False)
        rec = Recmat(recmat, target_device_idx=self.target_device_idx)
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
