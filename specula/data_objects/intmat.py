
import numpy as np

from specula.lib.modal_base_generator import compute_ifs_covmat
from specula.lib.mmse_reconstructor import compute_mmse_reconstructor
from iteras.data_objects.matrix import Matrix, matrix_array_name
from specula.data_objects.recmat import Recmat
from specula.data_objects.ifunc import IFunc
from iteras.lib import storage


class _ColsView:
    """
    Allows numpy-like indexing for columns.
    
    This class is initialized with a reference to the main intmat object,
    and not the intmat array directly, because some Intmat methods
    re-allocate the array, making all previous references invalid.
    """
    def __init__(self, intmat_obj): self.intmat_obj = intmat_obj
    def __getitem__(self, key): return self.intmat_obj.intmat[:, key]
    def __setitem__(self, key, value): self.intmat_obj.intmat[:, key] = self.intmat_obj.to_xp(value)

class _RowsView:
    """
    Allows numpy-like indexing for rows

    This class is initialized with a reference to the main intmat object,
    and not the intmat array directly, because some Intmat methods
    re-allocate the array, making all previous references invalid.
    """
    def __init__(self, intmat_obj): self.intmat_obj = intmat_obj
    def __getitem__(self, key): return self.intmat_obj.intmat[key, :]
    def __setitem__(self, key, value): self.intmat_obj.intmat[key, :] = self.intmat_obj.to_xp(value)

class Intmat(Matrix):
    """
    Interaction matrix data object.
    This class holds the interaction matrix (intmat) and related metadata for wavefront reconstruction.

    It is a :class:`~iteras.data_objects.matrix.Matrix`, whose matrix is also
    available as the ``intmat`` attribute.
    """
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
        """
        Note
        ----
        
        axes are [slopes, modes]

        Members .modes and .slopes allow numpy-like access, for example:

        intmat_obj.modes[3:5] += 1
        """
        if intmat is None:
            if nmodes is None or nslopes is None:
                raise ValueError('nmode sand nslopes must set if intmat is not passed')
            intmat = np.zeros((nslopes, nmodes))
        super().__init__(intmat, target_device_idx=target_device_idx, precision=precision)
        self.slope_mm = slope_mm
        self.slope_rms = slope_rms
        self.pupdata_tag = pupdata_tag
        self.subapdata_tag = subapdata_tag
        self.norm_factor = norm_factor

    @property
    def intmat(self):
        '''The interaction matrix, same as ``matrix``'''
        return self.matrix

    @intmat.setter
    def intmat(self, value):
        self.matrix = value

    # Views created on demand: stored in an attribute, they would make a reference
    # cycle, and the intmat array would be freed only by the garbage collector.
    @property
    def modes(self):
        return _ColsView(self)

    @property
    def slopes(self):
        return _RowsView(self)

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
        return self.ncols

    @property
    def nslopes(self):
        return self.nrows

    def get_fits_header(self):
        hdr = super().get_fits_header()
        hdr['PUP_TAG'] = self.pupdata_tag
        hdr['SA_TAG'] = self.subapdata_tag
        hdr['NORMFACT'] = self.norm_factor
        return hdr

    def _arrays_to_save(self):
        return {'matrix': self.matrix, 'slopemm': self.slope_mm, 'sloperms': self.slope_rms}

    def save(self, filename, overwrite=True):
        # FITS layout as before: the matrix in the first extension, after an empty primary HDU
        storage.write_object(filename, self.get_fits_header(), self._arrays_to_save(),
                             overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
            norm_factor = float(hdr.get('NORMFACT', 0.0))
            pupdata_tag = hdr.get('PUP_TAG', '')
            subapdata_tag = hdr.get('SA_TAG', '')
            # Optional arrays, read only if both are present
            if f.has('slopemm', 2) and f.has('sloperms', 3):
                slope_mm = f.read('slopemm', 2)
                slope_rms = f.read('sloperms', 3)
            else:
                slope_mm = slope_rms = None
            intmat = f.load(matrix_array_name(f), 1, target_device_idx=target_device_idx)
        return Intmat(intmat, slope_mm=slope_mm, slope_rms=slope_rms, pupdata_tag=pupdata_tag,
                      subapdata_tag=subapdata_tag, norm_factor=norm_factor,
                      target_device_idx=target_device_idx)

    def generate_rec(self, nmodes=None, cut_modes=0):
        if nmodes is not None:
            intmat = self.modes[:nmodes]
        else:
            intmat = self.intmat
        recmat = self.pseudo_invert(self.to_xp(intmat), n_modes_to_drop=cut_modes)
        rec = Recmat(recmat, target_device_idx=self.target_device_idx)
        rec.im_tag = self.norm_factor  # TODO wrong
        return rec

    def generate_rec_mmse(self, r0: float, L0: float, diameter: float, modal_base: IFunc,
                          c_noise: float | np.ndarray | list, nmodes: int = None) -> Recmat:
        if nmodes is not None:
            intmat = self.modes[:nmodes]
        else:
            intmat = self.intmat
        # atmosphere covariance matrix
        c_atm = compute_ifs_covmat(
            modal_base.mask_inf_func, diameter, modal_base.influence_function, r0, L0,
            oversampling=2, xp=self.xp, dtype=self.dtype
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
                                            c_inverse=False)
        rec = Recmat(recmat, target_device_idx=self.target_device_idx)
        return rec

    def pseudo_invert(self, matrix, n_modes_to_drop=0):
        """
        Pseudo-inverse of *matrix*, dropping the *n_modes_to_drop* smallest singular values.
        """
        if n_modes_to_drop == 0:
            return self.xp.linalg.pinv(matrix)
        u, s, vt = self.xp.linalg.svd(matrix, full_matrices=False)
        if not 0 < n_modes_to_drop < len(s):
            raise ValueError(f'pseudo_invert: n_modes_to_drop must be between 0 and {len(s) - 1}')
        n = len(s) - n_modes_to_drop
        return (vt[:n].T / s[:n]) @ u[:, :n].T

