
from astropy.io import fits

from specula import cpuArray
from specula.data_objects.ifunc_inv import cut_modes
from specula.base_data_obj import BaseDataObj


class M2C(BaseDataObj):
    """
    Modal to Command (M2C) matrix data object.
    This class holds the M2C matrix, which is used to convert modal coefficients to actuator commands.
    The M2C matrix is stored as [actuators, modes].
    """
    def __init__(self,
                 m2c,
                 nmodes: int=None,
                 norm_factor: float=None,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.m2c.M2C` object.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.m2c = self.to_xp(m2c, dtype=self.dtype)
        if nmodes is not None:
            self.set_nmodes(nmodes)
        self.norm_factor = norm_factor

    def get_value(self):
        '''
        Get the m2c matrix field as a numpy/cupy array
        '''
        return self.m2c

    def set_value(self, v):
        '''
        Set new values for the m2c matrix field
        Arrays are not reallocated
        '''
        assert v.shape == self.m2c.shape, \
            f"Error: input array shape {v.shape} does not match m2c shape {self.m2c.shape}"
        self.m2c[:]= self.to_xp(v)

    @property
    def nmodes(self):
        return self.m2c.shape[1]

    def set_nmodes(self, nmodes):
        self.m2c = self.m2c[:, :nmodes]

    def cut(self, start_mode=None, nmodes=None, idx_modes=None):
        self.m2c = cut_modes(self.m2c, start_mode=start_mode, nmodes=nmodes, idx_modes=idx_modes, modes_on_first_axis=False)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        raise NotImplementedError

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        return hdr

    def save(self, filename, overwrite:bool=False):
        """Saves the M2C to a file."""
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.m2c)))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the :class:`~specula.data_objects.m2c.M2C` from a file."""
        # pylint: disable=no-member
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            m2c = hdul[1].data
        return M2C(m2c=m2c, target_device_idx=target_device_idx)
