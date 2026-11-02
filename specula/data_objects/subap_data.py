import math

from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from iteras.lib import storage
from specula.lib.slopes_layout import SlopesLayout


class SubapData(BaseDataObj):
    """
    Subaperture data object.
    This class holds the information about the subapertures, i.e. the indices of the pixels
    belonging to each subaperture of a Shack-Hartmann sensor.
    """
    def __init__(self,
                 idxs,
                 display_map,
                 nx: int,
                 ny: int,
                 energy_th: float=0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Initialize a :class:`~specula.data_objects.subap_data.SubapData` object.
        
        Parameters
        ----------
        
        idxs : [1]
            np.array[n_subaps, n_pixels] of pixel indices in a flattened pixel array for
            each subaperture
        display_map : [1]
            np.array[n_subaps] of subaperture indices on a flattened nx * ny array,
            used for display only
        nx : int [1]
            number of subapertures in the X (horizontal) direction
        ny : int [1]
            number of subapertures in the Y (vertical) direction
        energy_th : float [1]
            energy threshold for subaperture validity (default: 0)
        target_device_idx : int [1]
            device index for computation (default: None)
        precision : int [1]
            precision for computation (default: None)
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.idxs = self.to_xp(idxs.astype(int))
        self.display_map = self.to_xp(display_map.astype(int))
        self.nx = int(nx)
        self.ny = int(ny)
        self.energy_th = float(energy_th)

    @property
    def n_subaps(self):
        return self.idxs.shape[0]

    @property
    def np_sub(self):
        return int(math.sqrt(self.idxs.shape[1]))

    def single_mask(self):
        f = self.xp.zeros(self.nx * self.ny, dtype=self.dtype)
        self.xp.put(f, self.display_map, 1)
        return f.reshape((self.nx, self.ny))

    def slopes_layout(self):
        """
        :class:`~specula.lib.slopes_layout.SlopesLayout` of the X/Y slopes
        of these subapertures: one position per subaperture on the
        (nx, ny) grid of :meth:`single_mask`, at :attr:`display_map`.
        """
        return SlopesLayout((self.nx, self.ny), self.display_map, SlopesLayout.XY)

    def subap_idx(self, n):
        """Returns the indices of subaperture `n`."""
        return self.idxs[n, :]

    def display_map_idx(self, n):
        """Returns the position of subaperture `n`."""
        return self.display_map[n]

    def save(self, filename, overwrite=False):
        """Saves the :class:`~specula.data_objects.subap_data.SubapData` to a file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['ENRGYTH'] = self.energy_th
        hdr['NP_SUB'] = self.np_sub
        hdr['NX'] = self.nx
        hdr['NY'] = self.ny
        storage.write_object(filename, hdr,
                             {'idxs': self.idxs.T,  # Transposed for IDL-saved data compatibility
                              'display_map': self.display_map},
                             layout='placeholder', overwrite=overwrite)

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the :class:`~specula.data_objects.subap_data.SubapData` from a file."""
        with storage.read_object(filename) as f:
            hdr = f.header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            energy_th = hdr.get('ENRGYTH')
            nx = hdr.get('NX')
            ny = hdr.get('NY')
            idxs = f.read('idxs', 1).T.copy()     # Transposed for IDL-saved compatibility
            display_map = f.read('display_map', 2)
        return SubapData(idxs=idxs, display_map=display_map, nx=nx, ny=ny, energy_th=energy_th,
                         target_device_idx=target_device_idx)
