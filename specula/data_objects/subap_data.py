import math

import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class SubapData(BaseDataObj):
    def __init__(self,
                 idxs,
                 display_map,
                 nx: int,
                 ny: int,
                 energy_th: float=0,
                 target_device_idx: int=None,
                 precision: int=None):
        '''
        Initialize a :class:`~specula.data_objects.subap_data.SubapData` object.
        
        idxs: np.array[n_subaps, n_pixels] of pixel indices in a flattened pixel array for each subaperture
        display_map: np.array[n_subaps] of subaperture indices on a flattened nx * ny array, used for display only
        nx: number of subapertures in the X (horizontal) direction
        ny: number of subapertures in the Y (vertical) direction
        '''
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
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)
        fits.append(filename, cpuArray(self.idxs.T))  # Transposed for IDL-saved data compatibility
        fits.append(filename, cpuArray(self.display_map))

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restores the :class:`~specula.data_objects.subap_data.SubapData` from a file."""
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            version = hdr.get('VERSION')
            if version != 1:
                raise ValueError(f"Unknown version {version} in file {filename}")
            energy_th = hdr.get('ENRGYTH')
            nx = hdr.get('NX')
            ny = hdr.get('NY')
            idxs = hdul[1].data.T     # Transposed for IDL-saved compatibility
            display_map = hdul[2].data
        return SubapData(idxs=idxs, display_map=display_map, nx=nx, ny=ny, energy_th=energy_th,
                         target_device_idx=target_device_idx)
