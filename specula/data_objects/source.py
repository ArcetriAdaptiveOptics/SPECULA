import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula.lib.n_phot import n_phot
from specula import ASEC2RAD

degree2rad = np.pi / 180.

class Source(BaseDataObj):
    '''source'''

    def __init__(self,
                 polar_coordinates: list, # TODO =[0.0,0.0],
                 magnitude: float,        # TODO =10.0,
                 wavelengthInNm: float,   # TODO =500.0,
                 height: float=float('inf'),
                 band: str='',
                 zeroPoint: float=0,
                 error_coord: tuple=(0., 0.),
                 verbose: bool=False,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.orig_polar_coordinates = np.array(polar_coordinates).copy()

        polar_coordinates = np.array(polar_coordinates, dtype=self.dtype) + np.array(error_coord, dtype=self.dtype)
        if any(error_coord):
            print(f'there is a desired error ({error_coord[0]},{error_coord[1]}) on source coordinates.')
            print(f'final coordinates are: {polar_coordinates[0]},{polar_coordinates[1]}')
        
        self.polar_coordinates = polar_coordinates
        self.height = height
        self.magnitude = magnitude
        self.wavelengthInNm = wavelengthInNm
        self.zeroPoint = zeroPoint
        self.band = band
        self.verbose = verbose
        self.error_coord = error_coord

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PCOORD0'] = self.orig_polar_coordinates[0]
        hdr['PCOORD1'] = self.orig_polar_coordinates[1]
        hdr['MAGNITUD'] = self.magnitude
        hdr['WAVELENG'] = self.wavelengthInNm
        hdr['HEIGHT'] = self.height
        hdr['BAND'] = self.band
        hdr['ZEROPNT'] = self.zeroPoint
        hdr['ERR_CRD0'] = self.error_coord[0]
        hdr['ERR_CRD1'] = self.error_coord[1]
        return hdr

    # There is no value to get/set
    def get_value(self):
        raise NotImplementedError

    def set_value(self, v, force_copy=True):
        raise NotImplementedError

    @property
    def polar_coordinates(self):
        return self._polar_coordinates

    @polar_coordinates.setter
    def polar_coordinates(self, value):
        self._polar_coordinates = np.array(value, dtype=self.dtype)

    @property
    def r(self):
        return self._polar_coordinates[0] * ASEC2RAD

    @property
    def r_arcsec(self):
        return self._polar_coordinates[0]

    @property
    def phi(self):
        return self._polar_coordinates[1] * degree2rad

    @property
    def phi_deg(self):
        return self._polar_coordinates[1]

    @property
    def x_coord(self):
        alpha = self._polar_coordinates[0] * ASEC2RAD
        d = self.height * np.sin(alpha)
        return np.cos(np.radians(self._polar_coordinates[1])) * d

    @property
    def y_coord(self):
        alpha = self._polar_coordinates[0] * ASEC2RAD
        d = self.height * np.sin(alpha)
        return np.sin(np.radians(self._polar_coordinates[1])) * d

    def phot_density(self):
        if self.zeroPoint > 0:
            e0 = self.zeroPoint
        else:
            e0 = None
        if self.band:
            band = self.band
        else:
            band = None

        res = n_phot(self.magnitude, band=band, lambda_=self.wavelengthInNm/1e9, width=1e-9, e0=e0)
        if self.verbose:
            print(f'source.phot_density: magnitude is {self.magnitude}, and flux (output of n_phot with width=1e-9, surf=1) is {res[0]}')
        return res[0]

    def save(self, filename, overwrite=False):
        hdr = self.get_fits_header()
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f'Error: unknown version {version} in header')
        return Source(polar_coordinates=[ hdr['PCOORD0'], hdr['PCOORD1']],
                 magnitude=hdr['MAGNITUD'],
                 wavelengthInNm=hdr['WAVELENG'],
                 height=hdr['HEIGHT'],
                 band=hdr['BAND'],
                 zeroPoint=hdr['ZEROPNT'],
                 error_coord=[ hdr['ERR_CRD0'], hdr['ERR_CRD1']],
                 target_device_idx=target_device_idx)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        return Source.from_header(hdr, target_device_idx=target_device_idx)