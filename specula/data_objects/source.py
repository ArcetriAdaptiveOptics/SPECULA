import numpy as np

from specula.base_data_obj import BaseDataObj
from specula.lib.n_phot import n_phot

sec2rad = 4.848e-6
degree2rad = np.pi / 180.

class Source(BaseDataObj):
    '''source'''

    def __init__(self,
                 polar_coordinate,
                 magnitude: float,
                 wavelengthInNm: float,
                 height: float=float('inf'),
                 band: str='',
                 zeroPoint: float=0,
                 zenithAngleInDeg: float=None,
                 error_coord=(0., 0.),
                 verbose: bool=False):
        super().__init__()
        
        if zenithAngleInDeg is not None:
            airmass = 1. / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
            height *= airmass
            print(f'get_source: changing source height by airmass value ({airmass})')

        polar_coordinate = np.array(polar_coordinate, dtype=self.dtype) + np.array(error_coord, dtype=self.dtype)
        if any(error_coord):
            print(f'there is a desired error ({error_coord[0]},{error_coord[1]}) on source coordinates.')
            print(f'final coordinates are: {polar_coordinate[0]},{polar_coordinate[1]}')
        
        self.polar_coordinate = polar_coordinate
        self.height = height
        self.magnitude = magnitude
        self.wavelengthInNm = wavelengthInNm
        self.zeroPoint = zeroPoint
        self.band = band
        self.verbose = verbose

    @property
    def polar_coordinate(self):
        return self._polar_coordinate

    @polar_coordinate.setter
    def polar_coordinate(self, value):
        self._polar_coordinate = np.array(value, dtype=self.dtype)

    @property
    def r(self):
        return self._polar_coordinate[0] * sec2rad

    @property
    def r_arcsec(self):
        return self._polar_coordinate[0]

    @property
    def phi(self):
        return self._polar_coordinate[1] * degree2rad

    @property
    def phi_deg(self):
        return self._polar_coordinate[1]

    @property
    def x_coord(self):
        alpha = self._polar_coordinate[0] * 4.848e-6
        d = self.height * np.sin(alpha)
        return np.cos(np.radians(self._polar_coordinate[1])) * d

    @property
    def y_coord(self):
        alpha = self._polar_coordinate[0] * 4.848e-6
        d = self.height * np.sin(alpha)
        return np.sin(np.radians(self._polar_coordinate[1])) * d

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

    # TODO: this is a data object, not a processing object
    def finalize(self):
        pass
