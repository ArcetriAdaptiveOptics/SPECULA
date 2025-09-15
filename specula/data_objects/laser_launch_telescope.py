
import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula.data_objects.simul_params import SimulParams

class LaserLaunchTelescope(BaseDataObj):
    '''
    Laser Launch Telescope
    
    args:
    ----------
    simul_params : SimulParams
        The simulation parameters object, required to get the zenith angle.
    spot_size : float
        The size of the laser spot in arcsec.
    tel_position : list
        The x, y and z position of the launch telescope w.r.t. the telescope in m.
    beacon_focus : float
        The distance from the telescope pupil to beacon focus in m.
    beacon_tt : list
        The tilt and tip of the beacon in arcsec.

    TODO the empty tel_position array is actually significant, because
         it is checked in the SH code to manage the kernels,
         but gives some problems for the FITS header (when reading from disk,
         it won't be empty anymore)
    '''

    def __init__(self,
                 simul_params: SimulParams = None,
                 spot_size: float = 0.0,
                 tel_position: list = [],
                 beacon_focus: float = 90e3,
                 beacon_tt: list = [0.0, 0.0],
                 target_device_idx: int = None,
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        if self.simul_params is not None:
            self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg
        else:
            self.zenithAngleInDeg = 0.0

        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg), dtype=self.dtype)
            print(f'AtmoEvolution: zenith angle is defined as: {self.zenithAngleInDeg} deg')
            print(f'AtmoEvolution: airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        self.spot_size = spot_size
        self.tel_pos = tel_position
        self.beacon_focus = beacon_focus * self.airmass
        self.beacon_tt = beacon_tt

    def get_value(self):
        raise NotImplementedError

    def set_value(self, v):
        raise NotImplementedError

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['SPOTSIZE'] = self.spot_size

        tel_pos = [0.0, 0.0, 0.0] if self.tel_pos == [] else self.tel_pos
        hdr['TELPOS_X'] = tel_pos[0]
        hdr['TELPOS_Y'] = tel_pos[1]
        hdr['TELPOS_Z'] = tel_pos[2]
        hdr['BEAC_FOC'] = self.beacon_focus
        hdr['BEAC_TT0'] = self.beacon_tt[0]
        hdr['BEAC_TT1'] = self.beacon_tt[1]
        return hdr

    def save(self, filename, overwrite=False):
        if not filename.endswith('.fits'):
            filename += '.fits'
        hdr = self.get_fits_header()
        # Save fits file
        fits.writeto(filename, np.zeros(2), hdr, overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f'Error: unknown version {version} in header')

        llt = LaserLaunchTelescope(
            spot_size = hdr['SPOTSIZE'],
            tel_position = [hdr['TELPOS_X'], hdr['TELPOS_Y'], hdr['TELPOS_Z']],
            beacon_focus = hdr['BEAC_FOC'],
            beacon_tt = [hdr['BEAC_TT0'], hdr['BEAC_TT1']],
            target_device_idx=target_device_idx)
        return llt

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        return LaserLaunchTelescope.from_header(hdr, target_device_idx=target_device_idx)
