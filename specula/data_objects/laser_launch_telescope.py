
import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj

class LaserLaunchTelescope(BaseDataObj):
    '''
    Laser Launch Telescope
    
    args:
    ----------
    spot_size : float
        The size of the laser spot in arcsec.
    tel_position : list
        The x, y and z position of the launch telescope w.r.t. the telescope in m.
    beacon_focus : float
        The distance from the telescope pupil to beacon focus in m.
    beacon_tt : list
        The tilt and tip of the beacon in arcsec.
    '''

    def __init__(self,
                 spot_size: float = 0.0,
                 tel_position: list = [],
                 beacon_focus: float = 90e3,
                 beacon_tt: list = [0.0, 0.0],
                 target_device_idx: int = None, 
                 precision: int = None
        ):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.spot_size = spot_size
        self.tel_pos = tel_position
        self.beacon_focus = beacon_focus
        self.beacon_tt = beacon_tt

    def get_value(self):
        raise NotImplementedError

    def set_value(self, v, force_copy=True):
        raise NotImplementedError

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['SPOTSIZE'] = self.spot_size
        hdr['TELPOS_X'] = self.tel_pos[0]
        hdr['TELPOS_Y'] = self.tel_pos[1]
        hdr['TELPOS_Z'] = self.tel_pos[2]
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
