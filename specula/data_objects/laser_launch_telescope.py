
import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula.data_objects.simul_params import SimulParams
from iteras.lib import storage


def launcher_in_wfs_frame(tel_position, beacon_tt, wfs_rotation_in_deg):
    """
    Launcher position and beacon tilt, given in the telescope frame, in the
    frame of a WFS rotated by wfs_rotation_in_deg (the rotAnglePhInDeg of SH).

    The WFS sees the pupil rotated by this angle (EFInterpolator): a point
    at p in the telescope pupil appears at R p, with R the rotation by
    +wfs_rotation_in_deg in the (x, y) plane, x along the columns and y along
    the rows of the pupil arrays. The launcher position (x, y; z unchanged)
    and the beacon tilt (x, y) are rotated in the same way.

    Parameters
    ----------
    tel_position : array_like [m]
        Launcher position [x, y, z] in the telescope frame.
    beacon_tt : array_like [arcsec]
        Beacon tilt [x, y] in the telescope frame.
    wfs_rotation_in_deg : float [deg]
        Rotation of the WFS.

    Returns
    -------
    tuple of list
        (position [x, y, z], tilt [x, y]) in the WFS frame.
    """
    a = np.radians(wfs_rotation_in_deg)
    c, s = np.cos(a), np.sin(a)
    x, y, z = (float(v) for v in tel_position)
    tx, ty = (float(v) for v in beacon_tt)
    return [c * x - s * y, s * x + c * y, z], [c * tx - s * ty, s * tx + c * ty]


class LaserLaunchTelescope(BaseDataObj):
    """
    Laser Launch Telescope data object.
    This class holds the parameters of the laser launch telescope, such as spot size,
    position, focus and tilt.
    """
    def __init__(self,
                 simul_params: SimulParams = None,
                 spot_size: float = 0.0,
                 tel_position: list = [],
                 beacon_focus: float = 90e3,
                 beacon_tt: list = [0.0, 0.0],
                 absolute_coordinates: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
        ):
        """
        Parameters
        ----------

        simul_params : SimulParams
            The simulation parameters object, required to get the zenith angle.
        spot_size : float [arcsec]
            The size of the laser spot.
        tel_position : list [m]
            The x, y and z position of the launch telescope w.r.t. the telescope.
        beacon_focus : float [m]
            The distance from the telescope pupil to beacon focus.
        beacon_tt : list [arcsec]
            The tilt and tip of the beacon.
        absolute_coordinates : bool, optional
            If False (default), tel_position and beacon_tt are given in the
            frame of the WFS that uses the launcher (in the configuration,
            they must already account for the WFS rotation). If True, they are
            given in the telescope frame, and each WFS rotates them by its own
            rotation (see launcher_in_wfs_frame()): one launcher can then serve
            WFSs with different rotations.

        TODO the empty tel_position array is actually significant, because
            it is checked in the SH code to manage the kernels,
            but gives some problems for the FITS header (when reading from disk,
            it won't be empty anymore)
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        if self.simul_params is not None:
            self.zenithAngleInDeg = self.simul_params.zenithAngleInDeg
        else:
            if len(tel_position) != 0:
                raise ValueError("Error: simul_params is required by LaserLaunchTelescope"
                                 " to get zenith angle when tel_position is provided")
            else:
                self.zenithAngleInDeg = 0.0

        if self.zenithAngleInDeg is not None:
            self.airmass = 1.0 / np.cos(np.radians(self.zenithAngleInDeg), dtype=self.dtype)
            self.logger.info(f'zenith angle is defined as: {self.zenithAngleInDeg} deg')
            self.logger.info(f'airmass is: {self.airmass}')
        else:
            self.airmass = 1.0

        self.spot_size = spot_size
        self.tel_pos = tel_position
        self.beacon_focus = beacon_focus * self.airmass
        self.beacon_tt = beacon_tt
        self.absolute_coordinates = bool(absolute_coordinates)

    def wfs_frame(self, wfs_rotation_in_deg):
        """
        Launcher position and beacon tilt in the frame of a WFS rotated by
        wfs_rotation_in_deg: rotated if absolute_coordinates, as they are
        otherwise.

        Returns
        -------
        tuple of list
            (position [x, y, z], tilt [x, y])
        """
        if not self.absolute_coordinates or len(self.tel_pos) == 0:
            return self.tel_pos, self.beacon_tt
        return launcher_in_wfs_frame(self.tel_pos, self.beacon_tt, wfs_rotation_in_deg)

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
        hdr['ABSCOORD'] = self.absolute_coordinates
        return hdr

    def save(self, filename, overwrite=False):
        filename = storage.object_filename(filename)
        storage.write_object(filename, self.get_fits_header(), {}, layout='placeholder',
                             overwrite=overwrite)

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f'Error: unknown version {version} in header')

        llt = LaserLaunchTelescope(
            simul_params=SimulParams(zenithAngleInDeg=0.0),
            spot_size = hdr['SPOTSIZE'],
            tel_position = [hdr['TELPOS_X'], hdr['TELPOS_Y'], hdr['TELPOS_Z']],
            beacon_focus = hdr['BEAC_FOC'],
            beacon_tt = [hdr['BEAC_TT0'], hdr['BEAC_TT1']],
            absolute_coordinates = hdr.get('ABSCOORD', False),
            target_device_idx=target_device_idx)
        return llt

    @staticmethod
    def restore(filename, target_device_idx=None):
        with storage.read_object(filename) as f:
            hdr = f.header
        return LaserLaunchTelescope.from_header(hdr, target_device_idx=target_device_idx)
