import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula.lib.n_phot import n_phot
from specula import ASEC2RAD

degree2rad = np.pi / 180.


def _refractivity_dry_air(wavelength_nm):
    """
    Refractivity N = (n - 1) of dry air at standard conditions
    (P = 101325 Pa, T = 288.15 K) using the Edlén (1966) formula.

    Parameters
    ----------
    wavelength_nm : float
        Wavelength in nanometers.

    Returns
    -------
    float
        Refractivity N = (n - 1) at standard conditions.
    """
    sigma = 1e3 / wavelength_nm   # wavenumber in 1/μm
    sigma2 = sigma ** 2
    return (8342.13 + 2406030.0 / (130.0 - sigma2) + 15997.0 / (38.9 - sigma2)) * 1e-8


def _isa_density_ratio(height_m):
    """
    Relative air density ρ(h) / ρ₀ from the International Standard Atmosphere (ISA).

    Uses the tropospheric lapse-rate model (0–11 000 m) and the isothermal
    stratospheric model (11 000–20 000 m).

    Parameters
    ----------
    height_m : float
        Height above sea level in metres.

    Returns
    -------
    float
        Density ratio ρ(h) / ρ₀.
    """
    h = float(height_m)
    if h <= 0.0:
        return 1.0
    elif h <= 11000.0:
        # Troposphere: T(h) = T0 - L*h, exponent = gM/(RL) - 1 = 4.25588
        return ((288.15 - 0.0065 * h) / 288.15) ** 4.25588
    else:
        # Stratosphere: isothermal at 216.65 K
        # ρ_rel(11 km) = (216.65/288.15)^4.25588
        rho_11 = (216.65 / 288.15) ** 4.25588
        return rho_11 * np.exp(-0.00015769 * (h - 11000.0))

class Source(BaseDataObj):
    """
    Source data object.
    Holds the properties of a source, such as polar coordinates, magnitude, wavelength,
    height, band, zero point, and error in coordinates.
    """
    def __init__(self,
                 polar_coordinates: list,
                 magnitude: float,
                 wavelengthInNm: float,
                 height: float = float('inf'),
                 band: str = '',
                 zero_point: float = 0,
                 error_coord: tuple = (0., 0.),
                 verbose: bool = False,
                 wfs_source: 'Source' = None,
                 enable_chromatic_effect: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        Initialize a :class:`~specula.data_objects.source.Source` object.

        Parameters
        ----------
        polar_coordinates : list
            The polar coordinates [radius, angle] of the source (in arcseconds and degrees).
        magnitude : float
            The magnitude of the source.
        wavelengthInNm : float
            The wavelength of the source in nanometers.
        height : float, optional
            The height of the source (default: infinity, i.e., astronomical source).
        band : str, optional
            The photometric band of the source (default: '').
        zero_point : float, optional
            The photometric zero point (default: 0).
        error_coord : tuple, optional
            Error to add to the polar coordinates (default: (0., 0.)).
        verbose : bool, optional
            If True, print verbose output (default: False).
        wfs_source : Source, optional
            Reference to the WFS :class:`~specula.data_objects.source.Source` object.
            Required when ``enable_chromatic_effect`` is True.
        enable_chromatic_effect : bool, optional
            If True, chromatic anisoplanatism shifts are computed via
            :meth:`compute_chromatic_shifts` and applied during atmospheric
            propagation (default: False).
        target_device_idx : int, optional
            Device index for computation (default: None).
        precision : int, optional
            Precision for computation (default: None).
        """
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
        self.zero_point = zero_point
        self.band = band
        self.verbose = verbose
        self.error_coord = error_coord
        self.wfs_source = wfs_source
        self.enable_chromatic_effect = enable_chromatic_effect
        self.chromatic_shifts_m = {}

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PCOORD0'] = self.orig_polar_coordinates[0]
        hdr['PCOORD1'] = self.orig_polar_coordinates[1]
        hdr['MAGNITUD'] = self.magnitude
        hdr['WAVELENG'] = self.wavelengthInNm
        hdr['HEIGHT'] = self.height
        hdr['BAND'] = self.band
        hdr['ZEROPNT'] = self.zero_point
        hdr['ERR_CRD0'] = self.error_coord[0]
        hdr['ERR_CRD1'] = self.error_coord[1]
        return hdr

    # There is no value to get/set
    def get_value(self):
        raise NotImplementedError

    def set_value(self, v):
        raise NotImplementedError

    @property
    def polar_coordinates(self):
        return self._polar_coordinates

    @polar_coordinates.setter
    def polar_coordinates(self, value):
        self._polar_coordinates = np.array(value, dtype=self.dtype)

    @property
    def r(self):
        """
        Get the radius of the source in radians.
        """
        return self._polar_coordinates[0] * ASEC2RAD

    @property
    def r_arcsec(self):
        """
        Get the radius of the source in arcseconds.
        """
        return self._polar_coordinates[0]

    @property
    def phi(self):
        """
        Get the angle of the source in radians.
        """
        return self._polar_coordinates[1] * degree2rad

    @property
    def phi_deg(self):
        """
        Get the angle of the source in degrees.
        """
        return self._polar_coordinates[1]

    @property
    def x_coord(self):
        """
        Get the x coordinate of the source in meters.
        """
        alpha = self._polar_coordinates[0] * ASEC2RAD
        d = self.height * np.sin(alpha)
        return np.cos(np.radians(self._polar_coordinates[1])) * d

    @property
    def y_coord(self):
        """
        Get the y coordinate of the source in meters.
        """
        alpha = self._polar_coordinates[0] * ASEC2RAD
        d = self.height * np.sin(alpha)
        return np.sin(np.radians(self._polar_coordinates[1])) * d

    def compute_chromatic_shifts(self, atmo_layer_list, zenith_angle_deg):
        """
        Pre-compute the chromatic lateral displacement for each *atmospheric* layer.

        Uses the Edlén (1966) refractivity formula for dry air and the
        International Standard Atmosphere (ISA) density profile to evaluate the
        differential lateral shift at each layer height between this source's
        wavelength and the WFS reference wavelength (plane-parallel approximation).

        The result is stored in :attr:`chromatic_shifts_m` as a **dict keyed by
        Layer object**, containing the signed lateral displacement in metres.
        Common layers (pupil stop, DM, etc.) are not included and will
        implicitly receive a zero shift in the propagation code.

        This method must be called (typically from
        :class:`~specula.processing_objects.atmo_propagation.AtmoPropagation`
        during setup) before the interpolators are built.

        Parameters
        ----------
        atmo_layer_list : list of Layer
            Atmospheric turbulence layers only (not common layers such as
            pupil stops or DMs).
        zenith_angle_deg : float
            Observation zenith angle in degrees.

        Notes
        -----
        If :attr:`enable_chromatic_effect` is False, :attr:`wfs_source` is None,
        or the two wavelengths are identical, all shifts are zero.
        """
        self.chromatic_shifts_m = {}

        if not self.enable_chromatic_effect:
            return
        if self.wfs_source is None:
            return
        if self.wavelengthInNm == self.wfs_source.wavelengthInNm:
            return

        delta_N = (_refractivity_dry_air(self.wavelengthInNm)
                   - _refractivity_dry_air(self.wfs_source.wavelengthInNm))
        tan_z = np.tan(np.radians(zenith_angle_deg))

        for layer in atmo_layer_list:
            rho_rel = _isa_density_ratio(float(layer.height))
            self.chromatic_shifts_m[layer] = delta_N * rho_rel * float(layer.height) * tan_z

    def phot_density(self):
        """
        Get the photometric density of the source.
        """
        if self.zero_point > 0:
            e0 = self.zero_point
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
                 zero_point=hdr['ZEROPNT'],
                 error_coord=[ hdr['ERR_CRD0'], hdr['ERR_CRD1']],
                 target_device_idx=target_device_idx)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        return Source.from_header(hdr, target_device_idx=target_device_idx)