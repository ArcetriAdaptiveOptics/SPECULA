import numpy as np
from astropy.io import fits

from specula.base_data_obj import BaseDataObj
from specula.lib.n_phot import n_phot
from specula.lib.air_refraction import MatharAirRefraction
from specula import ASEC2RAD

degree2rad = np.pi / 180.


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
                 telescope_altitude_m: float = 3064.0,
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
        telescope_altitude_m : float, optional
            Altitude of the telescope above sea level in meters (default: 3064.0 for ELT).
            It is used for computing atmospheric pressure and chromatic shifts if enabled.
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

        polar_coordinates = np.array(polar_coordinates, dtype=self.dtype) \
                          + np.array(error_coord, dtype=self.dtype)
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
        self.telescope_altitude_m = telescope_altitude_m

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
        
        Uses the MatharAirRefraction (Ciddor+Mathar) model to calculate precise 
        refractivity across Visible and Mid-IR bands. Then applies the NASA standard 
        atmospheric pressure profile to compute the exact lateral shift using the 
        Devaney 2024 plane-parallel equations (Eq. 1 and Eq. 6).
        
        Parameters
        ----------
        atmo_layer_list : list of Layer
            Atmospheric turbulence layers only.
        zenith_angle_deg : float
            Observation zenith angle in degrees.
        """
        self.chromatic_shifts_m = {}

        if not self.enable_chromatic_effect or self.wfs_source is None:
            return
        if self.wavelengthInNm == self.wfs_source.wavelengthInNm:
            return

        # 1. Compute delta refractivity using Standard Conditions (15 C, 101325 Pa, 0% RH)
        air_model = MatharAirRefraction()
        n_minus_1_sci = air_model.get_refractive_index(self.wavelengthInNm * 1e-9)
        n_minus_1_wfs = air_model.get_refractive_index(self.wfs_source.wavelengthInNm * 1e-9)

        delta_N = n_minus_1_wfs - n_minus_1_sci

        # 2. Parameters for Devaney 2024 Eq. 1
        zeta_rad = np.radians(zenith_angle_deg)
        sec_z = 1.0 / np.cos(zeta_rad)
        tan_z = np.tan(zeta_rad)

        g = 9.8 # m/s^2
        rho_s = 1.225 # kg/m^3

        # NASA Atmospheric Pressure Model P(h)
        def get_pressure_nasa(h_asl):
            if h_asl < 11000.0:
                T_h = 288.08 - 0.00649 * h_asl
                return 1012.9 * (T_h / 288.08)**5.256
            elif h_asl < 25000.0:
                return 226.5 * np.exp(1.73 - 0.000157 * h_asl)
            else:
                # Simplified model from Devaney 2024 for h > 25km
                T_h = 141.94 + 0.00299 * h_asl
                return 24.88 * (T_h / 216.6)**-11.388

        # Pressure at telescope altitude (P0 in mbar)
        P_0_mbar = get_pressure_nasa(self.telescope_altitude_m)

        # Lateral separation of two rays at the telescope aperture (Devaney Eq 1)
        # Note: Convert mbar to Pascal (1 mbar = 100 Pa)
        delta_b0 = delta_N * sec_z * tan_z * ((P_0_mbar * 100.0) / (g * rho_s))

        for layer in atmo_layer_list:
            # Assuming layer.height is the distance above the telescope
            h_asl = self.telescope_altitude_m + float(layer.height)
            P_h_mbar = get_pressure_nasa(h_asl)

            # Lateral separation at altitude h (Devaney Eq 6)
            shift_at_h = delta_b0 * (1.0 - (P_h_mbar / P_0_mbar))
            self.chromatic_shifts_m[layer] = shift_at_h

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