from specula import RAD2ASEC, fuse
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.make_xy import make_xy
from specula.lib.toccd import toccd


@fuse(kernel_name='abs2')
def abs2(u_fp, out, xp):
    out[:] = xp.real(u_fp * xp.conj(u_fp))


class CiaoCiaoSensor(BaseProcessingObj):
    """
    Ciao-Ciao sensor (rotating shearing interferometer) processing object.
    The object duplicates the input electric field, rotates one copy,
    applies a user-defined tip/tilt to the rotated copy, then coherently
    sums the two complex fields and outputs the resulting intensity.

    Parameters
    ----------
    wavelengthInNm : float
        Working wavelength in nanometers.
    number_px : int, optional
        Output side in pixels. If None, output has the same side as input EF.
        Default is None.
    diffRotAngleInDeg : float, optional
        Differential rotation angle in degrees applied to the second branch.
        Default is 180.0.
    tiltInArcsec : tuple[float, float] | float, optional
        User-defined tilt (x, y) in arcseconds applied to the rotated branch.
        If a scalar is provided, it is interpreted as x tilt and y=0.
        Default is (0.0, 0.0).
    rotAnglePhInDeg : float, optional
        Rotation angle in degrees applied to the input branch.
        Default is 0.0.
    xShiftPhInPixel : float, optional
        X shift in pixels applied to the input branch.
        Default is 0.0.
    yShiftPhInPixel : float, optional
        Y shift in pixels applied to the input branch.
        Default is 0.0.
    magnification : float, optional
        Magnification applied to the input branch.
        Default is 1.0.
    xShiftDiffPhInPixel : float, optional
        Additional differential X shift in pixels applied only to the second branch.
        Default is 0.0.
    yShiftDiffPhInPixel : float, optional
        Additional differential Y shift in pixels applied only to the second branch.
        Default is 0.0.
    magnificationDiff : float, optional
        Additional differential magnification applied only to the second branch.
        Default is 1.0.
    channel_flux : float, optional
        Relative flux of the input branch in [0, 1].
        The rotated branch uses (1 - channel_flux).
        Default is 0.5 (balanced channels).
    normalize_flux : bool, optional
        If True, normalize output intensity to input photons
        ``S0 * masked_area``. Default is True.
    target_device_idx : int, optional
        Target device index for GPU processing. Default is None.
    precision : int, optional
        Numerical precision (e.g., 32 or 64). Default is None.
    """

    def __init__(self,
                 wavelengthInNm: float,
                 number_px: int,
                 diffRotAngleInDeg: float = 180.0,
                 tiltInArcsec=(0.0, 0.0),
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,
                 yShiftPhInPixel: float = 0.0,
                 magnification: float = 1.0,
                 xShiftDiffPhInPixel: float = 0.0,
                 yShiftDiffPhInPixel: float = 0.0,
                 magnificationDiff: float = 1.0,
                 channel_flux: float = 0.5,
                 normalize_flux: bool = True,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.wavelength_in_nm = float(wavelengthInNm)
        if number_px <= 0:
            raise ValueError('number_px must be a positive integer')
        self.number_px = int(number_px)
        self.diff_rot_angle_in_deg = float(diffRotAngleInDeg)
        self.normalize_flux = bool(normalize_flux)
        self.rotAnglePhInDeg = float(rotAnglePhInDeg)
        self.xShiftPhInPixel = float(xShiftPhInPixel)
        self.yShiftPhInPixel = float(yShiftPhInPixel)
        self.magnification = float(magnification)
        self.xShiftDiffPhInPixel = float(xShiftDiffPhInPixel)
        self.yShiftDiffPhInPixel = float(yShiftDiffPhInPixel)
        self.magnificationDiff = float(magnificationDiff)
        if channel_flux < 0.0 or channel_flux > 1.0:
            raise ValueError('channel_flux must be in the [0, 1] range')
        self.channel_flux = channel_flux

        norm_flux_0 = 2.0 * self.channel_flux
        norm_flux_1 = 2.0 * (1.0 - self.channel_flux)
        self._channel_amp_0 = norm_flux_0 ** 0.5
        self._channel_amp_1 = norm_flux_1 ** 0.5

        if self.magnification < 1e-6:
            raise ValueError('magnification must be greater than 1e-6')
        if self.magnificationDiff < 1e-6:
            raise ValueError('magnificationDiff must be greater than 1e-6')

        self.tilt_in_arcsec = self._parse_tilt(tiltInArcsec)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self._out_i = Intensity(self.number_px, self.number_px,
                                precision=self.precision,
                                target_device_idx=self.target_device_idx)
        self.outputs['out_i'] = self._out_i

        self._ef = None
        self._rot_ef = None
        self._interf_ef = None
        self._interf_i = None
        self._tilt_exp = None

        self.ef_interpolator_in = None
        self.ef_interpolator_rot = None

    def _parse_tilt(self, tiltInArcsec):
        if isinstance(tiltInArcsec, (int, float)):
            return float(tiltInArcsec), 0.0

        if len(tiltInArcsec) != 2:
            raise ValueError('tiltInArcsec must be a scalar or a tuple/list of length 2')

        return float(tiltInArcsec[0]), float(tiltInArcsec[1])

    def _build_tilt_phase_map_nm(self, in_ef):
        tilt_x_arcsec, tilt_y_arcsec = self.tilt_in_arcsec
        theta_x = tilt_x_arcsec / RAD2ASEC
        theta_y = tilt_y_arcsec / RAD2ASEC

        xx, yy = make_xy(in_ef.size[0], in_ef.pixel_pitch, xp=self.xp)
        return (theta_x * xx + theta_y * yy) * 1e9

    def setup(self):
        super().setup()

        in_ef = self.local_inputs['in_ef']
        dimx, dimy = in_ef.size

        self.ef_interpolator_in = EFInterpolator(
            in_ef=in_ef,
            out_shape=in_ef.size,
            rotAnglePhInDeg=self.rotAnglePhInDeg,
            xShiftPhInPixel=self.xShiftPhInPixel,
            yShiftPhInPixel=self.yShiftPhInPixel,
            magnification=self.magnification,
            use_out_ef_cache=True,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        in_ef_for_rot = self.ef_interpolator_in.interpolated_ef()
        self.ef_interpolator_rot = EFInterpolator(
            in_ef=in_ef_for_rot,
            out_shape=in_ef_for_rot.size,
            rotAnglePhInDeg=self.diff_rot_angle_in_deg,
            xShiftPhInPixel=self.xShiftDiffPhInPixel,
            yShiftPhInPixel=self.yShiftDiffPhInPixel,
            magnification=self.magnificationDiff,
            use_out_ef_cache=True,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        self._ef = self.xp.zeros((dimx, dimy), dtype=self.complex_dtype)
        self._rot_ef = self.xp.zeros((dimx, dimy), dtype=self.complex_dtype)
        self._interf_ef = self.xp.zeros((dimx, dimy), dtype=self.complex_dtype)
        self._interf_i = self.xp.zeros((dimx, dimy), dtype=self.dtype)

        tilt_phase_nm = self._build_tilt_phase_map_nm(in_ef)
        tilt_phase_rad = tilt_phase_nm * ((2 * self.xp.pi) / self.wavelength_in_nm)
        self._tilt_exp = self.xp.exp(1j * tilt_phase_rad, dtype=self.complex_dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        self.ef_interpolator_in.interpolate()
        in_ef_interp = self.ef_interpolator_in.interpolated_ef()
        in_ef_interp.ef_at_lambda(self.wavelength_in_nm, out=self._ef)
        self._ef *= self._channel_amp_0

        self.ef_interpolator_rot.interpolate()
        rot_ef_interp = self.ef_interpolator_rot.interpolated_ef()
        rot_ef_interp.ef_at_lambda(self.wavelength_in_nm, out=self._rot_ef)
        self._rot_ef *= self._channel_amp_1
        self._rot_ef *= self._tilt_exp

    def trigger_code(self):
        self._interf_ef[:] = self._ef + self._rot_ef
        abs2(self._interf_ef, self._interf_i, xp=self.xp)

    def post_trigger(self):
        super().post_trigger()

        # Resample to output resolution if needed
        if self.number_px == self._interf_i.shape[0]:
            self._out_i.i[:] = self._interf_i
        else:
            self._out_i.i[:] = toccd(self._interf_i,
                                     (self.number_px, self.number_px),
                                     xp=self.xp)

        # Normalize flux
        if self.normalize_flux:
            in_ef = self.local_inputs['in_ef']
            phot = in_ef.S0 * in_ef.masked_area()
            output_sum = self.xp.sum(self._out_i.i)
            if output_sum > 0:
                self._out_i.i *= phot / output_sum

        self._out_i.generation_time = self.current_time
