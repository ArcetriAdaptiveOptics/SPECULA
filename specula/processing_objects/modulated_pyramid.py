from specula import cpuArray, fuse, np
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.interp2d import Interp2D

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_xy import make_xy
from specula.lib.make_mask import make_mask
from specula.lib.toccd import toccd
from specula.lib.calc_geometry import calc_geometry

@fuse(kernel_name='pyr_input')
def pyr_input(ef, ey, ex, out, xp):
    out[...] = ef * ey * ex


@fuse(kernel_name='pyr_fp_mask')
def pyr_fp_mask(u_fp, ffv, fpsf, masked_exp, xp):
    fpsf += xp.real(u_fp * xp.conj(u_fp)) * ffv
    u_fp *= masked_exp


@fuse(kernel_name='pyr_abs2_acc')
def pyr_abs2_acc(v, norm, ffv, acc, xp):
    v_norm = v * norm
    acc += xp.real(v_norm * xp.conj(v_norm)) * ffv


class ModulatedPyramid(BaseProcessingObj):
    r"""
    Pyramid wavefront sensor processing object. Includes tip-tilt modulation.
    
    This class implements a modulated pyramid WFS that works with point sources or
    can be used as a base class for extended source implementations. The modulation
    can be circular (standard), or linear (vertical, horizontal, alternating).
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object containing pixel_pupil and pixel_pitch
    wavelengthInNm : float [nm]
        Working wavelength in nanometers
    fov : float [arcsec]
        Field of view in arcseconds (a field stop may be applied in the focal plane to limit FoV)
    pup_diam : float [pixels]
        Pupil diameter in pixels
    output_resolution : int [pixels]
        Output CCD side length in pixels
    mod_amp : float [lambda/D], optional
        Modulation amplitude in lambda/D units (default: 3.0)
    mod_step : int [1], optional
        Number of modulation steps. If None, automatically calculated based on
        mod_amp and mod_type (default: None)
    mod_type : str
        Modulation type: 'circular', 'vertical', 'horizontal', or 'alternating'
        (default: 'circular')
    fov_errinf : float [1], optional
        Accepted error in reducing FoV (default: 0.1, i.e., -10%)
    fov_errsup : float [1], optional
        Accepted error in enlarging FoV (default: 2.0, i.e., +100%)
    pup_dist : float [pixels], optional
        Pupil distance in pixels. If None, calculated from pup_diam and pup_margin
    pup_margin : int [pixels], optional
        Margin around pupils in pixels (default: 2)
    fft_res : float [1], optional
        Minimum PSF sampling (default: 3.0, i.e., 3 pixels per PSF FWHM i.e. lambda/D)
    fp_obs : float [lambda/D], optional
        Diameter of the central obstruction of the focal plane mask in lambda/D
        units (default: None, no obstruction)
    pup_shifts : tuple [pixels], optional
        Static pupil shifts in pixels (x, y) (default: (0.0, 0.0)). The
        optional input in_misregistration can change them, and
        rotAnglePhInDeg, during the simulation. The shifts move the pupil
        images on the detector, while the rotation is applied to the input
        field.
    pyr_tlt_coeff : list [2,4], optional
        Pyramid tilt coefficients for each face. (default: None)
        The coefficient multiplies the x-tilt or y-tilt (rows) of each of each of the
        4 pyramid faces (columns). A coefficient greater than 1 pushes the pupil outward,
        away from the frame center, while a coefficient lower than 1 pulls it inward,
        towards the frame center, with respect to the nominal distance pup_dist.
        The x coefficient (first row) shifts in the horizontal direction,
        the y coefficient (second row) shifts in the vertical direction.
        The faces (columns) are numbered in array indices of the focal plane,
        with row 0 at the top (an image shown with origin='lower' is flipped
        vertically)::

              column ->
            +-----+-----+
            |  0  |  1  |   row
            +-----+-----+    |
            |  2  |  3  |    v
            +-----+-----+

        The slope of each face changes around the tip, and the edges between
        the faces move accordingly (see Notes).
        By default, no deviation is applied to the nominal tilt.
    pyr_roof_ld : float [lambda/D], optional
        Length of the roof defect in lambda/D units (default: 0.0, no roof).
        With a roof, two opposite faces meet along a segment (the roof)
        instead of in a single tip: if positive, faces 0 and 3 (see
        pyr_tlt_coeff), if negative, faces 1 and 2::

              pyr_roof_ld > 0             pyr_roof_ld < 0
            +-------+-----+            +-----+-------+
            |   0   |  1  |            |  0  |   1   |
            |      /+-----+            +-----+\      |
            +-----/       |            |       \-----+
            |  2  |   3   |            |   2   |  3  |
            +-----+-------+            +-------+-----+

        The midpoint of the roof stays on the optical axis. The length is
        exact for the nominal face slopes.
    pyr_edge_def_ld : float [lambda/D], optional
        Edge defect size in lambda/D units (default: 0.0)
    pyr_tip_def_ld : float [lambda/D], optional
        Tip defect size in lambda/D units (default: 0.0)
    pyr_tip_maya_ld : float [lambda/D], optional
        Maya Pyramid (i.e. flat tip) defect size in lambda/D units (default: 0.0)
    pyr_max_side_ld : float [lambda/D], optional
        Maximum radial support of the pyramid from the tip in lambda/D units.
        Outside this radius the pyramid surface is forced to 0.0; when 0.0 this
        option is disabled (default: 0.0)
    min_pup_dist : float [pixels], optional
        Minimum pupil distance constraint (default: None)
    rotAnglePhInDeg : float [deg], optional
        Rotation angle of input phase in degrees (default: 0.0)
    xShiftPhInPixel : float [pixels], optional
        X shift of input phase in pixels (default: 0.0)
    yShiftPhInPixel : float [pixels], optional
        Y shift of input phase in pixels (default: 0.0)
    magnification : float [1], optional
        Magnification factor applied to input phase (default: 1.0)
    force_extrapolation : bool
        Force extrapolation of input electric field (required by SprintPyr)
    target_device_idx : int [1], optional
        GPU device index (default: None, uses default device, -1 for CPU)
    precision : int [1], optional
        Numerical precision: 32 (1) or 64 (0) bits (default: None, uses system default)
    
    Notes
    -----
    The modulation types have different characteristics:
    - 'circular': Standard pyramid modulation, uniform flux distribution
    - 'vertical'/'horizontal': Linear modulation along one axis, flux weighted by 1/cos(tilt)
    - 'alternating': Switches between vertical and horizontal on consecutive frames
    - pyr_tip_def_ld and pyr_tip_maya_ld represent different types of pyramid tip imperfections.

    The pyramid surface is the envelope of the planes of its four faces: at
    each point of the focal plane the phase delay is the minimum over the
    four planes, so each point belongs to the face whose plane is lowest, as
    in a real prism. Each plane passes through the tip, with slopes given by
    pyr_tlt_coeff. The roof (pyr_roof_ld) lowers the planes of two opposite
    faces and raises the other two by the same amount: the two lowered faces
    then meet along a segment centred on the optical axis, and the edges
    between adjacent faces move by L / (2 sqrt(2)), with L the roof
    length. A decentred pyramid (beam off the tip) is
    not a parameter of the pyramid: it is equivalent to a static tip-tilt
    of the input field.

    For linear modulation, the flux weighting compensates for intensity loss at large tilts,
    ensuring uniform contribution from all modulation positions.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 pup_diam: float,
                 output_resolution: int,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 mod_type: str = 'circular',  # 'circular', 'vertical', 'horizontal', 'alternating'
                 fov_errinf: float = 0.1,
                 fov_errsup: float = 10,
                 pup_dist: float = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: list = None,
                 pyr_roof_ld: float = 0.0,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 pyr_max_side_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,
                 yShiftPhInPixel: float = 0.0,
                 magnification: float = 1.0,
                 force_extrapolation: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.fov = fov
        self.pup_diam = pup_diam

        result = self.calc_pyr_geometry(self.pixel_pupil,
                                        self.pixel_pitch,
                                        wavelengthInNm,
                                        self.fov,
                                        self.pup_diam,
                                        ccd_side=output_resolution,
                                        fov_errinf=fov_errinf,
                                        fov_errsup=fov_errsup,
                                        pup_dist=pup_dist,
                                        pup_margin=pup_margin,
                                        fft_res=fft_res,
                                        min_pup_dist=min_pup_dist)

        fov_res = result['fov_res']
        self.fp_masking = result['fp_masking']
        fft_res = result['fft_res']
        tilt_scale = result['tilt_scale']
        fft_sampling = result['fft_sampling']
        fft_padding = result['fft_padding']
        fft_totsize = result['fft_totsize']
        toccd_side = result['toccd_side']
        final_ccd_side = result['final_ccd_side']

        # Compute focal plane central obstruction dimension ratio
        # make_mask() takes it as a fraction of the field stop diameter, which is
        # fp_masking times the FFT frame (fft_totsize / fft_res lambda/D)
        if fp_obs is not None:
            fp_obsratio = fp_obs / (self.fp_masking * fft_totsize / fft_res)
        else:
            fp_obsratio = 0

        self.wavelength_in_nm = wavelengthInNm
        self.fov_res = fov_res
        self.fft_res = fft_res
        self.tilt_scale = tilt_scale
        self.fft_sampling = fft_sampling
        self.fft_padding = fft_padding
        self.fft_totsize = fft_totsize
        # The (-1)^(x+y) focal plane centering requires an even FFT size
        assert self.fft_totsize % 2 == 0, 'fft_totsize must be even'
        self.toccd_side = int(toccd_side)
        self.final_ccd_side = final_ccd_side
        self.pyr_tlt_coeff = pyr_tlt_coeff
        self.pyr_roof_ld = float(pyr_roof_ld)
        self.pyr_edge_def_ld = pyr_edge_def_ld
        self.pyr_tip_def_ld = pyr_tip_def_ld
        self.pyr_tip_maya_ld = pyr_tip_maya_ld
        self.pyr_max_side_ld = pyr_max_side_ld
        self.rotAnglePhInDeg = rotAnglePhInDeg
        self.xShiftPhInPixel = xShiftPhInPixel
        self.yShiftPhInPixel = yShiftPhInPixel
        self.magnification = magnification
        self.force_extrapolation = force_extrapolation
        self.pup_shifts = pup_shifts

        # interpolation settings
        self.pup_shift_interp = None
        self._do_pup_shift = False
        self._pup_pyr_interpolated = None
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        # Store modulation type
        valid_mod_types = ['circular', 'vertical', 'horizontal', 'alternating']
        if mod_type not in valid_mod_types:
            raise ValueError(f"mod_type must be one of {valid_mod_types}, got {mod_type}")
        self.mod_type = mod_type
        # Add iteration counter for alternating modulation
        self.iter = self.to_xp([0])
        # Photons per unit of masked area of the input field (S0 * pixel_pitch**2):
        # a host value, given to the GPU work of trigger_code() through this array
        self._s0_area = self.xp.zeros((), dtype=self.dtype)
        self._s0_area_host = None

        if mod_step is None:
            mod_step = self.default_mod_step(mod_amp, mod_type)
        elif mod_step == 0:
            raise ValueError('mod_step cannot be zero')
        elif mod_step < 0:
            raise ValueError('mod_step must be a positive integer')
        elif int(mod_step) != mod_step:
            raise ValueError('Modulation step number is not an integer')
        elif mod_type == 'circular' and mod_step < np.around(2 * np.pi * mod_amp):
            raise ValueError(
                f'Number of modulation steps is too small ({mod_step}), '
                f'it must be at least 2*pi times the modulation amplitude '
                f'({np.around(2 * np.pi * mod_amp)})!'
            )
        elif mod_type != 'circular' and mod_step < round(mod_amp) * 2 + 1:
            raise ValueError(
                f'Number of modulation steps is too small ({mod_step}), '
                f'linear modulation needs at least {round(mod_amp) * 2 + 1} '
                f'(about one per lambda/D along the 2*mod_amp path)!'
            )

        if self.pyr_tlt_coeff is not None:
            self.pyr_tlt_coeff = self.xp.array(self.pyr_tlt_coeff)
            if self.pyr_tlt_coeff.shape != (2,4):
                raise ValueError(f'Unexpected pyr_tlt_coeff shape: expected shape is (2,4), input shape is {self.pyr_tlt_coeff.shape}')
            if self.xp.min(self.pyr_tlt_coeff) < 0:
                raise ValueError(f'Expected pyr_tlt_coeff to be positive to preserve pupil ordering, but the minimum is: {self.xp.min(self.pyr_tlt_coeff):1.2f}')

        self.mod_steps = int(mod_step)
        self.mod_amp = mod_amp
        self.flux_factor_vector = None
        self.factor = None

        self.out_i = Intensity(final_ccd_side, final_ccd_side, precision=self.precision,
                               target_device_idx=self.target_device_idx)
        self.psf_tot = BaseValue(value=self.xp.zeros((fft_totsize, fft_totsize), dtype=self.dtype),
                                 target_device_idx=self.target_device_idx,
                                 precision=precision)
        self.psf_bfm = BaseValue(value=self.xp.zeros((fft_totsize, fft_totsize), dtype=self.dtype),
                                 target_device_idx=self.target_device_idx,
                                 precision=precision)
        self.transmission = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                      target_device_idx=self.target_device_idx,
                                      precision=precision)        
        self.flux_frac_inside_ccd = BaseValue(value=self.xp.ones(1, dtype=self.dtype),
                                      target_device_idx=self.target_device_idx,
                                      precision=precision)

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.inputs['in_misregistration'] = InputValue(type=BaseValue, optional=True)
        self._misregistration = None     # last value applied from in_misregistration
        self.outputs['out_i'] = self.out_i
        self.outputs['out_psf_tot'] = self.psf_tot
        self.outputs['out_psf_bfm'] = self.psf_bfm
        self.outputs['out_transmission'] = self.transmission
        self.outputs['out_flux_frac_inside_detector'] = self.flux_frac_inside_ccd

        # The input field is multiplied by (-1)^(x+y), so that its FFT comes out
        # already fftshift-ed (fft_totsize is always even): the focal plane, the
        # pyramid mask and the PSFs are all centered, and no fftshift is needed.
        # The sub-pixel shift phase aligning the pyramid tip with the FFT grid
        # center is included in the same 1D vectors (all these terms are separable)
        self.tlt_f_y, self.tlt_f_x = self.get_tlt_f(fft_sampling, fft_padding)
        # Orthogonal tilt vectors used to generate the tip-tilt modulation path
        self.tilt_x, self.tilt_y = self.get_modulation_tilts(fft_sampling)
        # Focal plane mask (field stop) to limit the WFS field of view
        self.fp_mask = self.get_fp_mask(fft_totsize, self.fp_masking, obsratio=fp_obsratio)
        # Complex phase delay of the pyramid prism and field stop
        self.masked_exp = self.get_masked_exp(self.get_pyr_tlt(fft_sampling, fft_padding))

        self.pup_pyr_tot = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)

        self.ttexp_x = None
        self.ttexp_y = None
        self._u_t = None
        self._u_y = None
        self.ifft_norm = 1.0 / (self.fft_totsize * self.fft_totsize)
        self.ef = self.xp.zeros((fft_sampling, fft_sampling), dtype=self.complex_dtype)

        # Derived classes can disable streams
        self.stream_enable = True

    @staticmethod
    def default_mod_step(mod_amp: float, mod_type: str = 'circular') -> int:
        """
        Default number of modulation steps for a modulation amplitude [lambda/D].

        Circular modulation: 1 point for mod_amp = 0, otherwise a multiple of
        4 points, more than 2*pi*mod_amp (e.g. 8 points for mod_amp = 1).
        Linear modulation: 1 point for mod_amp = 0, otherwise an odd number of
        points (one in [0, 0]), about one per lambda/D along the path.
        """
        if mod_type == 'circular':
            return int(max(1, round(mod_amp * 2) * 4))
        return int(round(mod_amp) * 2 + 1)

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil.'
                                                  'Contains complex amplitude and phase information '
                                                  'that will be modulated and propagated through the pyramid.'),
                'in_misregistration': InputDesc(BaseValue, 'Mis-registration [pupil shift x, pupil shift y '
                                                           '(pixels, as pup_shifts), rotation (deg, as '
                                                           'rotAnglePhInDeg)], applied when it changes: '
                                                           'each change builds new interpolators and captures '
                                                           'the CUDA graph again, so it is meant for occasional '
                                                           'changes, not for a value that changes at every step. '
                                                           'Keep its source on the CPU to avoid a GPU '
                                                           'synchronization at each step (optional).')}
    
    @classmethod
    def output_names(cls):
        return {'out_i': OutputDesc(Intensity, 'Output intensity on detector CCD (shape: final_ccd_side × final_ccd_side)'),
                'out_psf_tot': OutputDesc(BaseValue, 'Total PSF after focal plane mask application (shape: fft_totsize × fft_totsize)'),
                'out_psf_bfm': OutputDesc(BaseValue, 'PSF before focal plane mask (shape: fft_totsize × fft_totsize)'),
                'out_transmission': OutputDesc(BaseValue, 'Scalar value representing total flux transmission through the system'),
                'out_flux_frac_inside_detector': OutputDesc(BaseValue, 'Fraction of total flux that falls inside the CCD area')}

    def calc_pyr_geometry(self,
        DpupPix,                # number of pixels of input phase array
        pixel_pitch,            # pixel sampling [m] of DpupPix
        lambda_,                # working lambda of the sensor [nm]
        FoV,                    # requested FoV in arcsec
        pup_diam,               # pupil diameter in subapertures
        ccd_side,               # requested output ccd side, in pixels
        fov_errinf=0.1,         # accepted error in reducing FoV, default = 0.1 (-10%)
        fov_errsup=0.5,         # accepted error in enlarging FoV, default = 0.5 (+50%)
        pup_dist=None,          # pupil distance in subapertures, optional
        pup_margin=2,           # zone of respect around pupils for margins, optional, default=2px
        fft_res=3.0,            # requested minimum PSF sampling, 1.0 = 1 pixel / PSF, default=3.0
        min_pup_dist=None,
    ):
        # Calculate pup_distance if not given, using the pup_margin
        if pup_dist is None:
            pup_dist = pup_diam + pup_margin * 2

        if min_pup_dist is None:
            min_pup_dist = pup_diam + pup_margin * 2

        if pup_dist < min_pup_dist:
            raise ValueError(f"Error: pup_dist (px) = {pup_dist} is"
                             f"not enough to hold the pupil geometry."
                             f" Minimum allowed distance is {min_pup_dist}")

        min_ccd_side = pup_dist + pup_diam + pup_margin * 2
        if ccd_side < min_ccd_side:
            raise ValueError(f"Error: ccd_side (px) = {ccd_side} is"
                             f" not enough to hold the pupil geometry."
                             f" Minimum allowed side is {min_ccd_side}")

        internal_ccd_side = int(self.xp.around(fft_res * pup_diam / 2) * 2)

        # Theoretical fft resolution, and minimum fft resolution to hold the pupil geometry
        fft_res = internal_ccd_side / float(pup_diam)
        fft_res_min = (pup_dist + pup_diam) / pup_diam * 1.1
        if fft_res < fft_res_min:
            fft_res = fft_res_min

        result = calc_geometry(
            DpupPix,
            pixel_pitch,
            lambda_,
            FoV,
            fov_errinf=fov_errinf,
            fov_errsup=fov_errsup,
            fft_res=fft_res,
        )

        fft_res = result['fft_res']

        # Recompute toccd_side from the final fft_res (not the pre-bump internal_ccd_side
        # above): otherwise, whenever fft_res_min increased fft_res, the pupils come out
        # smaller than pup_diam pixels after the toccd() rebin below.
        toccd_side = int(self.xp.around(fft_res * pup_diam / 2) * 2)

        result.update(
            {
            'tilt_scale': fft_res / ((pup_dist / float(pup_diam)) / 2.0),
            'toccd_side': toccd_side,
            'final_ccd_side': ccd_side
            }
        )

        return result

    def _face_planes(self):
        """
        Gradients and offsets of the four face planes.

        Face j is the plane tip + offset[j] + grad[j, 0] * u + grad[j, 1] * v,
        with (u, v) [pixels] relative to the tip, u along columns and v along
        rows, in units of tilt pixels. Face j has slopes pyr_tlt_coeff[:, j]
        around the tip. The roof (pyr_roof_ld) lowers faces 0 and 3 by
        delta/2 and raises faces 1 and 2 by delta/2 (opposite signs for a
        negative roof).

        Returns
        -------
        grad : ndarray, shape (4, 2)
        offset : ndarray, shape (4,)
        """
        if self.pyr_tlt_coeff is not None:
            k = cpuArray(self.pyr_tlt_coeff).astype(np.float64)
        else:
            k = np.ones((2, 4))

        # Signs of (u, v) in the quadrant of each face, for the nominal pyramid
        sign_u = np.array((-1, 1, -1, 1))
        sign_v = np.array((-1, -1, 1, 1))
        grad = np.stack((-k[0] * sign_u, -k[1] * sign_v), axis=1)

        # Roof: with nominal slopes, the two lowered faces meet along the
        # diagonal segment u + v = 0 (faces 0 and 3, delta > 0) or u = v
        # (faces 1 and 2, delta < 0), with |u| <= |delta|/2: length
        # L = sqrt(2) * |delta|.
        roof_len = self.pyr_roof_ld * self.fft_res        # [pixels], signed
        delta = roof_len / np.sqrt(2)
        offset = np.array((-delta / 2, delta / 2, delta / 2, -delta / 2))
        return grad, offset

    def _tip_coords(self, p, c):
        """Pixel centres relative to the tip [pixels], as (u, v) broadcastable to (2A, 2A)"""
        A = int(round((p + c) / 2.0))
        coord = np.arange(2 * A) - (A - 0.5)
        return coord[None, :], coord[:, None]

    @staticmethod
    def _tip_height(p, c):
        """
        Height of the face planes at the tip [tilt pixels]: the value of the
        former quadrant formula (x + y counted from the outer corner of each
        quadrant), so that the nominal pyramid is unchanged.
        """
        return 2 * int(round((p + c) / 2.0)) - 1

    def get_pyr_faces(self, p, c):
        """
        Planes of the four pyramid faces, before the envelope.

        Returns an array of shape (4, 2A, 2A), A = round((p + c) / 2), with
        the phase delay of each face plane in units of tilt pixels (as
        get_pyr_tlt() before the division by tilt_scale). The tip lies at
        the corner between the four central pixels. See _face_planes().
        """
        u, v = self._tip_coords(p, c)
        grad, offset = self._face_planes()
        tip = self._tip_height(p, c)
        faces = np.empty((4, v.shape[0], u.shape[1]))
        for j in range(4):
            faces[j] = tip + offset[j] + grad[j, 0] * u + grad[j, 1] * v
        return faces

    def _tip_points(self):
        """
        Vertices of the pyramid surface [pixels, (u, v)]: the tip, or the two
        ends of the roof, where the planes of the roof faces meet the plane
        of each of the other two faces.
        """
        if self.pyr_roof_ld == 0:
            return [np.zeros(2)]
        grad, offset = self._face_planes()
        a, b = (0, 3) if self.pyr_roof_ld > 0 else (1, 2)
        points = []
        for o in sorted({0, 1, 2, 3} - {a, b}):
            # face a = face b and face a = face o: (g_a - g_i) . r = off_i - off_a
            m = np.array([grad[a] - grad[b], grad[a] - grad[o]])
            rhs = np.array([offset[b] - offset[a], offset[o] - offset[a]])
            points.append(np.linalg.solve(m, rhs))
        return points

    def _tip_distance(self, u, v):
        """Distance [pixels] from the tip, or from the roof segment"""
        points = self._tip_points()
        if len(points) == 1:
            return np.sqrt(u ** 2 + v ** 2)
        p0, p1 = points
        d = p1 - p0
        # Position along the segment, clipped to its ends
        t = np.clip(((u - p0[0]) * d[0] + (v - p0[1]) * d[1]) / (d @ d), 0, 1)
        return np.sqrt((u - p0[0] - t * d[0]) ** 2 + (v - p0[1] - t * d[1]) ** 2)

    def _edge_distance(self, faces):
        """
        Distance [pixels] of each pixel from the nearest edge of its face.

        Each face is the convex region where its plane is the lowest, so the
        distance from its border is the minimum, over the other faces i, of
        the distance from the line where plane i equals the own plane:
        (plane_i - plane_own) / |grad_i - grad_own|. Pixels on a roof, where
        two planes are equal, get distance 0 whichever face owns them.
        """
        grad, _ = self._face_planes()
        own = faces.argmin(axis=0)
        envelope = faces.min(axis=0)
        dist = np.full(envelope.shape, np.inf)
        for j in range(4):
            sel = own == j
            for i in range(4):
                if i != j:
                    norm = np.sqrt(np.sum((grad[i] - grad[j]) ** 2))
                    dist[sel] = np.minimum(dist[sel], (faces[i][sel] - envelope[sel]) / norm)
        return dist

    def get_pyr_tlt(self, p, c):
        """
        Phase delay of the pyramid [tilt pixels / tilt_scale], in three steps:

        1. geometry: envelope of the four face planes (each point belongs to
           the lowest one), with the flat top of pyr_tip_maya_ld;
        2. random manufacturing defects along the edges (pyr_edge_def_ld)
           and around the tip or the roof (pyr_tip_def_ld), measured from
           the actual edges and tip of step 1;
        3. finite support (pyr_max_side_ld).

        Everything is computed on the CPU, as get_fp_mask(), to keep the
        full-frame float64 temporaries out of the GPU memory pool.
        """
        faces = self.get_pyr_faces(p, c)
        pyr_tlt = faces.min(axis=0)
        u, v = self._tip_coords(p, c)

        # 1. Flat top ("maya" tip): the surface is cut r below its maximum,
        # which lies at the tip (or along the roof). With nominal slopes the
        # flat top is a square with its corners on the edges, at r from the
        # tip, i.e. with diagonal pyr_tip_maya_ld.
        if self.pyr_tip_maya_ld > 0:
            r = self.pyr_tip_maya_ld * self.fft_res / 2
            grad, offset = self._face_planes()
            # Height of the surface at the tip (the same at both roof ends)
            p0 = self._tip_points()[0]
            top = np.min(self._tip_height(p, c) + offset + grad @ p0)
            n_flat = int(np.sum(pyr_tlt > top - r))
            pyr_tlt = np.minimum(pyr_tlt, top - r)
            self.logger.info(f'get_pyr_tlt: {n_flat} pixels in the flat top of the pyramid'
                             ' (pyramid imperfect tip, maya)')

        # 2. Random defects, with values up to the maximum of the surface
        if self.pyr_edge_def_ld > 0:
            idx_edge = np.nonzero(self._edge_distance(faces) <= self.pyr_edge_def_ld * self.fft_res / 2)
            pyr_tlt[idx_edge] = pyr_tlt.max() * np.random.rand(len(idx_edge[0]))
            self.logger.info(f'get_pyr_tlt: {len(idx_edge[0])} pixels set to random values'
                             ' to consider pyramid imperfect edges')

        if self.pyr_tip_def_ld > 0:
            idx_tip = np.nonzero(self._tip_distance(u, v) <= self.pyr_tip_def_ld * self.fft_res / 2)
            pyr_tlt[idx_tip] = pyr_tlt.max() * np.random.rand(len(idx_tip[0]))
            self.logger.info(f'get_pyr_tlt: {len(idx_tip[0])} pixels set to random values'
                             ' to consider pyramid imperfect tip')

        # 3. Limit the pyramid support to a finite radial extent from the optical axis
        if self.pyr_max_side_ld > 0:
            max_side = self.pyr_max_side_ld * self.fft_res / 2
            idx_max = np.nonzero(np.sqrt(u ** 2 + v ** 2) > max_side)
            if len(idx_max[0]) > 0:
                pyr_tlt[idx_max] = 0.0
                self.logger.info(
                    f'get_pyr_tlt: {len(idx_max[0])} pixels outside the support radius '
                    f'({self.pyr_max_side_ld} lambda/D) set to 0'
                )

        return self.to_xp(pyr_tlt, dtype=self.dtype) / self.tilt_scale

    def get_tlt_f(self, p, c):
        """
        Separable phase applied to the input field, returned as (y, x) vectors.
        It combines a half-pixel shift (in the FFT space of size p+c) that aligns
        the pyramid tip with the FFT grid center, and the (-1)^n factor that
        centers the focal plane.
        """
        n = np.arange(int(p))
        v = self.to_xp(np.exp(1j * np.pi * n * (1 - 1 / (p + c))), dtype=self.complex_dtype)
        return v, v

    def get_fp_mask(self, totsize, mask_ratio, obsratio=0):
        # Computed on the CPU: on the GPU, its full-frame float64 temporaries
        # would raise the memory pool size well above the mask itself
        return self.to_xp(make_mask(totsize, diaratio=mask_ratio, obsratio=obsratio, xp=np))

    def get_masked_exp(self, pyr_tlt):
        """
        Centered complex focal plane mask: phase delay of the pyramid prism
        times the field stop. It includes a phase ramp that rolls the output
        pupil plane by fft_padding/2 pixels in both axes, to re-center the
        four pupils within the final array.
        """
        iu = self.xp.array(1j, dtype=self.complex_dtype)  # complex unit
        k = np.arange(self.fft_totsize)
        ramp = np.exp(-2j * np.pi * k * (self.fft_padding // 2) / self.fft_totsize)
        ramp = self.to_xp(ramp, dtype=self.complex_dtype)
        masked_exp = self.xp.exp(-2 * self.xp.pi * iu * pyr_tlt, dtype=self.complex_dtype)
        masked_exp *= self.fp_mask
        masked_exp *= ramp[:, None]
        masked_exp *= ramp[None, :]
        return masked_exp

    def get_modulation_tilts(self, p):
        """Modulation tilts along x and y, as 1D vectors (the tilt maps are separable)"""
        p = int(p)
        xx, yy = make_xy(p, p // 2, xp=np)
        scale = np.pi / ((xx.max() - xx.min()) / 2)
        return xx[0, :] * scale, yy[:, 0] * scale

    def cache_ttexp(self):
        """
        Cache the tip/tilt modulation phasors and the flux weights of the
        modulation steps. The modulation tilt a * tilt_x + b * tilt_y is
        separable, so its phasors are stored as 1D vectors along x and y.
        """
        # Determine number of rotation variants needed
        if self.mod_type == 'alternating':
            n_rotations = 2  # Both vertical and horizontal
        else:
            n_rotations = 1  # Only one orientation

        # Tilt coefficients along x (a) and y (b) for each rotation and modulation step
        a = np.zeros((n_rotations, self.mod_steps))
        b = np.zeros((n_rotations, self.mod_steps))

        # MODULATION MODE (extended source case moved to a different class):
        # Handle different modulation types
        if self.mod_type == 'circular':
            # CIRCULAR MODULATION MODE: Standard pyramid modulation
            for tt in range(self.mod_steps):
                angle = 2 * np.pi * (tt / self.mod_steps)
                a[0, tt] = self.mod_amp * np.sin(angle)
                b[0, tt] = self.mod_amp * np.cos(angle)

            # Equal flux for all modulation steps
            self.flux_factor_vector = self.xp.ones(self.mod_steps, dtype=self.dtype)
        elif self.mod_type in ['vertical', 'horizontal', 'alternating']:
            # LINEAR MODULATION: Generate vertical case first
            for rotation_idx in range(n_rotations):
                for tt in range(self.mod_steps):
                    # Linear modulation from -mod_amp to +mod_amp
                    tilt_value = self.mod_amp * (2 * tt / (self.mod_steps - 1) - 1)

                    if self.mod_type == 'horizontal' or (self.mod_type == 'alternating' and rotation_idx == 1):
                        # Horizontal modulation uses tilt_x
                        a[rotation_idx, tt] = tilt_value
                    else:
                        # Vertical modulation uses tilt_y
                        b[rotation_idx, tt] = tilt_value

            # Calculate flux correction for linear modulation
            # Use integrated intensity over each step interval
            self.flux_factor_vector = self.xp.zeros(self.mod_steps, dtype=self.dtype)

            if self.mod_steps == 1:
                self.flux_factor_vector[0] = 1.0
            else:
                for tt in range(self.mod_steps):
                    # For linear modulation, use the average intensity over the step interval
                    # This accounts for the continuous interval each discrete point represents

                    # Step boundaries in tilt space
                    if tt == 0:
                        # First point: from -mod_amp to midpoint with next
                        tilt_mid = self.mod_amp * (2 * 0.5 / (self.mod_steps - 1) - 1) # this is > - self.mod_amp
                        avg_tilt = (-self.mod_amp + tilt_mid) / 2 # this means that normalized_angle cannot be - pi/2
                    elif tt == self.mod_steps - 1:
                        # Last point: from midpoint with previous to +mod_amp
                        tilt_mid = self.mod_amp * (2 * (tt - 0.5) / (self.mod_steps - 1) - 1) # this is < self.mod_amp
                        avg_tilt = (tilt_mid + self.mod_amp) / 2 # this means that normalized_angle cannot be pi/2
                    else:
                        # Middle points: average over symmetric interval
                        avg_tilt = self.mod_amp * (2 * tt / (self.mod_steps - 1) - 1)

                    # Convert to flux factor - INVERTED: more weight at the edges
                    normalized_angle = self.xp.abs(avg_tilt) * self.xp.pi / (2 * self.mod_amp)
                    # Use 1/cos(angle) to compensate for intensity loss at large tilts
                    self.flux_factor_vector[tt] = 1.0 / self.xp.cos(normalized_angle)

        if self.mod_amp > 0.0:
            self.logger.info(f'Cached circular modulation with {self.mod_steps} steps, '
                f'amplitude: {self.mod_amp:.2f}')
        else:
            self.logger.info('Running unmodulated pyramid')

        # Shape: (n_rotations, mod_steps, fft_sampling)
        self.ttexp_x = self.to_xp(np.exp(-1j * a[:, :, None] * self.tilt_x), dtype=self.complex_dtype)
        self.ttexp_y = self.to_xp(np.exp(-1j * b[:, :, None] * self.tilt_y), dtype=self.complex_dtype)
        # Input phasors including the constant tlt_f phase
        self._ex = self.ttexp_x * self.tlt_f_x
        self._ey = self.ttexp_y * self.tlt_f_y

        # Common setup for both modes
        self.ffv = self.flux_factor_vector[:, self.xp.newaxis, self.xp.newaxis]
        self.factor = 1.0 / self.xp.sum(self.flux_factor_vector)

        # Buffers of focal_plane_field(). Only their first fft_sampling
        # columns are written, the padding remains zero.
        if self._u_t is None:
            p, n = self.fft_sampling, self.fft_totsize
            self._u_t = self.xp.zeros((p, n), dtype=self.complex_dtype)
            self._u_y = self.xp.zeros((n, n), dtype=self.complex_dtype)

    @property
    def ttexp(self):
        """
        Tip/tilt modulation phasors, shape (n_rotations, mod_steps, fft_sampling, fft_sampling).
        Built on request from the separable vectors, for inspection only.
        """
        return self.ttexp_y[:, :, :, None] * self.ttexp_x[:, :, None, :]

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        self._update_misregistration()
        self.ef_interpolator.interpolate()
        self.ef_interpolator.interpolated_ef().ef_at_lambda(self.wavelength_in_nm, out=self.ef)

        self._update_s0_area()

    def _update_misregistration(self):
        '''Apply a new value of the in_misregistration input, when it changes'''
        misreg = self.local_inputs.get('in_misregistration')
        if misreg is None or misreg.generation_time != self.current_time:
            return
        value = tuple(float(v) for v in cpuArray(misreg.value).ravel())
        if len(value) != 3:
            raise ValueError(f'in_misregistration must have 3 values (shift x, shift y, rotation), got {value}')
        if value == self._misregistration:
            return
        self._misregistration = value
        self.pup_shifts = value[:2]
        self._make_pup_shift_interp()
        self.update_interpolator_parameters(rotAnglePhInDeg=value[2])

    def _make_pup_shift_interp(self):
        '''The interpolator of the pupil shifts (pup_shifts), only if they are not zero'''
        if tuple(float(v) for v in self.pup_shifts) == (0.0, 0.0):
            self._do_pup_shift = False
            return
        # Calculate scaling factor from FFT size to CCD size
        imscale = float(self.fft_totsize) / float(self.toccd_side)
        pup_shiftx = float(self.pup_shifts[0]) * imscale
        pup_shifty = float(self.pup_shifts[1]) * imscale

        # Create the interpolated pupil pyramid array
        if getattr(self, '_pup_pyr_interpolated', None) is None or self._pup_pyr_interpolated is self.pup_pyr_tot:
            self._pup_pyr_interpolated = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)

        self.pup_shift_interp = Interp2D(
            (self.fft_totsize, self.fft_totsize),     # Input shape
            (self.fft_totsize, self.fft_totsize),     # Output shape (same)
            rotInDeg=0,                               # No rotation
            rowShiftInPixels=pup_shifty,              # Y shift
            colShiftInPixels=pup_shiftx,              # X shift
            dtype=self.dtype,
            xp=self.xp
        )
        self._do_pup_shift = True

    def _update_s0_area(self):
        '''Copy S0 * pixel_pitch**2 of the input field to the GPU, when it changes'''
        in_ef = self.local_inputs['in_ef']
        s0_area = in_ef.S0 * in_ef.pixel_pitch ** 2
        if s0_area != self._s0_area_host:
            self._s0_area_host = s0_area
            self._s0_area.fill(s0_area)

    def step_phasors(self):
        '''Input phasors (y, x) of the modulation steps for the current frame'''
        if self.mod_type == 'alternating':
            # Select rotation based on current iteration
            rotation_idx = self.iter[0] % 2
            return self._ey[rotation_idx], self._ex[rotation_idx]
        return self._ey[0], self._ex[0]

    def focal_plane_field(self, ey, ex):
        '''
        Focal plane field of a modulation step: 2D FFT of the input field
        times the (ey, ex) phasors, zero-padded to fft_totsize.
        It is computed as two passes of 1D FFTs, the first one only on the
        fft_sampling non-zero lines of the padded field.
        '''
        p = self.fft_sampling
        # Input field, transposed: one line for each x, along y
        pyr_input(self.ef.T, ex[:, None], ey[None, :], self._u_t[:, :p], xp=self.xp)
        # FFT along y, back to (ky, x) in the first p columns
        u_ky = self.xp.fft.fft(self._u_t, axis=-1)
        self._u_y[:, :p] = u_ky.T
        del u_ky
        # FFT along x
        return self.xp.fft.fft(self._u_y, axis=-1)

    def trigger_code(self):
        ey, ex = self.step_phasors()
        psf_bfm = self.psf_bfm.value
        psf_bfm[:] = 0
        self.pup_pyr_tot[:] = 0

        # One modulation step at a time, to keep a single padded field in memory
        for i in range(0, self.mod_steps):
            # Fourier Transform to propagate the input electric field, with a
            # sub-pixel shift and the tip-tilt of this step, to the (centered) Focal Plane
            u_fp = self.focal_plane_field(ey[i], ex[i])

            # Accumulate the focal plane PSF and apply in place the phase
            # delay of the pyramid and field stop
            pyr_fp_mask(u_fp, self.ffv[i], psf_bfm, self.masked_exp, xp=self.xp)

            # In-place Inverse Fourier Transform to return to the Pupil Plane
            # 'forward' normalization is faster and we normalize correctly later in pyr_abs2_acc()
            pyr_ef = self._scipy_ifft2(u_fp, overwrite_x=True, norm='forward')

            # Calculate intensity and apply weighted accumulation for flux correction
            pyr_abs2_acc(pyr_ef, self.ifft_norm, self.ffv[i], self.pup_pyr_tot, xp=self.xp)
            # Release the FFT output before the next step allocates a new one
            del u_fp, pyr_ef

        # Normalize by the integration time/total modulation weight
        psf_bfm *= self.factor
        # PSF after the focal plane mask
        self.xp.multiply(psf_bfm, self.fp_mask, out=self.psf_tot.value)

        # Calculate the total optical transmission of the system
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) / self.xp.sum(psf_bfm)

        self._pupils_to_ccd()

    def _pupils_to_ccd(self):
        '''
        Normalize the pupils to the photons of the input field and the
        transmission, and rebin them to the CCD (out_i). GPU work only, part of
        trigger_code() (and of its CUDA graph): derived classes that override
        trigger_code() call it at its end.
        '''
        # Photons of the input field: S0 times its masked area (ElectricField.masked_area())
        phot = self._s0_area * self.xp.sum(self.local_inputs['in_ef'].field[0])

        self.pup_pyr_tot *= (phot / self.xp.sum(self.pup_pyr_tot)) * self.transmission.value
#        if phot == 0: slows down?
#            self.logger.warning('total intensity at PYR entrance is zero')

        # Apply pupil shifts using the dedicated interpolator
        # Note: this is a static shift, not a time-varying one as in PASSATA
        if self._do_pup_shift:
            self.pup_shift_interp.interpolate(self.pup_pyr_tot, out=self._pup_pyr_interpolated)
        else:
            # Use the original pupil pyramid array directly
            self._pup_pyr_interpolated = self.pup_pyr_tot

        toccd_shape = (self.toccd_side, self.toccd_side)
        if self.final_ccd_side >= self.toccd_side:
            delta = (self.final_ccd_side - self.toccd_side) // 2
            toccd(self._pup_pyr_interpolated, toccd_shape, xp=self.xp,
                  out=self.out_i.i[delta:delta + self.toccd_side, delta:delta + self.toccd_side])
        else:
            ccd_internal = toccd(self._pup_pyr_interpolated, toccd_shape, xp=self.xp)
            delta = (self.toccd_side - self.final_ccd_side) // 2
            self.out_i.i[:] = ccd_internal[delta:delta + self.final_ccd_side, delta:delta + self.final_ccd_side]
            self.flux_frac_inside_ccd.value[:] = self.xp.sum(self.out_i.i[:])/self.xp.sum(ccd_internal)

    def post_trigger(self):
        super().post_trigger()

        if self.mod_type == 'alternating':
            # Increment iteration counter at the end (not in trigger_code(): the
            # capture of a CUDA graph runs trigger_code() once more)
            self.iter[0] += 1

        self.flux_frac_inside_ccd.generation_time = self.current_time
        self.out_i.generation_time = self.current_time
        self.psf_tot.generation_time = self.current_time
        self.psf_bfm.generation_time = self.current_time
        self.transmission.generation_time = self.current_time

    def setup(self):
        super().setup()

        self.cache_ttexp()

        # Get input electric field
        in_ef = self.local_inputs['in_ef']

        self.ef_interpolator = EFInterpolator(
            in_ef = in_ef,
            out_shape=(self.fft_sampling, self.fft_sampling),
            rotAnglePhInDeg=self.rotAnglePhInDeg,
            xShiftPhInPixel=self.xShiftPhInPixel,
            yShiftPhInPixel=self.yShiftPhInPixel,
            magnification=self.magnification,
            mask_threshold=self._mask_threshold,
            force_extrapolation=self.force_extrapolation,
            use_out_ef_cache=True,
            target_device_idx=self.target_device_idx,
            precision=self.precision
        )

        # Separate interpolator for pup_shifts if needed
        self._make_pup_shift_interp()

        self._update_s0_area()
        if self.stream_enable:
            super().build_stream()

    def update_interpolator_parameters(self, xShiftPhInPixel=None, yShiftPhInPixel=None,
                                       rotAnglePhInDeg=None, magnification=None):
        '''
        Change the misregistration parameters of the input field interpolation.

        The interpolation is currently done in prepare_trigger(), outside the
        CUDA graph, so the graph would not need to be captured again. It is
        invalidated anyway as a safeguard, in case the interpolation is moved
        into trigger_code(): parameter changes are rare, and the cost is a
        single new capture. Always use this method instead of calling
        ef_interpolator.update_parameters() directly.

        Parameters are the same as EFInterpolator.update_parameters();
        the ones set to None are left unchanged.
        '''
        self.ef_interpolator.update_parameters(xShiftPhInPixel=xShiftPhInPixel,
                                               yShiftPhInPixel=yShiftPhInPixel,
                                               rotAnglePhInDeg=rotAnglePhInDeg,
                                               magnification=magnification)
        self.invalidate_graph()
