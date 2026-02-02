from specula import fuse
from specula.lib.extrapolation_2d import EFInterpolator
from specula.lib.interp2d import Interp2D

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_xy import make_xy
from specula.lib.make_mask import make_mask
from specula.lib.toccd import toccd
from specula.lib.calc_geometry import calc_geometry
from specula.lib.utils import make_subpixel_shift_phase

@fuse(kernel_name='pyr1_fused')
def pyr1_fused(u_fp, ffv, fpsf, masked_exp, xp):
    psf = xp.real(u_fp * xp.conj(u_fp))
    fpsf += psf * ffv
    u_fp_pyr = u_fp * masked_exp
    return u_fp_pyr


@fuse(kernel_name='pyr1_abs2')
def pyr1_abs2(v, norm, ffv, xp):
    v_norm = v * norm
    return xp.real(v_norm * xp.conj(v_norm)) * ffv


class ModulatedPyramid(BaseProcessingObj):
    """
    Pyramid wavefront sensor with tip-tilt modulation.
    
    This class implements a modulated pyramid WFS that works with point sources or
    can be used as a base class for extended source implementations. The modulation
    can be circular (standard), or linear (vertical, horizontal, alternating).
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters object containing pixel_pupil and pixel_pitch
    wavelengthInNm : float
        Working wavelength in nanometers
    fov : float
        Field of view in arcseconds (a field stop may be applied in the focal plane to limit FoV)
    pup_diam : int
        Pupil diameter in pixels
    output_resolution : int
        Output CCD side length in pixels
    mod_amp : float, optional
        Modulation amplitude in lambda/D units (default: 3.0)
    mod_step : int, optional
        Number of modulation steps. If None, automatically calculated based on
        mod_amp and mod_type (default: None)
    mod_type : str, optional
        Modulation type: 'circular', 'vertical', 'horizontal', or 'alternating'
        (default: 'circular')
    fov_errinf : float, optional
        Accepted error in reducing FoV (default: 0.1, i.e., -10%)
    fov_errsup : float, optional
        Accepted error in enlarging FoV (default: 2.0, i.e., +100%)
    pup_dist : int, optional
        Pupil distance in pixels. If None, calculated from pup_diam and pup_margin
    pup_margin : int, optional
        Margin around pupils in pixels (default: 2)
    fft_res : float, optional
        Minimum PSF sampling (default: 3.0, i.e., 3 pixels per PSF FWHM i.e. lambda/D)
    fp_obs : float, optional
        Focal plane central obstruction diameter in pixels (default: None)
    pup_shifts : tuple, optional
        Static pupil shifts in pixels (x, y) (default: (0.0, 0.0))
    pyr_tlt_coeff : float, optional
        Pyramid tilt coefficients for custom face geometry (default: None)
        WARNING: not implemented/tested yet
    pyr_edge_def_ld : float, optional
        Edge defect size in lambda/D units (default: 0.0)
    pyr_tip_def_ld : float, optional
        Tip defect size in lambda/D units (default: 0.0)
    pyr_tip_maya_ld : float, optional
        Maya Pyramid (i.e. flat tip) defect size in lambda/D units (default: 0.0)
    min_pup_dist : float, optional
        Minimum pupil distance constraint (default: None)
    rotAnglePhInDeg : float, optional
        Rotation angle of input phase in degrees (default: 0.0)
    xShiftPhInPixel : float, optional
        X shift of input phase in pixels (default: 0.0)
    yShiftPhInPixel : float, optional
        Y shift of input phase in pixels (default: 0.0)
    target_device_idx : int, optional
        GPU device index (default: None, uses default device, -1 for CPU)
    precision : int, optional
        Numerical precision: 32 (1) or 64 (0) bits (default: None, uses system default)
    
    Inputs
    ------
    in_ef : ElectricField
        Input electric field from the telescope pupil. Contains complex amplitude
        and phase information that will be modulated and propagated through the pyramid.   
    
    Outputs
    -------
    out_i : Intensity
        Output intensity on detector CCD (shape: final_ccd_side × final_ccd_side)
    out_psf_tot : BaseValue
        Total PSF after focal plane mask application (shape: fft_totsize × fft_totsize)
    out_psf_bfm : BaseValue
        PSF before focal plane mask (shape: fft_totsize × fft_totsize)
    out_transmission : BaseValue
        Scalar value representing total flux transmission through the system

    Notes
    -----
    The modulation types have different characteristics:
    - 'circular': Standard pyramid modulation, uniform flux distribution
    - 'vertical'/'horizontal': Linear modulation along one axis, flux weighted by 1/cos(tilt)
    - 'alternating': Switches between vertical and horizontal on consecutive frames
    - pyr_tip_def_ld and pyr_tip_maya_ld represent different types of pyramid tip imperfections.
    
    For linear modulation, the flux weighting compensates for intensity loss at large tilts,
    ensuring uniform contribution from all modulation positions.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float, # TODO =750,
                 fov: float,            # TODO =2.0,
                 pup_diam: int,         # TODO =30,
                 output_resolution: int,# TODO =80,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 mod_type: str = 'circular',  # 'circular', 'vertical', 'horizontal', 'alternating'
                 fov_errinf: float = 0.1,
                 fov_errsup: float = 2,
                 pup_dist: int = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: float = None,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,
                 yShiftPhInPixel: float = 0.0,
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
        fp_obsratio = fp_obs / (fft_totsize / fft_res) if fp_obs is not None else 0

        self.wavelength_in_nm = wavelengthInNm
        self.fov_res = fov_res
        self.fft_res = fft_res
        self.tilt_scale = tilt_scale
        self.fft_sampling = fft_sampling
        self.fft_padding = fft_padding
        self.fft_totsize = fft_totsize
        self.toccd_side = int(toccd_side)
        self.final_ccd_side = final_ccd_side
        self.pyr_tlt_coeff = pyr_tlt_coeff
        self.pyr_edge_def_ld = pyr_edge_def_ld
        self.pyr_tip_def_ld = pyr_tip_def_ld
        self.pyr_tip_maya_ld = pyr_tip_maya_ld
        self.rotAnglePhInDeg = rotAnglePhInDeg
        self.xShiftPhInPixel = xShiftPhInPixel
        self.yShiftPhInPixel = yShiftPhInPixel
        self.pup_shifts = pup_shifts

        # interpolation settings
        self.pup_shift_interp = None
        self._do_pup_shift = False
        self._pup_pyr_interpolated = None
        self._amplitude_is_binary = None
        self._mask_threshold = 1e-3  # threshold to consider a pixel inside the mask

        # Store modulation type
        valid_mod_types = ['circular', 'vertical', 'horizontal', 'alternating']
        if mod_type not in valid_mod_types:
            raise ValueError(f"mod_type must be one of {valid_mod_types}, got {mod_type}")
        self.mod_type = mod_type
        # Add iteration counter for alternating modulation
        self.iter = self.to_xp([0])

        if mod_step is None:
            if mod_type == 'circular':
                # In the circular case we want to ensure:
                # - 1 point for mod_amp = 0
                # - a multiple of 4 points for mod_amp > 0
                # - more than 2*pi*mod_amp points for mod_amp > 0
                # For example, for mod_amp = 1, we want 8 points
                mod_step = max([1.0, round(mod_amp * 2.)*4.])
            else:
                # In the linear case we want:
                # - 1 point for mod_amp = 0
                # - a point in [0, 0] for mod_amp > 0
                # - more than 2 points for mod_amp > 0
                mod_step = round(mod_amp)*2.+1.0
        elif mod_step == 0:
            raise ValueError('mod_step cannot be zero')
        elif mod_step < 0:
            raise ValueError('mod_step must be a positive integer')
        elif int(mod_step) != mod_step:
            raise ValueError('Modulation step number is not an integer')
        elif mod_step < self.xp.around(2 * self.xp.pi * mod_amp):
            raise ValueError(
                f'Number of modulation steps is too small ({mod_step}), '
                f'it must be at least 2*pi times the modulation amplitude '
                f'({self.xp.around(2 * self.xp.pi * mod_amp)})!'
            )

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

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_i'] = self.out_i
        self.outputs['out_psf_tot'] = self.psf_tot
        self.outputs['out_psf_bfm'] = self.psf_bfm
        self.outputs['out_transmission'] = self.transmission

        self.pyr_tlt = self.get_pyr_tlt(fft_sampling, fft_padding)
        self.tlt_f = self.get_tlt_f(fft_sampling, fft_padding)
        self.tilt_x, self.tilt_y = self.get_modulation_tilts(fft_sampling)
        self.fp_mask = self.get_fp_mask(fft_totsize, self.fp_masking, obsratio=fp_obsratio)

        iu = 1j  # complex unit
        myexp = self.xp.exp(-2 * self.xp.pi * iu * self.pyr_tlt, dtype=self.complex_dtype)
        self.shifted_masked_exp = self.xp.fft.fftshift(myexp * self.fp_mask)

        self.pup_pyr_tot = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)

        self.ttexp = None
        self.ttexp_shape = None
        self.u_tlt = None
        self.roll_array = [self.fft_padding//2, self.fft_padding//2]
        self.roll_axis = [0,1]
        self.ifft_norm = 1.0 / (self.fft_totsize * self.fft_totsize)
        # These two are used in the graph-launched trigger code and we manage them separately
        self.pyr_image = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)
        self.fpsf = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)
        self.ef = self.xp.zeros((fft_sampling, fft_sampling), dtype=self.complex_dtype)

        # Derived classes can disable streams
        self.stream_enable = True

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

        internal_ccd_side = self.xp.around(fft_res * pup_diam / 2) * 2

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

        result.update(
            {
            'tilt_scale': fft_res / ((pup_dist / float(pup_diam)) / 2.0),
            'toccd_side': internal_ccd_side,
            'final_ccd_side': ccd_side
            }
        )

        return result

    def get_pyr_tlt(self, p, c):
        A = int((p + c) // 2)
        pyr_tlt = self.xp.zeros((2 * A, 2 * A), dtype=self.dtype)
        y, x = self.xp.mgrid[0:A,0:A]

        if self.pyr_tlt_coeff is not None:
            raise NotImplementedError('pyr_tlt_coeff is not tested yet')

            k = self.pyr_tlt_coeff

            tlt_basis = y
            tlt_basis -= self.xp.mean(tlt_basis)

            pyr_tlt[0:A, 0:A] = k[0, 0] * tlt_basis + k[1, 0] * tlt_basis.T
            pyr_tlt[A:2*A, 0:A] = k[0, 1] * tlt_basis + k[1, 1] * tlt_basis.T
            pyr_tlt[A:2*A, A:2*A] = k[0, 2] * tlt_basis + k[1, 2] * tlt_basis.T
            pyr_tlt[0:A, A:2*A] = k[0, 3] * tlt_basis + k[1, 3] * tlt_basis.T

            pyr_tlt[0:A, 0:A] -= self.xp.min(pyr_tlt[0:A, 0:A])
            pyr_tlt[A:2*A, 0:A] -= self.xp.min(pyr_tlt[A:2*A, 0:A])
            pyr_tlt[A:2*A, A:2*A] -= self.xp.min(pyr_tlt[A:2*A, A:2*A])
            pyr_tlt[0:A, A:2*A] -= self.xp.min(pyr_tlt[0:A, A:2*A])

        else:
            #pyr_tlt[0:A, 0:A] = tlt_basis + tlt_basis.T
            #pyr_tlt[A:2*A, 0:A] = A - 1 - tlt_basis + tlt_basis.T
            #pyr_tlt[A:2*A, A:2*A] = 2 * A - 2 - tlt_basis - tlt_basis.T
            #pyr_tlt[0:A, A:2*A] = A - 1 + tlt_basis - tlt_basis.T
            pyr_tlt[:A, :A] = x + y
            pyr_tlt[:A, A:] = x[:,::-1] + y
            pyr_tlt[A:, :A] = x + y[::-1]
            pyr_tlt[A:, A:] = x[:,::-1] + y[::-1]

        xx, yy = make_xy(A * 2, A, xp=self.xp)

        # distance from edge
        dx = self.xp.sqrt(xx ** 2)
        dy = self.xp.sqrt(yy ** 2)
        idx_edge = self.xp.where((dx <= self.pyr_edge_def_ld * self.fft_res / 2) | 
                            (dy <= self.pyr_edge_def_ld * self.fft_res / 2))[0]
        if len(idx_edge) > 0:
            pyr_tlt[idx_edge] = self.xp.max(pyr_tlt) * self.xp.random.rand(len(idx_edge[0]))
            print(f'get_pyr_tlt: {len(idx_edge[0])} pixels set to 0 to consider pyramid imperfect edges')

        # distance from tip
        d = self.xp.sqrt(xx ** 2 + yy ** 2)
        idx_tip = self.xp.where(d <= self.pyr_tip_def_ld * self.fft_res / 2)[0]
        if len(idx_tip) > 0:
            pyr_tlt[idx_tip] = self.xp.max(pyr_tlt) * self.xp.random.rand(len(idx_tip[0]))
            print(f'get_pyr_tlt: {len(idx_tip[0])} pixels set to 0 to consider pyramid imperfect tip')

        # distance from tip
        idx_tip_m = self.xp.where(d <= self.pyr_tip_maya_ld * self.fft_res / 2)[0]
        if len(idx_tip_m) > 0:
            pyr_tlt[idx_tip_m] = self.xp.min(pyr_tlt[idx_tip_m])
            print(f'get_pyr_tlt: {len(idx_tip_m[0])} pixels set to 0 to consider pyramid imperfect tip')

        return pyr_tlt / self.tilt_scale

    def get_tlt_f(self, p, c):
        """Generate tilt factor for pyramid de-rotation"""        
        p = int(p)
        # The shift amount is 0.5 pixels in the normalized space of size 2*(p+c)
        shift_amount = (2 * p) / (2 * (p + c))

        tlt_f = make_subpixel_shift_phase(
            shape=2 * p,
            shift_x=shift_amount,
            shift_y=shift_amount,
            xp=self.xp,
            dtype=self.complex_dtype,
            quarter=True,
            zero_sampled=True
        )

        return tlt_f

    def get_fp_mask(self, totsize, mask_ratio, obsratio=0):
        return make_mask(totsize, diaratio=mask_ratio, obsratio=obsratio, xp=self.xp)

    def get_modulation_tilts(self, p):
        p = int(p)
        xx, yy = make_xy(p, p // 2, xp=self.xp)
        xmin = self.xp.min(xx)
        xmax = self.xp.max(xx)
        tilt_x = xx * self.xp.pi / ((xmax - xmin) / 2)
        tilt_y = yy * self.xp.pi / ((xmax - xmin) / 2)
        return tilt_x, tilt_y

    def cache_ttexp(self):
        """Cache tip/tilt exponentials for modulation or extended source"""

        iu = 1j  # complex unit

        # Determine number of rotation variants needed
        if self.mod_type == 'alternating':
            n_rotations = 2  # Both vertical and horizontal
        else:
            n_rotations = 1  # Only one orientation

        # Initialize ttexp array with rotation dimension
        # Shape: (n_rotations, mod_steps, height, width)
        self.ttexp = self.xp.zeros((n_rotations, self.mod_steps, self.tilt_x.shape[0], self.tilt_x.shape[1]),
                                dtype=self.complex_dtype)

        # MODULATION MODE (extended source case moved to a different class):
        # Handle different modulation types
        if self.mod_type == 'circular':
            # CIRCULAR MODULATION MODE: Standard pyramid modulation
            for tt in range(self.mod_steps):
                angle = 2 * self.xp.pi * (tt / self.mod_steps)
                pup_tt = (self.mod_amp * self.xp.sin(angle) * self.tilt_x +
                        self.mod_amp * self.xp.cos(angle) * self.tilt_y)

                self.ttexp[0, tt, :, :] = self.xp.exp(-iu * pup_tt, dtype=self.complex_dtype)

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
                        pup_tt = tilt_value * self.tilt_x
                    else:
                        # Vertical modulation uses tilt_y
                        pup_tt = tilt_value * self.tilt_y

                    self.ttexp[rotation_idx, tt, :, :] = self.xp.exp(-iu * pup_tt, dtype=self.complex_dtype)

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

        print(f'Cached circular modulation with {self.mod_steps} steps, '
            f'amplitude: {self.mod_amp:.2f}')

        # Common setup for both modes
        self.ffv = self.flux_factor_vector[:, self.xp.newaxis, self.xp.newaxis]
        self.factor = 1.0 / self.xp.sum(self.flux_factor_vector)
        self.ttexp_shape = self.ttexp.shape[1:]

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        self.ef_interpolator.interpolate()
        self.ef_interpolator.interpolated_ef().ef_at_lambda(self.wavelength_in_nm, out=self.ef)

    def trigger_code(self):
        # Select rotation based on current iteration for alternating modulation
        rotation_idx = self.iter[0] % 2

        # Select the appropriate ttexp slice (no rotation needed!)
        ttexp_current = self.ttexp[rotation_idx]

        u_tlt_const = self.ef * self.tlt_f
        tmp = u_tlt_const[self.xp.newaxis, :, :] * ttexp_current
        self.u_tlt[:, 0:self.ttexp_shape[1], 0:self.ttexp_shape[2]] = tmp
        self.pyr_image *=0
        self.fpsf *=0

        for i in range(0, self.mod_steps):
            u_fp = self.xp.fft.fft2(self.u_tlt[i], axes=(-2, -1))
            u_fp_pyr = pyr1_fused(u_fp, self.ffv[i], self.fpsf, self.shifted_masked_exp, xp=self.xp)

            # 'forward' normalization is faster and we normalize correctly later in pyr1_abs2()
            pyr_ef = self.xp.fft.ifft2(u_fp_pyr, axes=(-2, -1), norm='forward')
            self.pyr_image += pyr1_abs2(pyr_ef, self.ifft_norm , self.ffv[i], xp=self.xp)

        self.psf_bfm.value[:] = self.xp.fft.fftshift(self.fpsf)
        self.psf_tot.value[:] = self.psf_bfm.value * self.fp_mask
        self.pup_pyr_tot[:] = self.xp.roll(self.pyr_image, self.roll_array, self.roll_axis )
        self.psf_tot.value *= self.factor
        self.psf_bfm.value *= self.factor
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) / self.xp.sum(self.psf_bfm.value)

    def post_trigger(self):
        super().post_trigger()

        # Always use the working field (like SH always uses self._wf1)
        in_ef = self.local_inputs['in_ef']
        phot = in_ef.S0 * in_ef.masked_area()

        self.pup_pyr_tot *= (phot / self.xp.sum(self.pup_pyr_tot)) * self.transmission.value
#        if phot == 0: slows down?
#            print('WARNING: total intensity at PYR entrance is zero')

        # Apply pupil shifts using the dedicated interpolator
        # Note: this is a static shift, not a time-varying one as in PASSATA
        if self._do_pup_shift:
            self.pup_shift_interp.interpolate(self.pup_pyr_tot, out=self._pup_pyr_interpolated)
        else:
            # Use the original pupil pyramid array directly
            self._pup_pyr_interpolated = self.pup_pyr_tot

        ccd_internal = toccd(self._pup_pyr_interpolated, (self.toccd_side, self.toccd_side), xp=self.xp)

        if self.final_ccd_side > self.toccd_side:
            delta = (self.final_ccd_side - self.toccd_side) // 2
            self.out_i.i[delta:delta + ccd_internal.shape[0], delta:delta + ccd_internal.shape[1]] = ccd_internal
        elif self.final_ccd_side < self.toccd_side:
            delta = (self.toccd_side - self.final_ccd_side) // 2
            self.out_i.i[:] = ccd_internal[delta:delta + self.final_ccd_side, delta:delta + self.final_ccd_side]
        else:
            self.out_i.i[:] = ccd_internal

        self.out_i.generation_time = self.current_time
        self.psf_tot.generation_time = self.current_time
        self.psf_bfm.generation_time = self.current_time
        self.transmission.generation_time = self.current_time

        if self.mod_type == 'alternating':
            # Increment iteration counter at the end
            self.iter[0] += 1

    def setup(self):
        super().setup()

        self.u_tlt = self.xp.zeros((self.mod_steps, self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)
        self.cache_ttexp()

        # Get input electric field
        in_ef = self.local_inputs['in_ef']

        self.ef_interpolator = EFInterpolator(
            in_ef = in_ef,
            out_shape=(self.fft_sampling, self.fft_sampling),
            rotAnglePhInDeg=self.rotAnglePhInDeg,
            xShiftPhInPixel=self.xShiftPhInPixel,
            yShiftPhInPixel=self.yShiftPhInPixel,
            mask_threshold=self._mask_threshold,
            target_device_idx=self.target_device_idx,
            precision=self.precision,
        )

        # Create separate interpolator for pup_shifts if needed
        if self.pup_shifts != (0.0, 0.0):
            # Calculate scaling factor from FFT size to CCD size
            imscale = float(self.fft_totsize) / float(self.toccd_side)
            pup_shiftx = float(self.pup_shifts[0]) * imscale
            pup_shifty = float(self.pup_shifts[1]) * imscale

            # Create the interpolated pupil pyramid array
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
        else:
            self._do_pup_shift = False

        if self.stream_enable:
            super().build_stream()
