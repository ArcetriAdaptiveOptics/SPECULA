from specula import fuse, RAD2ASEC
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.zernike_generator import ZernikeGenerator


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


class ExtSourcePyramid(ModulatedPyramid):
    """
    Pyramid wavefront sensor for extended sources.
    This version computes on the fly the pupil phase for each extended source point
    to reduce memory usage compared to ModulatedPyramid with precomputed ttexp array.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 pup_diam: int,
                 output_resolution: int,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 mod_type: str = 'circular',
                 fov_errinf: float = 0.5,
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
        super().__init__(
            simul_params=simul_params,
            wavelengthInNm=wavelengthInNm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_step=mod_step,
            mod_type=mod_type,
            fov_errinf=fov_errinf,
            fov_errsup=fov_errsup,
            pup_dist=pup_dist,
            pup_margin=pup_margin,
            fft_res=fft_res,
            fp_obs=fp_obs,
            pup_shifts=pup_shifts,
            pyr_tlt_coeff=pyr_tlt_coeff,
            pyr_edge_def_ld=pyr_edge_def_ld,
            pyr_tip_def_ld=pyr_tip_def_ld,
            pyr_tip_maya_ld=pyr_tip_maya_ld,
            min_pup_dist=min_pup_dist,
            rotAnglePhInDeg=rotAnglePhInDeg,
            xShiftPhInPixel=xShiftPhInPixel,
            yShiftPhInPixel=yShiftPhInPixel,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.ffv = None
        self.ext_ttf = None
        self.ext_source_coeff = None

        # Add dedicated input for extended source coefficients
        self.inputs['ext_source_coeff'] = InputValue(type=BaseValue)


    def cache_ttexp(self):
        # set ext_source_coeff if not already set
        if self.ext_source_coeff is None:
            self.ext_source_coeff = self.local_inputs['ext_source_coeff']
            # Update modulation steps to match source points
            self.mod_steps = int(self.ext_source_coeff.value.shape[0])
            print(f'Setting up extended source with {self.mod_steps} points')

            # Cache Zernike modes for tip, tilt, focus (static for all frames)
            zg = ZernikeGenerator(self.fft_sampling, xp=self.xp, dtype=self.dtype)
            ext_xtilt = zg.getZernike(2)  # tip
            ext_ytilt = zg.getZernike(3)  # tilt
            ext_focus = zg.getZernike(4)  # focus
            self.ext_ttf = self.xp.stack([ext_xtilt, ext_ytilt, ext_focus], axis=0)

            # Set ttexp_shape for trigger_code
            self.ttexp_shape = (0, self.tilt_x.shape[0], self.tilt_x.shape[1])

        # Set flux factor vector from source (will be updated in trigger if PSF changes)
        coeff_flux  = self.ext_source_coeff.value[:, 3]
        self.flux_factor_vector = self.to_xp(coeff_flux)

        # Clean up very small flux values
        max_flux = self.xp.max(self.xp.abs(self.flux_factor_vector))
        threshold = max_flux * 1e-5
        small_idx = self.xp.abs(self.flux_factor_vector) < threshold
        self.flux_factor_vector[small_idx] = 0.0

        self.factor = 1.0 / self.xp.sum(self.flux_factor_vector)


    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Update tt cache in case the source was updated
        if self.ext_source_coeff.generation_time == self.current_time:
            # Source was updated this timestep, refresh ttexp, flux factors and ffv
            self.mod_steps = int(self.ext_source_coeff.value.shape[0])
            self.cache_ttexp()


    def trigger_code(self):
        iu = 1j  # complex unit

        # Get extended source coefficients for current frame
        # convert them from nm to rad
        coeff_ttf = self.ext_source_coeff.value[:,:3]
        # Reset output arrays for this frame
        self.pyr_image *= 0
        self.fpsf *= 0

        u_tlt_const = self.ef * self.tlt_f
        u_tlt_i = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)

        for i in range(self.mod_steps):
            # Invert focus sign
            coeff_with_sign = coeff_ttf[i].copy()
            coeff_with_sign[2] *= -1

            # Compute pupil phase for each extended source point
            pup_phase = self.xp.sum(coeff_with_sign[:, None, None] * self.ext_ttf, axis=0)
            ttexp_i = self.xp.exp(-iu * pup_phase, dtype=self.complex_dtype)

            # Compute u_tlt for this point
            u_tlt_i[0:self.ttexp_shape[1], 0:self.ttexp_shape[2]] = u_tlt_const * ttexp_i

            # ffvi must have same dimensions as fpsf
            ffvi = self.flux_factor_vector[i]

            # FFT and PSF calculation as in ModulatedPyramid
            u_fp = self.xp.fft.fft2(u_tlt_i, axes=(-2, -1))
            u_fp_pyr = pyr1_fused(
                u_fp, ffvi, self.fpsf, self.shifted_masked_exp, xp=self.xp
            )
            pyr_ef = self.xp.fft.ifft2(u_fp_pyr, axes=(-2, -1), norm='forward')
            self.pyr_image += pyr1_abs2(pyr_ef, self.ifft_norm, ffvi, xp=self.xp)

        # Final output assignments
        self.psf_bfm.value[:] = self.xp.fft.fftshift(self.fpsf)
        self.psf_tot.value[:] = self.psf_bfm.value * self.fp_mask
        self.pup_pyr_tot[:] = self.xp.roll(self.pyr_image, self.roll_array, self.roll_axis)
        self.psf_tot.value *= self.factor
        self.psf_bfm.value *= self.factor
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) \
            / self.xp.sum(self.psf_bfm.value)
