from specula import fuse
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
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
    def cache_ttexp(self):
        # Cache Zernike modes for tip, tilt, focus (static for all frames)
        zg = ZernikeGenerator(self.fft_sampling, xp=self.xp, dtype=self.dtype)
        ext_xtilt = zg.getZernike(2)  # tip
        ext_ytilt = zg.getZernike(3)  # tilt
        ext_focus = zg.getZernike(4)  # focus
        self.ext_ttf = self.xp.stack([ext_xtilt, ext_ytilt, ext_focus], axis=0)

        # Set flux factor vector from source (will be updated in trigger if PSF changes)
        coeff_flux  = self.ext_source_coeff.value[:, 3]
        self.flux_factor_vector = self.to_xp(coeff_flux)

        # Clean up very small flux values
        max_flux = self.xp.max(self.xp.abs(self.flux_factor_vector))
        threshold = max_flux * 1e-5
        small_idx = self.xp.abs(self.flux_factor_vector) < threshold
        self.flux_factor_vector[small_idx] = 0.0

        # Cache constant for u_tlt if ef and tlt_f are static
        self.u_tlt_const = self.ef * self.tlt_f

        # Set ttexp_shape for trigger_code
        self.ttexp_shape = (0, self.tilt_x.shape[0], self.tilt_x.shape[1])
        self.ffv = None
        self.factor = 1.0 / self.xp.sum(self.flux_factor_vector)

    def trigger_code(self):
        iu = 1j  # complex unit

        # Get extended source coefficients for current frame
        coeff_ttf = self.ext_source_coeff.value[:,:3]

        # Reset output arrays for this frame
        self.pyr_image *= 0
        self.fpsf *= 0

        u_tlt_const = self.ef * self.tlt_f
        u_tlt_i = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.complex_dtype)

        for i in range(self.mod_steps):
            # Compute pupil phase for each extended source point
            # coeff_ttf[i] shape: (3,)
            # self.ext_ttf shape: (3, N, N)
            pup_phase = self.xp.tensordot(coeff_ttf[i], self.ext_ttf, axes=([0], [0]))
            # pup_phase shape: (N, N)
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
