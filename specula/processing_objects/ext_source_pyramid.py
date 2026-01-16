from specula import fuse
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
                 max_batch_size: int = 1024,
                 max_flux_ratio_thr: float = 1e-3,
                 cuda_stream_enable: bool = True,
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
        self.valid_idx = None
        # Invert focus sign for phase calculation
        self.ttf_signs = self.xp.array([1.0, 1.0, -1.0], dtype=self.dtype)

        # Max batch size for processing (to be adjusted based on GPU memory)
        self.max_batch_size = max_batch_size

        # CUDA stream enable key, it can be disabled for debugging purposes
        self.stream_enable = cuda_stream_enable

        # Threshold for flux filtering (only if stream disabled)
        self.max_flux_ratio_thr = max_flux_ratio_thr

        if self.stream_enable:
            print('CUDA stream enabled for extended source pyramid processing'
                  ' Ignoring flux thresholding to maintain constant processing load.')

        # Pre-allocated buffers for CUDA graph compatibility (allocated in cache_ttexp)
        self._fpsf_buffer = None
        self._pyr_image_buffer = None
        self._n_chunks = 0

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

        # Clean up very small flux values (only if stream disabled)
        # When stream_enable=True, we need constant n_valid for CUDA graph
        if not self.stream_enable:
            max_flux = self.xp.max(self.xp.abs(self.flux_factor_vector))
            threshold = max_flux * self.max_flux_ratio_thr
            small_idx = self.xp.abs(self.flux_factor_vector) < threshold
            self.flux_factor_vector[small_idx] = 0.0
            print(f'Points with flux below {threshold:.3e} set to zero:'
                  f' {self.xp.sum(small_idx)} out of {self.mod_steps}')

            self.valid_idx = self.xp.where(self.flux_factor_vector > 0.0)[0]
        else:
            # With stream enabled, process all points (no filtering)
            # to keep constant loop iterations for CUDA graph
            print(f'CUDA stream enabled: processing all {self.mod_steps} points')
            self.valid_idx = self.xp.arange(self.mod_steps)

        self._n_chunks = (len(self.valid_idx) + self.max_batch_size - 1) // self.max_batch_size
        self._fpsf_buffer = self.xp.zeros((self._n_chunks, *self.fpsf.shape),
                                          dtype=self.dtype)
        self._pyr_image_buffer = self.xp.zeros((self._n_chunks, *self.pyr_image.shape),
                                               dtype=self.dtype)

        self.factor = 1.0 / self.xp.sum(self.flux_factor_vector)


    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Update tt cache in case the source was updated
        if self.ext_source_coeff.generation_time == self.current_time:
            # Source was updated this timestep, refresh ttexp, flux factors and ffv
            self.mod_steps = int(self.ext_source_coeff.value.shape[0])
            self.cache_ttexp()

        # Reset output arrays for this frame
        self.pyr_image *= 0
        self.fpsf *= 0


    def trigger_code(self):
        iu = 1j  # complex unit

        # Get extended source coefficients for current frame (only valid points)
        coeff_ttf = self.ext_source_coeff.value[self.valid_idx, :3]
        ffv_valid = self.flux_factor_vector[self.valid_idx]
        n_valid = self.valid_idx.shape[0]

        # Process in chunks using enumerate for direct chunk indexing
        for chunk_idx, start_idx in enumerate(range(0, n_valid, self.max_batch_size)):
            end_idx = min(start_idx + self.max_batch_size, n_valid)

            # Get chunk data
            coeff_chunk = coeff_ttf[start_idx:end_idx]
            ffv_chunk = ffv_valid[start_idx:end_idx]

            # Compute pupil phases for this chunk
            pup_phases = self.xp.sum(coeff_chunk[:, :, None, None] \
                                    * self.ttf_signs[None, :, None, None] \
                                    * self.ext_ttf[None, :, :, :],
                                    axis=1)

            # Compute ttexp for this chunk
            ttexp_batch = self.xp.exp(-iu * pup_phases, dtype=self.complex_dtype)

            # Prepare u_tlt_batch for this chunk
            u_tlt_const = self.ef * self.tlt_f
            chunk_size = end_idx - start_idx
            u_tlt_batch = self.xp.zeros((chunk_size, self.fft_totsize, self.fft_totsize),
                                        dtype=self.complex_dtype)
            u_tlt_batch[:, 0:self.ttexp_shape[1], 0:self.ttexp_shape[2]] = \
                u_tlt_const[None, :, :] * ttexp_batch

            # Batch FFT
            u_fp_batch = self.xp.fft.fft2(u_tlt_batch, axes=(-2, -1))

            # Store PSF contribution
            psf_batch = self.xp.real(u_fp_batch * self.xp.conj(u_fp_batch))
            self._fpsf_buffer[chunk_idx] = \
                self.xp.sum(psf_batch * ffv_chunk[:, None, None], axis=0)

            # Apply pyramid mask
            u_fp_pyr_batch = u_fp_batch * self.shifted_masked_exp[None, :, :]

            # Batch inverse FFT
            pyr_ef_batch = self.xp.fft.ifft2(u_fp_pyr_batch, axes=(-2, -1), norm='forward')

            # Store pyramid image contribution
            pyr_ef_norm = pyr_ef_batch * self.ifft_norm
            pyr_images = self.xp.real(pyr_ef_norm * self.xp.conj(pyr_ef_norm))
            self._pyr_image_buffer[chunk_idx] = \
                self.xp.sum(pyr_images * ffv_chunk[:, None, None], axis=0)

        # Final reduction using pre-computed chunk count
        self.fpsf[:] = self.xp.sum(self._fpsf_buffer, axis=0)
        self.pyr_image[:] = self.xp.sum(self._pyr_image_buffer, axis=0)


    def post_trigger(self):
        # Final output assignments (before parent post_trigger)
        self.psf_bfm.value[:] = self.xp.fft.fftshift(self.fpsf)
        self.psf_tot.value[:] = self.psf_bfm.value * self.fp_mask
        self.pup_pyr_tot[:] = self.xp.roll(self.pyr_image, self.roll_array, self.roll_axis)
        self.psf_tot.value *= self.factor
        self.psf_bfm.value *= self.factor
        trasmission_factor = 1 / (self.xp.sum(self.psf_bfm.value) + 1e-20)
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) * trasmission_factor
        # Call parent post_trigger
        super().post_trigger()
