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
    
    This class extends ModulatedPyramid to handle extended sources by computing
    pupil phases on-the-fly for each source point, reducing memory usage compared
    to pre-computing and storing all tip-tilt exponentials.
    
    The extended source is represented by a set of point sources, each with
    tip, tilt, focus coefficients and flux. Processing is done in batches to
    manage GPU memory efficiently.
    
    Extended Source Specific Parameters
    ------------------------------------
    max_batch_size : int, optional
        Maximum number of source points processed simultaneously (default: 1024).
        Larger values increase GPU memory usage but may improve performance.
        Reduce this value if you encounter out-of-memory errors.
    max_flux_ratio_thr : float, optional
        Flux threshold ratio for filtering low-flux source points (default: 1e-3).
        Points with flux below (max_flux * max_flux_ratio_thr) are ignored,
        but their flux is redistributed to four points in the middle of the pyramid faces.
        Only used when cuda_stream_enable=False. When enabled, reduces computation
        but may affect accuracy for sources with very faint extended components.
    cuda_stream_enable : bool, optional
        Enable CUDA stream for graph capture and optimized GPU execution (default: True).
        When True, all source points are processed (flux thresholding disabled) to
        maintain constant computational load required for CUDA graph compatibility.
        Set to False for debugging or when source point count varies significantly
        between frames and you want to use flux thresholding.
    target_device_idx : int, optional
        GPU device index (default: None, uses default device)
    precision : int, optional
        Numerical precision: 32 or 64 bits (default: None, uses system default)

    Inherited Parameters (Not Used)
    --------------------------------
    mod_amp, mod_step, mod_type
        These control tip-tilt modulation for point sources. In extended source mode,
        they are ignored. The "modulation" is implicitly defined by the spatial
        distribution of source points provided via the ext_source_coeff input.
    
    Extended Source Specific Inputs
    ------
    ext_source_coeff : BaseValue
        Extended source coefficients array of shape (n_points, 4) with columns:
        [tip_coeff, tilt_coeff, focus_coeff, flux]. Typically provided by an
        ExtendedSource object. This input replaces the modulation pattern used
        in point source mode.
    
    Notes
    -----
    No specific outputs for extended source pyramid, uses same outputs as ModulatedPyramid.
    
    Memory usage scales with (max_batch_size * fft_totsize^2 * 16 bytes) for complex
    arrays. For a 512x512 FFT grid and batch_size=1024, this is ~8 GB per batch.
    
    Processing modes:
    - cuda_stream_enable=True (default): All source points processed, optimal performance,
      required for CUDA graph acceleration, no flux filtering
    - cuda_stream_enable=False: Flux filtering enabled, variable processing load,
      useful for debugging or sources with many low-flux points

    See Also
    --------
    ModulatedPyramid : Parent class for point source pyramid WFS
    ExtendedSource : Source object that generates ext_source_coeff

    Performance Tips
    ----------------
    1. For large sources (many points): use cuda_stream_enable=True (default)
       with max_batch_size=512-1024 depending on GPU memory
    2. For debugging or flux filtering: set cuda_stream_enable=False
       and adjust max_flux_ratio_thr (default 1e-3)
    3. Monitor GPU memory with nvidia-smi during first iterations

    Examples
    --------
    >>> # Usage with CUDA stream
    >>> pyr = ExtSourcePyramid(
    ...     simul_params=params,
    ...     wavelengthInNm=500,
    ...     fov=2.0,
    ...     pup_diam=30,
    ...     output_resolution=80,
    ...     ...
    ...     max_batch_size=512  # Adjust based on available GPU memory
    ... )
    
    >>> # Flux filtering mode recommended for very large sources with many 
    >>> # points with low flux (ExtendedSource class with source_type='FROM_PSF')
    >>> pyr = ExtSourcePyramid(
    ...     simul_params=params,
    ...     wavelengthInNm=500,
    ...     fov=2.0,
    ...     pup_diam=30,
    ...     output_resolution=80,
    ...     ...
    ...     cuda_stream_enable=False,
    ...     max_flux_ratio_thr=1e-3  # Filter points with flux < max_flux/10000
    ... )
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
                 max_batch_size: int = 128,
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

        # Validate parameters
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        if not 0 < max_flux_ratio_thr < 1:
            raise ValueError(f"max_flux_ratio_thr must be in (0, 1),"
                             f" got {max_flux_ratio_thr}")

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

        if self.stream_enable and hasattr(self.xp, '__name__') and self.xp.__name__ == 'cupy':
            print('CUDA stream enabled for extended source pyramid processing'
                  ' Ignoring flux thresholding to maintain constant processing load.')

        # Pre-allocated buffers for CUDA graph compatibility (allocated in cache_ttexp)
        self._fpsf_buffer = None
        self._pyr_image_buffer = None
        self._n_chunks = 0
        self._coeff_padded = None
        self._ffv_padded = None
        self._u_tlt_batch = None

        # Pre-allocate face center coefficients (4 points at pyramid face centers)
        # These will be used to redistribute filtered flux
        self._face_centers_idx = None  # Indices where face centers are stored in coeff array
        self._face_centers_ttf = None  # Pre-computed TTF coordinates (initialized in cache_ttexp)

        # Add dedicated input for extended source coefficients
        self.inputs['ext_source_coeff'] = InputValue(type=BaseValue)


    def _get_pyramid_face_angles_at_fov_radius(self):
        """
        Calculate 4 points at the corners between pyramid faces, at a radius
        corresponding to the field of view of the extended source.
        
        Returns
        -------
        face_angles_ttf : ndarray
            Array of shape (4, 3) with [tip, tilt, focus=0] coordinates
        mean_radius : float
            Radius in lambda/D units corresponding to half the FoV
        """
        # Calculate mean radius from pyramid FoV and sampling
        mean_radius = (self.fov / 2.0) / self.fov_res * (1.0 / self.fft_res)

        # Place 4 points at face centers (perpendicular to pyramid edges)
        # Standard pyramid has 4 faces with normals at 0°, 90°, 180°, 270°
        angles = self.xp.array([0, 90, 180, 270], dtype=self.dtype) * self.xp.pi / 180

        face_angles_tt = self.xp.stack([
            mean_radius * self.xp.cos(angles),  # tip
            mean_radius * self.xp.sin(angles),  # tilt
        ], axis=1)

        # Add zero focus component
        face_angles_ttf = self.xp.hstack([
            face_angles_tt,
            self.xp.zeros((4, 1), dtype=self.dtype)
        ])

        return face_angles_ttf, mean_radius


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

            # Pre-compute face center TTF coordinates (once for all)
            if self._face_centers_ttf is None:
                face_angles_ttf, _ = self._get_pyramid_face_angles_at_fov_radius()
                self._face_centers_ttf = face_angles_ttf

            # Pre-allocate buffers for batch processing (constant size for FFT plan reuse)
            self._coeff_padded = self.xp.zeros((self.max_batch_size, 3), dtype=self.dtype)
            self._ffv_padded = self.xp.zeros(self.max_batch_size, dtype=self.dtype)
            self._u_tlt_batch = self.xp.zeros(
                (self.max_batch_size, self.fft_totsize, self.fft_totsize),
                dtype=self.complex_dtype)

        # Always update face centers when stream disabled (in case source was updated)
        if not self.stream_enable:
            # Check if we need to append face centers
            # (either first time or source was updated and lost them)
            current_size = self.ext_source_coeff.value.shape[0]
            if self._face_centers_idx is None or self._face_centers_idx[0] >= current_size:
                # Create face centers with flux initialized to zero
                face_centers_with_flux = self.xp.hstack([
                    self._face_centers_ttf,
                    self.xp.zeros((4, 1), dtype=self.dtype)  # Initial flux = 0
                ])

                # Append face centers
                self.ext_source_coeff.value = self.xp.vstack([
                    self.ext_source_coeff.value,
                    face_centers_with_flux
                ])
                self._face_centers_idx = self.xp.arange(current_size, current_size + 4)
                self.mod_steps = current_size + 4
            else:
                # Face centers already exist, just reset their flux to zero
                # (TTF coordinates are constant, no need to update)
                self.ext_source_coeff.value[self._face_centers_idx, 3] = 0.0
                self.mod_steps = self.ext_source_coeff.value.shape[0]
        else:
            # Stream enabled: just update mod_steps
            self.mod_steps = int(self.ext_source_coeff.value.shape[0])

        # Set flux factor vector from source (will be updated in trigger if PSF changes)
        coeff_flux  = self.ext_source_coeff.value[:, 3]
        self.flux_factor_vector = self.to_xp(coeff_flux)

        # Clean up very small flux values (only if stream disabled)
        # When stream_enable=True, we need constant n_valid for CUDA graph
        if not self.stream_enable:
            # Compute total flux before filtering (excluding face centers)
            n_original = self._face_centers_idx[0]  # Face centers start here
            original_flux = self.flux_factor_vector[:n_original]
            total_flux = self.xp.sum(original_flux) + 1e-20

            max_flux = self.xp.max(self.xp.abs(original_flux))
            threshold = max_flux * self.max_flux_ratio_thr

            # Create boolean mask for points below threshold
            below_threshold = original_flux <= threshold
            not_valid_idx = self.xp.where(below_threshold)[0]
            n_filtered = len(not_valid_idx)

            # Compute lost flux ratio
            lost_flux = self.xp.sum(original_flux[not_valid_idx])
            lost_flux_ratio =  lost_flux / total_flux

            # Set flux factors below threshold to zero
            self.flux_factor_vector[:n_original][not_valid_idx] = 0.0

            # Redistribute lost flux to 4 face centers
            if n_filtered > 0 and lost_flux > 0:
                self.ext_source_coeff.value[self._face_centers_idx, 3] = lost_flux / 4.0
            else:
                self.ext_source_coeff.value[self._face_centers_idx, 3] = 0.0

            # Update flux factor vector after redistribution
            self.flux_factor_vector = self.ext_source_coeff.value[:, 3]

            self.valid_idx = self.xp.where(self.flux_factor_vector > 0.0)[0]

            print(f'Points with flux below {threshold:.3e} set to zero:'
                  f' {n_filtered} out of {n_original}'
                  f', {lost_flux_ratio*100:.1f}% of flux')

            # Reallocate buffers only when stream disabled (dynamic filtering)
            n_chunks_needed = (len(self.valid_idx) + self.max_batch_size - 1) \
                            // self.max_batch_size
        else:
            # With stream enabled, process all points (no filtering)
            # to keep constant loop iterations for CUDA graph
            print(f'Stream enabled: processing all {self.mod_steps} points')
            self.valid_idx = self.xp.arange(self.mod_steps)

            # Allocate buffers once because with stream enabled valid_idx is constant
            n_chunks_needed = (len(self.valid_idx) + self.max_batch_size - 1) \
                            // self.max_batch_size

        self.factor = 1.0 / (self.xp.sum(self.flux_factor_vector) + 1e-20)

        if self._n_chunks != n_chunks_needed:
            self._n_chunks = n_chunks_needed
            self._fpsf_buffer = self.xp.zeros((self._n_chunks, *self.fpsf.shape),
                                            dtype=self.dtype)
            self._pyr_image_buffer = self.xp.zeros((self._n_chunks, *self.pyr_image.shape),
                                                dtype=self.dtype)
        else:
            # Clear buffers
            self._fpsf_buffer[:] = 0
            self._pyr_image_buffer[:] = 0

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
        u_tlt_const = self.ef * self.tlt_f

        # Get extended source coefficients for current frame (only valid points)
        coeff_ttf = self.ext_source_coeff.value[self.valid_idx, :3]
        ffv_valid = self.flux_factor_vector[self.valid_idx]
        n_valid = self.valid_idx.shape[0]

        # Process in chunks
        for chunk_idx, start_idx in enumerate(range(0, n_valid, self.max_batch_size)):
            end_idx = min(start_idx + self.max_batch_size, n_valid)
            chunk_size = end_idx - start_idx

            # Copy chunk data into pre-allocated padded arrays (rest remains zero)
            self._coeff_padded[:] = 0
            self._ffv_padded[:] = 0
            self._coeff_padded[:chunk_size] = coeff_ttf[start_idx:end_idx]
            self._ffv_padded[:chunk_size] = ffv_valid[start_idx:end_idx]

            # Compute pupil phases - ALWAYS full batch size
            pup_phases = self.xp.sum(self._coeff_padded[:, :, None, None] \
                                    * self.ttf_signs[None, :, None, None] \
                                    * self.ext_ttf[None, :, :, :],
                                    axis=1)

            # Compute ttexp - ALWAYS full batch size
            ttexp_batch = self.xp.exp(-iu * pup_phases, dtype=self.complex_dtype)

            # Prepare u_tlt_batch - ALWAYS full batch size (reuse pre-allocated buffer)
            self._u_tlt_batch[:] = 0
            self._u_tlt_batch[:, 0:self.ttexp_shape[1], 0:self.ttexp_shape[2]] = \
                u_tlt_const[None, :, :] * ttexp_batch

            # Batch FFT - ALWAYS same size
            u_fp_batch = self.xp.fft.fft2(self._u_tlt_batch, axes=(-2, -1))

            # Store PSF contribution - use only valid results
            psf_batch = self.xp.real(u_fp_batch * self.xp.conj(u_fp_batch))
            self._fpsf_buffer[chunk_idx] = \
                self.xp.sum(psf_batch * self._ffv_padded[:, None, None], axis=0)

            # Apply pyramid mask - ALWAYS full batch size
            u_fp_pyr_batch = u_fp_batch * self.shifted_masked_exp[None, :, :]

            # Batch inverse FFT - ALWAYS same size (no need for separate padding)
            pyr_ef_batch = self.xp.fft.ifft2(u_fp_pyr_batch, axes=(-2, -1), norm='forward')

            # Store pyramid image contribution - use only valid results
            pyr_ef_norm = pyr_ef_batch * self.ifft_norm
            pyr_images = self.xp.real(pyr_ef_norm * self.xp.conj(pyr_ef_norm))
            self._pyr_image_buffer[chunk_idx] = \
                self.xp.sum(pyr_images * self._ffv_padded[:, None, None], axis=0)

        # Final reduction
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
