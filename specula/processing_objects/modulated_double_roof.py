from specula import fuse, np

from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.data_objects.simul_params import SimulParams

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


class ModulatedDoubleRoof(ModulatedPyramid):
    """
    Pyramid wavefront sensor with double roof processing object.
    Includes tip-tilt modulation and double roof.

    The two roofs have one edge each: roof 1 a vertical edge (along the
    rows), roof 2 a horizontal one. Each roof is the envelope of the planes
    of its two faces (their maximum: the phase delay is lowest on the edge).
    The pyramid defects of ModulatedPyramid apply to each roof as follows:

    - pyr_edge_def_ld: random phase on a strip of this width along the edge;
    - pyr_tip_maya_ld: truncated edge, a flat strip of this width along the
      edge (the roof analogue of the flat tip of the pyramid);
    - pyr_max_side_ld: finite support, as for the pyramid;
    - pyr_tip_def_ld: not supported (a roof has no tip), it must be 0;
    - pyr_tlt_coeff: not supported.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 pup_diam: float,
                 output_resolution: int,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
                 fov_errinf: float = 0.5,
                 fov_errsup: float = 2,
                 pup_dist: float = None,
                 pup_margin: int = 2,
                 fft_res: float = 3.0,
                 fp_obs: float = None,
                 pup_shifts = (0.0, 0.0),
                 pyr_tlt_coeff: float = None,
                 pyr_edge_def_ld: float = 0.0,
                 pyr_tip_def_ld: float = 0.0,
                 pyr_tip_maya_ld: float = 0.0,
                 pyr_max_side_ld: float = 0.0,
                 min_pup_dist: float = None,
                 rotAnglePhInDeg: float = 0.0,
                 xShiftPhInPixel: float = 0.0,    # same as SH
                 yShiftPhInPixel: float = 0.0,    # same as SH
                 target_device_idx: int = None,
                 precision: int = None
                ):
        if pyr_tip_def_ld != 0:
            raise ValueError('pyr_tip_def_ld is not supported by ModulatedDoubleRoof: '
                             'a roof has no tip')
        super().__init__(simul_params=simul_params,
                 wavelengthInNm=wavelengthInNm,
                 fov=fov,
                 pup_diam=pup_diam,
                 output_resolution=output_resolution,
                 mod_amp=mod_amp,
                 mod_step=mod_step,
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
                 pyr_max_side_ld=pyr_max_side_ld,
                 min_pup_dist=min_pup_dist,
                 rotAnglePhInDeg=rotAnglePhInDeg,
                 xShiftPhInPixel=xShiftPhInPixel,
                 yShiftPhInPixel=yShiftPhInPixel,
                 target_device_idx=target_device_idx,
                 precision=precision)

        self.stream_enable = True

        self.pup_diam = pup_diam
        self.pup_dist = pup_dist

        # After initialization, create the second roof's exponential
        # (the base class masked_exp is the first roof's one, see get_pyr_tlt())
        self.masked_exp_roof2 = self.get_masked_exp(self.roof2_tlt)

        # Pre-allocate arrays to avoid memory allocation in trigger_code
        self.roof1_image = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)
        self.roof2_image = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)

        # Pre-calculate mid points
        self.mid_h = self.fft_totsize // 2
        self.mid_w = self.fft_totsize // 2

    def get_pyr_tlt(self, p, c):
        """
        Phase delays of the two roofs [tilt pixels / tilt_scale]. Stores both
        (roof1_tlt, roof2_tlt) and returns the first one, which becomes the
        masked_exp of the base class.

        As in ModulatedPyramid.get_pyr_tlt(), in three steps, on the CPU:
        geometry (two face planes, truncated edge), random edge defects,
        finite support.
        """
        if self.pyr_tlt_coeff is not None:
            raise NotImplementedError('pyr_tlt_coeff is not tested yet for double roof')

        u, v = self._tip_coords(p, c)
        n = u.shape[1]
        # Row and column indices, for the tilts that shift each roof's pupils to one side
        y, x = np.mgrid[0:n, 0:n]

        # Face planes of each roof, (u, v) gradients, and the tilt across its edge.
        # Roof 1: -u - 1/2 (left) and u - 1/2 (right); roof 2: the same rotated
        # by 90 degrees (-v - 1/2 and v - 1/2), as np.rot90 of roof 1.
        grad1 = np.array([[-1.0, 0.0], [1.0, 0.0]])
        grad2 = np.array([[0.0, -1.0], [0.0, 1.0]])
        # The former code added A - 1 - 0.5 * y to roof 1 and A - 1 - 0.5 * x to
        # roof 2, but roof 2 was a view (rot90) of roof 1, so both tilts went
        # into both roofs: the effective tilt, kept here, is twice that one.
        tilt1 = 2 * ((n // 2) - 1 - 0.5 * y)
        tilt2 = 2 * ((n // 2) - 1 - 0.5 * x)

        roofs = []
        for grad, tilt in ((grad1, tilt1), (grad2, tilt2)):
            faces = np.stack([grad[k, 0] * u + grad[k, 1] * v - 0.5 for k in range(2)])
            roof_tlt = faces.max(axis=0)

            # Truncated edge: flat strip of width pyr_tip_maya_ld along the edge,
            # r above the bottom of the V (-1/2, on the edge)
            if self.pyr_tip_maya_ld > 0:
                r = self.pyr_tip_maya_ld * self.fft_res / 2
                roof_tlt = np.maximum(roof_tlt, -0.5 + r)
                self.logger.info(f'get_pyr_tlt: flat strip of {2 * r:.1f} pixels along the '
                                 'roof edge (pyramid imperfect tip, maya)')

            roof_tlt = roof_tlt + tilt

            # Random defects along the edge, with values up to the maximum of the roof
            if self.pyr_edge_def_ld > 0:
                dist = self._edge_distance(-faces, -grad)
                idx_edge = np.nonzero(dist <= self.pyr_edge_def_ld * self.fft_res / 2)
                roof_tlt[idx_edge] = roof_tlt.max() * np.random.rand(len(idx_edge[0]))
                self.logger.info(f'get_pyr_tlt: {len(idx_edge[0])} pixels set to random values'
                                 ' to consider pyramid imperfect edges')

            # Finite support
            if self.pyr_max_side_ld > 0:
                max_side = self.pyr_max_side_ld * self.fft_res / 2
                idx_max = np.nonzero(np.sqrt(u ** 2 + v ** 2) > max_side)
                roof_tlt[idx_max] = 0.0
                self.logger.info(
                    f'get_pyr_tlt: {len(idx_max[0])} pixels outside the support radius '
                    f'({self.pyr_max_side_ld} lambda/D) set to 0'
                )

            roofs.append(self.to_xp(roof_tlt, dtype=self.dtype) / self.tilt_scale)

        self.roof1_tlt, self.roof2_tlt = roofs
        return self.roof1_tlt


    def can_accelerate(self):
        return self.stream_enable

    def trigger_code(self):
        ey, ex = self.step_phasors()
        psf_bfm = self.psf_bfm.value
        psf_bfm[:] = 0

        # Clear pre-allocated arrays instead of creating new ones
        self.roof1_image *= 0
        self.roof2_image *= 0

        # Focal plane and roof masks are centered, and the masks re-center
        # the pupils (see ModulatedPyramid): no fftshift or roll is needed
        for i in range(0, self.mod_steps):
            u_fp = self.focal_plane_field(ey[i], ex[i])

            # Process first roof
            u_fp_roof1 = pyr1_fused(u_fp, self.ffv[i], psf_bfm, self.masked_exp, xp=self.xp)
            pyr_ef_roof1 = self._scipy_ifft2(u_fp_roof1, overwrite_x=True, norm='forward')
            self.roof1_image += pyr1_abs2(pyr_ef_roof1, self.ifft_norm, self.ffv[i], xp=self.xp)

            # Process second roof. The focal plane PSF is the same for the two roofs,
            # so it is accumulated in psf_bfm only by the first one.
            u_fp_roof2 = u_fp * self.masked_exp_roof2
            pyr_ef_roof2 = self._scipy_ifft2(u_fp_roof2, overwrite_x=True, norm='forward')
            self.roof2_image += pyr1_abs2(pyr_ef_roof2, self.ifft_norm, self.ffv[i], xp=self.xp)
            del u_fp, u_fp_roof1, pyr_ef_roof1, u_fp_roof2, pyr_ef_roof2

        # Combine the two roof images to create 4 sub-pupils
        self._combine_roof_images()

        psf_bfm *= self.factor
        self.xp.multiply(psf_bfm, self.fp_mask, out=self.psf_tot.value)
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) / self.xp.sum(psf_bfm)

        self._pupils_to_ccd()

    def _combine_roof_images(self):
        """Combine two roof images into a 4-quadrant pyramid-like pattern"""
        # rotate by 90 degrees roof2_image
        roof2_rotated = self.xp.rot90(self.roof2_image)

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.imshow(self.roof1_image)
            plt.colorbar()
            plt.title("Roof 1 Image")
            plt.figure(figsize=(10, 5))
            plt.imshow(self.roof2_image)
            plt.colorbar()
            plt.title("Roof 2 Image")
            plt.figure(figsize=(10, 5))
            plt.imshow(roof2_rotated)
            plt.colorbar()
            plt.title("Roof 2 Image Rotated")
            plt.figure(figsize=(10, 5))
            plt.imshow(self.roof1_image + roof2_rotated)
            plt.colorbar()
            plt.title("Combined Roof Images")
            plt.show()

        # Combine the shifted images
        self.pup_pyr_tot[:] = self.roof1_image + roof2_rotated
