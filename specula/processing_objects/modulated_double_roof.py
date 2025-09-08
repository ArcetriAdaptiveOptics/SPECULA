from specula import fuse

from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.lib.make_xy import make_xy
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
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float, # TODO =750,
                 fov: float,            # TODO =2.0,
                 pup_diam: int,         # TODO =30,
                 output_resolution: int,# TODO =80,
                 mod_amp: float = 3.0,
                 mod_step: int = None,
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
                 xShiftPhInPixel: float = 0.0,    # same as SH
                 yShiftPhInPixel: float = 0.0,    # same as SH
                 target_device_idx: int = None,
                 precision: int = None
                ):
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
        iu = 1j  # complex unit
        roof1_exp = self.xp.exp(-2 * self.xp.pi * iu * self.roof1_tlt, dtype=self.complex_dtype)
        roof2_exp = self.xp.exp(-2 * self.xp.pi * iu * self.roof2_tlt, dtype=self.complex_dtype)

        self.shifted_masked_exp_roof1 = self.xp.fft.fftshift(roof1_exp * self.fp_mask)
        self.shifted_masked_exp_roof2 = self.xp.fft.fftshift(roof2_exp * self.fp_mask)

        # Pre-allocate arrays to avoid memory allocation in trigger_code
        self.roof1_image = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)
        self.roof2_image = self.xp.zeros((self.fft_totsize, self.fft_totsize), dtype=self.dtype)
        self.roof2_factor = self.xp.ones((self.fft_totsize, self.fft_totsize), dtype=self.dtype)

        # Pre-calculate mid points
        self.mid_h = self.fft_totsize // 2
        self.mid_w = self.fft_totsize // 2

    def get_pyr_tlt(self, p, c):
        A = int((p + c) // 2)
        # Create two separate roofs instead of a 4-faced pyramid
        roof1_tlt = self.xp.zeros((2 * A, 2 * A), dtype=self.dtype)
        roof2_tlt = self.xp.zeros((2 * A, 2 * A), dtype=self.dtype)

        y, x = self.xp.mgrid[0:2*A,0:2*A]

        if self.pyr_tlt_coeff is not None:
            raise NotImplementedError('pyr_tlt_coeff is not tested yet for double roof')
        else:
            # First roof: horizontal separation (left/right)
            # Left half: decreasing from center to left edge
            roof1_tlt[:, :A] = A - 1 - x[:, :A]
            # Right half: increasing from center to right edge
            roof1_tlt[:, A:] = x[:, A:] - A

            # Second roof: vertical separation (top/bottom)
            roof2_tlt = self.xp.rot90(roof1_tlt)

            # add to roof1_tlt a tilt in the other direction to shift the image on one side
            roof1_tlt += A - 1 - 0.5*y

            # add to roof2_tlt a tilt in the other direction to shift the image on one side
            roof2_tlt += A - 1 - 0.5*x

        # Apply edge and tip defects to both roofs
        xx, yy = make_xy(A * 2, A, xp=self.xp)

        for roof_tlt in [roof1_tlt, roof2_tlt]:
            # distance from edge
            dx = self.xp.sqrt(xx ** 2)
            dy = self.xp.sqrt(yy ** 2)
            idx_edge = self.xp.where((dx <= self.pyr_edge_def_ld * self.fft_res / 2) |
                                (dy <= self.pyr_edge_def_ld * self.fft_res / 2))
            if len(idx_edge[0]) > 0:
                roof_tlt[idx_edge] = self.xp.max(roof_tlt) * self.xp.random.rand(len(idx_edge[0]))

            # distance from tip
            d = self.xp.sqrt(xx ** 2 + yy ** 2)
            idx_tip = self.xp.where(d <= self.pyr_tip_def_ld * self.fft_res / 2)
            if len(idx_tip[0]) > 0:
                roof_tlt[idx_tip] = self.xp.max(roof_tlt) * self.xp.random.rand(len(idx_tip[0]))

            # distance from tip (maya)
            idx_tip_m = self.xp.where(d <= self.pyr_tip_maya_ld * self.fft_res / 2)
            if len(idx_tip_m[0]) > 0:
                roof_tlt[idx_tip_m] = self.xp.min(roof_tlt[idx_tip_m])

        # Store both roofs
        self.roof1_tlt = roof1_tlt / self.tilt_scale
        self.roof2_tlt = roof2_tlt / self.tilt_scale

        # Return the first roof for compatibility (the second will be accessed directly)
        return self.roof1_tlt

    def trigger_code(self):
        u_tlt_const = self.ef * self.tlt_f
        tmp = u_tlt_const[self.xp.newaxis, :, :] * self.ttexp
        self.u_tlt[:, 0:self.ttexp_shape[1], 0:self.ttexp_shape[2]] = tmp
        self.pyr_image *= 0
        self.fpsf *= 0

        # Clear pre-allocated arrays instead of creating new ones
        self.roof1_image *= 0
        self.roof2_image *= 0

        for i in range(0, self.mod_steps):
            u_fp = self.xp.fft.fft2(self.u_tlt[i], axes=(-2, -1))

            # Process first roof
            u_fp_roof1 = pyr1_fused(u_fp, self.ffv[i], self.fpsf, self.shifted_masked_exp_roof1, xp=self.xp)
            pyr_ef_roof1 = self.xp.fft.ifft2(u_fp_roof1, axes=(-2, -1), norm='forward')
            self.roof1_image += pyr1_abs2(pyr_ef_roof1, self.ifft_norm, self.ffv[i], xp=self.xp)

            # Process second roof
            u_fp_roof2 = pyr1_fused(u_fp, self.ffv[i], self.fpsf, self.shifted_masked_exp_roof2, xp=self.xp)
            pyr_ef_roof2 = self.xp.fft.ifft2(u_fp_roof2, axes=(-2, -1), norm='forward')
            self.roof2_image += pyr1_abs2(pyr_ef_roof2, self.ifft_norm, self.ffv[i], xp=self.xp)

        self.roof1_image[:] = self.xp.roll(self.roof1_image, self.roll_array, self.roll_axis)
        self.roof2_image[:] = self.xp.roll(self.roof2_image, self.roll_array, self.roll_axis)

        # Combine the two roof images to create 4 sub-pupils
        self._combine_roof_images()

        self.psf_bfm.value[:] = self.xp.fft.fftshift(self.fpsf)
        self.psf_tot.value[:] = self.psf_bfm.value * self.fp_mask
        self.pup_pyr_tot[:] = self.pyr_image
        self.psf_tot.value *= self.factor
        self.psf_bfm.value *= self.factor
        self.transmission.value[:] = self.xp.sum(self.psf_tot.value) / self.xp.sum(self.psf_bfm.value)

    def _combine_roof_images(self):
        """Combine two roof images into a 4-quadrant pyramid-like pattern"""
        # Clear the output image
        self.pyr_image *= 0

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
        self.pyr_image[:] = self.roof1_image + roof2_rotated
