
import numpy as np

from specula import fuse
from specula.lib.make_mask import make_mask
from specula.lib.make_xy import make_xy
from specula.lib.utils import unravel_index_2d
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.base_value import BaseValue

from specula.processing_objects.slopec import Slopec


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)

class ShSlopec(Slopec):
    def __init__(self,
                 subapdata: SubapData,
                 sn: Slopes=None,
                 thr_value: float = -1,
                 exp_weight: float = 1.0,
                 filtmat=None,
                 weightedPixRad: float = 0.0,
                 windowing: bool = False,
                 weight_int_pixel_dt: float=0,
                 window_int_pixel: bool=False,
                 window_int_threshold: float=1.0,
                 vecWeiPixRadT: list=None,
                 target_device_idx: int = None,
                 precision: int = None):

        # Set subaperture data before initializing base class
        # because we need to know the number of subapertures
        self.subapdata = subapdata

        super().__init__(sn=sn, filtmat=filtmat, weight_int_pixel_dt=weight_int_pixel_dt,
                         target_device_idx=target_device_idx, precision=precision)
        self.thr_value = thr_value
        self.thr_mask_cube = BaseValue(target_device_idx=self.target_device_idx,
                                       precision=precision)
        self.xweights = None
        self.yweights = None
        self.xcweights = None
        self.ycweights = None
        self.mask_weighted = None
        self.weighted_pix_rad = weightedPixRad
        self.vec_wei_pix_rad_t = vecWeiPixRadT
        self.windowing = windowing
        self.thr_ratio_value = 0.0
        self.thr_pedestal = False
        self.mult_factor = 0.0
        self.quadcell_mode = False
        self.two_steps_cog = False
        self.cog_2ndstep_size = 0
        self.store_thr_mask_cube = False   # Todo should it become a parameter?

        self.exp_weight = exp_weight
        self.window_int_pixel = window_int_pixel
        self.window_int_threshold = window_int_threshold
        self.int_pixels_weight = None

        self.accumulated_slopes = Slopes(self.nslopes(), target_device_idx=self.target_device_idx)
        self.set_xy_weights()
        self.outputs['out_subapdata'] = self.subapdata

        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

    def nsubaps(self):
        return self.subapdata.n_subaps

    def nslopes(self):
        return self.subapdata.n_subaps * 2

    @property
    def subap_idx(self):
        return self.subapdata.idxs

    def set_xy_weights(self):
        if self.subapdata:
            out = self.computeXYweights(self.subapdata.np_sub, self.exp_weight, self.weighted_pix_rad,
                                          self.quadcell_mode, self.windowing)
            self.mask_weighted = self.to_xp(out['mask_weighted'])
            self.xweights = self.to_xp(out['x'])
            self.yweights = self.to_xp(out['y'])
            self.xcweights = self.to_xp(out['xc'])
            self.ycweights = self.to_xp(out['yc'])
            self.xweights_flat = self.xweights.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)
            self.yweights_flat = self.yweights.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)
            self.mask_weighted_flat = self.mask_weighted.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)

    def computeXYweights(self, np_sub, exp_weight, weightedPixRad, quadcell_mode=False, windowing=False):
        """
        Compute XY weights for SH slope computation.

        Parameters:
        np_sub (int): Number of subapertures.
        exp_weight (float): Exponential weight factor.
        weightedPixRad (float): Radius for weighted pixels.
        quadcell_mode (bool): Whether to use quadcell mode.
        windowing (bool): Whether to apply windowing.
        """
        # Generate x, y coordinates
        x, y = make_xy(np_sub, 1.0, xp=np, dtype=self.dtype)

        # Compute weights in quadcell mode or otherwise
        if quadcell_mode:
            x = np.where(x > 0, 1.0, -1.0)
            y = np.where(y > 0, 1.0, -1.0)
            xc, yc = x.copy(), y.copy()
        else:
            xc, yc = x.copy(), y.copy()
            # Apply exponential weights if exp_weight is not 1
            x = np.where(x > 0, np.power(x, exp_weight), -np.power(np.abs(x), exp_weight))
            y = np.where(y > 0, np.power(y, exp_weight), -np.power(np.abs(y), exp_weight))

        # Adjust xc, yc for centroid calculations in two steps (as in IDL)
        xc = np.where(xc > 0, np.abs(xc), -np.abs(xc))
        yc = np.where(yc > 0, np.abs(yc), -np.abs(yc))

        # Apply windowing or weighted pixel mask
        if weightedPixRad != 0:
            if windowing:
                # Windowing case (must be an integer)
                mask_weighted = make_mask(np_sub, diaratio=(2.0 * weightedPixRad / np_sub), xp=np)
            else:
                # Weighted Center of Gravity (WCoG)
                mask_weighted = self.psf_gaussian(np_sub, [2*weightedPixRad, 2*weightedPixRad])
                mask_weighted /= np.max(mask_weighted)

            mask_weighted[mask_weighted < 1e-6] = 0.0

            x *= mask_weighted.astype(self.dtype)
            y *= mask_weighted.astype(self.dtype)
        else:
            mask_weighted = np.ones((np_sub, np_sub), dtype=self.dtype)

        return {"x": x, "y": y, "xc": xc, "yc": yc, "mask_weighted": mask_weighted}

    def trigger_code(self):
        if self.vec_wei_pix_rad_t is not None:
            idxW = self.xp.where(self.current_time_seconds > self.vec_wei_pix_rad_t[:, 1])[-1]
            if len(idxW) > 0:
                self.weighted_pix_rad = self.vec_wei_pix_rad_t[idxW, 0]
                if self.verbose:
                    print(f'self.weighted_pix_rad: {self.weighted_pix_rad}')
                self.set_xy_weights()

        if self.weight_int_pixel_dt > 0:
            self.do_accumulation(self.current_time)

        self.calc_slopes_nofor()

    def calc_slopes_nofor(self):
        """
        Calculate slopes without a for-loop over subapertures.
        """
        if self.verbose and self.subapdata is None:
            print('subapdata is not valid.')
            return

        in_pixels = self.local_inputs['in_pixels'].pixels

        n_subaps = self.subapdata.n_subaps
        np_sub = self.subapdata.np_sub

        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError("Only one between _thr_value and _thr_ratio_value can be set.")

        # Reform pixels based on the subaperture index
        idx2d = unravel_index_2d(self.subap_idx, in_pixels.shape, self.xp)
        pixels = in_pixels[idx2d].T

        if self.weight_int_pixel:

            if self.int_pixels_weight is None:
                self.int_pixels_weight = self.xp.ones_like(pixels, dtype=self.dtype)

            n_weight_applied = 0
            if self.int_pixels is not None and self.int_pixels.generation_time == self.current_time:
                # Reshape accumulated pixels to match the format
                int_pixels_weight = self.int_pixels.pixels[idx2d].T.astype(self.dtype)
                int_pixels_weight -= self.xp.min(int_pixels_weight, axis=0, keepdims=True)
                max_temp = self.xp.max(int_pixels_weight, axis=0)

                # Handle subapertures with zero or negative max values
                valid_mask = max_temp > 0

                if not self.xp.any(valid_mask):
                    int_pixels_weight.fill(1.0)
                elif self.window_int_pixel:
                    # Apply windowing condition exactly like IDL in 2D
                    above_threshold = int_pixels_weight >= self.window_int_threshold

                    # IDL: reverse(weight, 1) - flip only first dimension
                    weight_flipped = self.xp.flip(int_pixels_weight, axis=0)
                    above_threshold_flipped = weight_flipped >= self.window_int_threshold

                    # Combine with OR
                    window_mask = above_threshold | above_threshold_flipped

                    # Convert to weights
                    int_pixels_weight = window_mask.astype(self.dtype)

                    # Handle invalid subapertures
                    int_pixels_weight[:, ~valid_mask] = 1.0

                    n_weight_applied = self.xp.sum(self.xp.any(int_pixels_weight > 0, axis=0))
                else:
                    # Normalize by max value for valid subapertures
                    int_pixels_weight[:, valid_mask] /= max_temp[valid_mask]
                    int_pixels_weight[:, ~valid_mask] = 1.0
                    n_weight_applied = self.xp.sum(valid_mask)

                self.int_pixels_weight[:] = int_pixels_weight

            # Apply weights to pixels
            pixels *= self.int_pixels_weight

            if self.verbose:  # pragma: no cover
                print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

        # Calculate flux and max flux per subaperture
        flux_per_subaperture_vector = self.xp.sum(pixels, axis=0)
        max_flux_per_subaperture = self.xp.max(flux_per_subaperture_vector)

        # Thresholding logic
        if self.thr_ratio_value > 0:
            thr = self.thr_ratio_value * max_flux_per_subaperture
            thr = thr[:, self.xp.newaxis] * self.xp.ones((1, np_sub * np_sub))
        elif self.thr_pedestal or self.thr_value > 0:
            thr = self.thr_value
        else:
            thr = 0

        if self.thr_pedestal:
            clamp_generic_less(thr, 0, pixels, xp=self.xp)
        else:
            pixels -= thr
            clamp_generic_less(0, 0, pixels, xp=self.xp)

        if self.store_thr_mask_cube:
            thr_mask_cube = thr.reshape(np_sub, np_sub, n_subaps)

        # Compute denominator for slopes
        subap_tot = self.xp.sum(pixels * self.mask_weighted_flat, axis=0)
        mean_subap_tot = self.xp.mean(subap_tot)
        factor = 1.0 / subap_tot

# TEST replacing these three lines with clamp_generic_more
#        idx_le_0 = self.xp.where(subap_tot <= mean_subap_tot * 1e-3)[0]
#        if len(idx_le_0) > 0:
#            factor[idx_le_0] = 0.0
        clamp_generic_more( 1.0 / (mean_subap_tot * 1e-3), 0, factor, xp=self.xp)

        # Compute slopes
        sx = self.xp.sum(pixels * self.xweights_flat * factor[self.xp.newaxis, :], axis=0)
        sy = self.xp.sum(pixels * self.yweights_flat * factor[self.xp.newaxis, :], axis=0)

        if self.mult_factor != 0:
            sx *= self.mult_factor
            sy *= self.mult_factor
            print("WARNING: multiplication factor in the slope computer!")

        if self.store_thr_mask_cube:
            self.thr_mask_cube.value = thr_mask_cube
            self.thr_mask_cube.generation_time = self.current_time

        self.slopes.xslopes = sx
        self.slopes.yslopes = sy
        self.slopes.generation_time = self.current_time

        self.flux_per_subaperture_vector.value[:] = flux_per_subaperture_vector
        self.total_counts.value[0] = self.xp.sum(flux_per_subaperture_vector)
        self.subap_counts.value[0] = self.xp.mean(flux_per_subaperture_vector)

        if self.verbose:  # pragma: no cover
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    def psf_gaussian(self, np_sub, fwhm):
        """Generates a 2D Gaussian PSF.

        Args:
            np_sub (int): Number of sub-apertures (pixels) in one dimension.
            fwhm (list): Full width at half maximum (FWHM) in pixels for x and y directions.

        Returns:
            np.ndarray: 2D array representing the Gaussian PSF.
        """
        cntrd = (np_sub - 1) / 2.0

        x = np.arange(np_sub) - cntrd  # from -(np_sub-1)/2 to +(np_sub-1)/2
        y = np.arange(np_sub) - cntrd

        st_dev_x = fwhm[0] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        st_dev_y = fwhm[1] / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        gaussian_x = np.exp(-0.5 * (x / st_dev_x)**2)
        gaussian_y = np.exp(-0.5 * (y / st_dev_y)**2)

        gaussian = np.outer(gaussian_x, gaussian_y)
        return gaussian

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time
