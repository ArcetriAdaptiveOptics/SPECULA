import logging

import numpy as np

from specula import fuse
from specula.base_processing_obj import OutputDesc
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.lib.make_mask import make_mask
from specula.lib.make_xy import make_xy

from specula.processing_objects.slopec import Slopec, sum_product


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


@fuse(kernel_name='sh_slopes_normalize')
def sh_slopes_normalize(subap_tot, sx_raw, sy_raw, mean_subap_tot, mult_factor, sx, sy, xp):
    # Subapertures with a denominator below 1e-3 times the average (negative
    # ones included, possible with clamp_negative=False) get zero slopes
    factor = xp.where(subap_tot < mean_subap_tot * 1e-3, 0, 1.0 / subap_tot) * mult_factor
    sx[:] = sx_raw * factor
    sy[:] = sy_raw * factor


class ShSlopec(Slopec):
    """
    Shack-Hartmann slopes computer processing object.
    Computes Shack-Hartmann slopes from pixel data using the subaperture intensities.

    On GPU, trigger_code() is captured in a CUDA graph (see setup()), which
    includes compute_slopes() and the slope corrections of the base class
    (slope null, filtering, slopes map). The pixel accumulation for
    weight_int_pixel_dt and the other operations that change from step to
    step, or that need a CPU-GPU synchronization, are done in prepare_trigger()
    instead. Scalar parameters (thr_value, thr_ratio_value, thr_pedestal,
    mult_factor) are frozen in the graph: call invalidate_graph() after
    changing them. The inputs must be updated in place by their producers
    (a reallocated input raises an error).

    clamp_negative (see Slopec): the negative pixels are set to zero after
    the threshold subtraction. Subapertures with a denominator below 1e-3
    times the average, negative ones included (possible with
    clamp_negative=False), get zero slopes. It has no effect with
    thr_pedestal, which sets the pixels below the threshold to zero.

    Parameters
    ----------
    subapdata : SubapData
        Valid subapertures and their pixels on the detector.
    sn : Slopes, optional
        Slope null (see Slopec).
    thr_value : float [counts], optional
        If > 0, subtracted from all the pixels before the centroid. Default
        -1 (no subtraction). Only one of thr_value and thr_ratio_value can
        be set.
    thr_ratio_value : float, optional
        If > 0, a threshold equal to this fraction of the brightest pixel of
        each subaperture is subtracted from its pixels. Default 0.
    exp_weight : float, optional
        Exponent of the pixel coordinates in the centroid weights:
        ``sign(x) |x|^exp_weight``. Default 1 (center of gravity).
    filtmat : array_like, optional
        Slope filter (see Slopec).
    weightedPixRad : float [pixels], optional
        If > 0, the centroid weights are multiplied by a Gaussian of FWHM
        2 * weightedPixRad (weighted center of gravity), or, with windowing,
        by a circular window of radius weightedPixRad. Default 0.
    windowing : bool, optional
        Use a circular window instead of the Gaussian (see weightedPixRad).
        Default False.
    weight_int_pixel_dt : float [s], optional
        If > 0, the pixels are accumulated over this time and the
        accumulated image of each subaperture (minus its minimum) is used to
        weight its pixels: normalized to its maximum, or, with
        window_int_pixel, as a binary window. Default 0.
    window_int_pixel : bool, optional
        With weight_int_pixel_dt > 0, use as weight the binary window of the
        pixels above window_int_threshold in the accumulated image (made
        symmetric by a flip). Default False.
    window_int_threshold : float [counts], optional
        Threshold of window_int_pixel, on the accumulated image minus its
        minimum. Default 1.
    vecWeiPixRadT : array_like, optional
        Time-dependent weightedPixRad: rows [weightedPixRad, time in s]; from
        each time on, the corresponding radius is used. Default None.
    interleave : bool, optional
        Slope order (see Slopec). Default False.
    clamp_negative : bool, optional
        Set the negative pixels to zero (see Slopec). Default True.
    target_device_idx : int, optional
        Device index (-1 for CPU). Default: global setting.
    precision : int, optional
        0 for double, 1 for single precision. Default: global setting.

    Derived classes do not use a CUDA graph, since their compute_slopes()
    may have host-side logic. Those with GPU-only code can opt in calling
    self.build_stream(capture=False) in their setup().
    """

    def __init__(self,
                 subapdata: SubapData,
                 sn: Slopes=None,
                 thr_value: float = -1,
                 thr_ratio_value: float = 0.0,
                 exp_weight: float = 1.0,
                 filtmat=None,
                 weightedPixRad: float = 0.0,
                 windowing: bool = False,
                 weight_int_pixel_dt: float=0,
                 window_int_pixel: bool=False,
                 window_int_threshold: float=1.0,
                 vecWeiPixRadT: list=None,
                 interleave: bool=False,
                 clamp_negative: bool=True,
                 target_device_idx: int = None,
                 precision: int = None):

        # Set subaperture data before initializing base class
        # because we need to know the number of subapertures
        self.subapdata = subapdata

        super().__init__(sn=sn,
                         filtmat=filtmat,
                         weight_int_pixel_dt=weight_int_pixel_dt,
                         interleave=interleave,
                         clamp_negative=clamp_negative,
                         target_device_idx=target_device_idx,
                         precision=precision)
        self.thr_value = thr_value
        self.xweights = None
        self.yweights = None
        self.xcweights = None
        self.ycweights = None
        self.mask_weighted = None
        self.weighted_pix_rad = weightedPixRad
        self.vec_wei_pix_rad_t = vecWeiPixRadT
        self.windowing = windowing
        # Per-subaperture threshold, as a fraction of the brightest pixel of each subaperture
        self.thr_ratio_value = thr_ratio_value
        self.thr_pedestal = False
        self.mult_factor = 0.0
        self.quadcell_mode = False

        self.exp_weight = exp_weight
        self.window_int_pixel = window_int_pixel
        self.window_int_threshold = window_int_threshold
        # Pixel weights, shape (n_subaps, np_sub*np_sub) like the subaperture pixels
        self._int_pixels_weight = None
        # Weights for the slope computation, see set_xy_weights()
        self._weights = None

        self.set_xy_weights()
        self.outputs['out_subapdata'] = self.subapdata

        self.slopes.layout = self.subapdata.slopes_layout()

    @classmethod
    def output_names(cls):
        result =super().output_names()
        result.update({ 
            'out_subapdata': OutputDesc(SubapData, 'Subaperture data with geometry information')         
        })
        return result

    def nsubaps(self):
        return self.subapdata.n_subaps

    def nslopes(self):
        return self.subapdata.n_subaps * 2

    @property
    def subap_idx(self):
        return self.subapdata.idxs

    @property
    def int_pixels_weight(self):
        '''Pixel weights, shape (np_sub*np_sub, n_subaps)'''
        if self._int_pixels_weight is None:
            return None
        return self._int_pixels_weight.T

    def setup(self):
        super().setup()
        # The CUDA graph is captured at the first trigger(), since the input
        # pixels are only available then. Derived classes must opt in, see
        # the class docstring.
        if type(self) is ShSlopec:
            self.build_stream(capture=False)

    def set_xy_weights(self):
        if self.subapdata:
            out = self.computeXYweights(self.subapdata.np_sub, self.exp_weight, self.weighted_pix_rad,
                                          self.quadcell_mode, self.windowing)
            self.mask_weighted = self.to_xp(out['mask_weighted'], dtype=self.dtype)
            self.xweights = self.to_xp(out['x'], dtype=self.dtype)
            self.yweights = self.to_xp(out['y'], dtype=self.dtype)
            self.xcweights = self.to_xp(out['xc'], dtype=self.dtype)
            self.ycweights = self.to_xp(out['yc'], dtype=self.dtype)
            # Denominator, x and y weights as rows of a single (3, np_sub*np_sub) array.
            # Updated in place, since its address is frozen in the CUDA graph.
            weights = self.xp.vstack([self.mask_weighted.ravel(), self.xweights.ravel(), self.yweights.ravel()])
            if self._weights is None:
                self._weights = weights
            else:
                self._weights[:] = weights

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

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        if self.vec_wei_pix_rad_t is not None:
            idxW = self.xp.where(self.current_time_seconds > self.vec_wei_pix_rad_t[:, 1])[0]
            if len(idxW) > 0:
                i_last = idxW[-1]
                weighted_pix_rad = self.xp.asarray(self.vec_wei_pix_rad_t[i_last, 0]).item()
                if weighted_pix_rad != self.weighted_pix_rad:
                    self.weighted_pix_rad = weighted_pix_rad
                    self.logger.debug(f'self.weighted_pix_rad: {self.weighted_pix_rad}')
                    self.set_xy_weights()

        if self.weight_int_pixel_dt > 0:
            self.do_accumulation(self.current_time)

        if self.weight_int_pixel:
            if self._int_pixels_weight is None:
                self._int_pixels_weight = self.xp.ones(self.subap_idx.shape, dtype=self.dtype)
            if self.int_pixels is not None and self.int_pixels.generation_time == self.current_time:
                self.update_int_pixels_weight()

    def update_int_pixels_weight(self):
        """
        Update the pixel weights from the accumulated pixels.
        Rows are subapertures, columns are the subaperture pixels.
        """
        n_weight_applied = 0
        int_pixels_weight = self.xp.take(self.int_pixels.pixels, self.subap_idx).astype(self.dtype)
        int_pixels_weight -= self.xp.min(int_pixels_weight, axis=1, keepdims=True)
        max_temp = self.xp.max(int_pixels_weight, axis=1)

        # Handle subapertures with zero or negative max values
        valid_mask = max_temp > 0

        if not self.xp.any(valid_mask):
            int_pixels_weight.fill(1.0)
        elif self.window_int_pixel:
            # Apply windowing condition exactly like IDL in 2D
            above_threshold = int_pixels_weight >= self.window_int_threshold

            # IDL: reverse(weight, 1) - flip only the pixel dimension
            weight_flipped = self.xp.flip(int_pixels_weight, axis=1)
            above_threshold_flipped = weight_flipped >= self.window_int_threshold

            # Combine with OR
            window_mask = above_threshold | above_threshold_flipped

            # Convert to weights
            int_pixels_weight = window_mask.astype(self.dtype)

            # Handle invalid subapertures
            int_pixels_weight[~valid_mask, :] = 1.0

            n_weight_applied = self.xp.sum(self.xp.any(int_pixels_weight > 0, axis=1))
        else:
            # Normalize by max value for valid subapertures
            int_pixels_weight[valid_mask, :] /= max_temp[valid_mask, None]
            int_pixels_weight[~valid_mask, :] = 1.0
            n_weight_applied = self.xp.sum(valid_mask)

        self._int_pixels_weight[:] = int_pixels_weight

        self.logger.debug(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

    def compute_slopes(self):
        self.calc_slopes_nofor()

    def calc_slopes_nofor(self):
        """
        Calculate slopes without a for-loop over subapertures.
        GPU operations only, so that it can be part of a CUDA graph.
        """
        if self.subapdata is None:
            self.logger.warning('subapdata is not valid.')
            return

        in_pixels = self.local_inputs['in_pixels'].pixels

        n_subaps = self.subapdata.n_subaps

        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError("Only one between _thr_value and _thr_ratio_value can be set.")

        # Subaperture pixels, shape (n_subaps, np_sub*np_sub)
        pixels = self.xp.take(in_pixels, self.subap_idx).astype(self.dtype, copy=False)

        if self.weight_int_pixel:
            # Weights are updated by prepare_trigger()
            pixels *= self._int_pixels_weight

        # Calculate flux per subaperture
        flux_per_subaperture_vector = self.flux_per_subaperture_vector.value
        self.xp.sum(pixels, axis=1, out=flux_per_subaperture_vector)

        # Thresholding logic
        if self.thr_ratio_value > 0:
            # One threshold per subaperture (row)
            thr = self.thr_ratio_value * self.xp.max(pixels, axis=1, keepdims=True)
        elif self.thr_pedestal or self.thr_value > 0:
            thr = self.thr_value
        else:
            thr = 0

        if self.thr_pedestal:
            clamp_generic_less(thr, 0, pixels, xp=self.xp)
        else:
            # In place, with ufuncs: on these large arrays they are faster than fused kernels
            if self.thr_ratio_value > 0 or thr != 0:
                self.xp.subtract(pixels, thr, out=pixels)
            if self.clamp_negative:
                self.xp.maximum(pixels, 0, out=pixels)

        # Denominator, x and y weighted sums, computed together
        subap_tot, sx_raw, sy_raw = sum_product(pixels[None, :, :], self._weights[:, None, :],
                                                xp=self.xp)
        mean_subap_tot = self.xp.mean(subap_tot)

        if self.mult_factor != 0:
            mult_factor = self.mult_factor
            self.logger.warning("multiplication factor in the slope computer!")
        else:
            mult_factor = 1.0

        # Write the slopes directly into the slopes vector, using views
        slopes = self.slopes.slopes
        if self.slopes.interleave:
            sx, sy = slopes[0::2], slopes[1::2]
        else:
            sx, sy = slopes[:n_subaps], slopes[n_subaps:]
        if self.xp is np:
            # Dark subapertures are expected, and set to zero
            with np.errstate(divide='ignore'):
                sh_slopes_normalize(subap_tot, sx_raw, sy_raw, mean_subap_tot, mult_factor,
                                    sx, sy, xp=self.xp)
        else:
            sh_slopes_normalize(subap_tot, sx_raw, sy_raw, mean_subap_tot, mult_factor,
                                sx, sy, xp=self.xp)

        self.xp.sum(flux_per_subaperture_vector, keepdims=True, out=self.total_counts.value)
        self.xp.mean(flux_per_subaperture_vector, keepdims=True, out=self.subap_counts.value)

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

        # Here and not in trigger_code(), since it needs a CPU-GPU synchronization
        if self.logger.isEnabledFor(logging.DEBUG):
            sx = self.slopes.xslopes
            self.logger.debug(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")
