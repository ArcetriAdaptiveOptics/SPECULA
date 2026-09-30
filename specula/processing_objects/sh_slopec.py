import logging

import numpy as np

from specula import cp
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.lib.make_mask import make_mask
from specula.lib.make_xy import make_xy

from specula.processing_objects.slopec import Slopec

# Threshold modes of the sh_subap_sums kernel
THR_SUBTRACT = 0    # pixels -= thr, then negative pixels are set to zero
THR_PEDESTAL = 1    # pixels below thr are set to zero
THR_RATIO = 2       # like THR_SUBTRACT, with thr = thr_ratio * subaperture max

# GPU kernels computing the slopes in two launches, so that the per-subaperture
# pixel cube is never allocated. Reductions are done in a fixed order,
# so that results are reproducible.
_sh_kernels_src = r'''
#define BLOCK 256

// Tree reduction of N values over the block, result in v[] of all threads
template<int N, typename T, bool MAX>
__device__ void block_reduce(T (&v)[N], T (*sh)[BLOCK]) {
    int tid = threadIdx.x;
    for (int k = 0; k < N; k++)
        sh[k][tid] = v[k];
    __syncthreads();
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s)
            for (int k = 0; k < N; k++)
                sh[k][tid] = MAX ? max(sh[k][tid], sh[k][tid + s]) : sh[k][tid] + sh[k][tid + s];
        __syncthreads();
    }
    for (int k = 0; k < N; k++)
        v[k] = sh[k][0];
    __syncthreads();
}

// One block per subaperture. For subaperture s, writes into sums[s, :]:
// flux (before thresholding), and the sums of the thresholded pixels
// weighted by the three rows of weights (denominator, x and y).
template<typename F, typename T>
__global__ void sh_subap_sums(const F *frame, const long long *idx,
                              const T *pix_weight, int use_pix_weight,
                              const T *weights, int np2,
                              T thr_value, T thr_ratio, int thr_mode, T *sums) {
    __shared__ T sh[3][BLOCK];
    const long long s = blockIdx.x;
    const long long *sidx = idx + s * np2;
    const T *spw = pix_weight + s * np2;

    T flux[1] = {0};
    T vmax[1] = {-3.0e38};
    for (int p = threadIdx.x; p < np2; p += BLOCK) {
        T v = T(frame[sidx[p]]);
        if (use_pix_weight) v *= spw[p];
        flux[0] += v;
        vmax[0] = max(vmax[0], v);
    }
    block_reduce<1, T, false>(flux, sh);

    T thr = thr_value;
    if (thr_mode == 2) {
        block_reduce<1, T, true>(vmax, sh);
        thr = thr_ratio * vmax[0];
    }

    T acc[3] = {0, 0, 0};
    for (int p = threadIdx.x; p < np2; p += BLOCK) {
        T v = T(frame[sidx[p]]);
        if (use_pix_weight) v *= spw[p];
        if (thr_mode == 1) {
            v = v < thr ? T(0) : v;
        } else {
            v -= thr;
            v = v < 0 ? T(0) : v;
        }
        for (int k = 0; k < 3; k++)
            acc[k] += v * weights[k * np2 + p];
    }
    block_reduce<3, T, false>(acc, sh);

    if (threadIdx.x == 0) {
        sums[s * 4 + 0] = flux[0];
        sums[s * 4 + 1] = acc[0];
        sums[s * 4 + 2] = acc[1];
        sums[s * 4 + 3] = acc[2];
    }
}

// Single block. Normalizes the slopes by the subaperture denominator, setting
// to zero those with a denominator below 1e-3 times the average. Slopes are
// written at sx[i * stride] and sy[i * stride]. Also writes the flux outputs.
template<typename T>
__global__ void sh_slopes_normalize(const T *sums, int n_subaps, T mult_factor,
                                    T *sx, T *sy, int stride,
                                    T *flux, T *total_counts, T *subap_counts) {
    __shared__ T sh[2][BLOCK];
    T tot[2] = {0, 0};
    for (int i = threadIdx.x; i < n_subaps; i += BLOCK) {
        tot[0] += sums[i * 4 + 0];
        tot[1] += sums[i * 4 + 1];
    }
    block_reduce<2, T, false>(tot, sh);

    T mean_subap_tot = tot[1] / T(n_subaps);
    T max_factor = T(1) / (mean_subap_tot * T(1e-3));
    for (int i = threadIdx.x; i < n_subaps; i += BLOCK) {
        T factor = T(1) / sums[i * 4 + 1];
        factor = factor > max_factor ? T(0) : factor;
        factor *= mult_factor;
        sx[i * stride] = sums[i * 4 + 2] * factor;
        sy[i * stride] = sums[i * 4 + 3] * factor;
        flux[i] = sums[i * 4 + 0];
    }
    if (threadIdx.x == 0) {
        total_counts[0] = tot[0];
        subap_counts[0] = tot[0] / T(n_subaps);
    }
}
'''
_SH_BLOCK = 256

_ctypes = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double',
           np.dtype(np.int16): 'short', np.dtype(np.uint16): 'unsigned short',
           np.dtype(np.int32): 'int', np.dtype(np.uint32): 'unsigned int',
           np.dtype(np.int64): 'long long', np.dtype(np.uint64): 'unsigned long long'}
_sh_kernels = {}


def _sh_kernel(name, *dtypes):
    '''
    Kernel *name* for the given template dtypes, compiled at first use.
    Kernels are loaded separately for each device.
    '''
    expr = f"{name}<{', '.join(_ctypes[np.dtype(d)] for d in dtypes)}>"
    key = (expr, cp.cuda.Device().id)
    if key not in _sh_kernels:
        module = cp.RawModule(code=_sh_kernels_src, options=('-std=c++14',), name_expressions=[expr])
        _sh_kernels[key] = module.get_function(expr)
    return _sh_kernels[key]


class ShSlopec(Slopec):
    """
    Shack-Hartmann slopes computer processing object.
    Computes Shack-Hartmann slopes from pixel data using the subaperture intensities.

    On GPU, trigger_code() is captured in a CUDA graph (see setup()), which
    also includes the slope corrections of the base class (slope null,
    filtering, slopes map). The pixel accumulation for weight_int_pixel_dt
    and the other operations that change from step to step, or that need
    a CPU-GPU synchronization, are done in prepare_trigger() instead.
    Scalar parameters (thr_value, thr_ratio_value, thr_pedestal, mult_factor)
    are frozen in the graph: call invalidate_graph() after changing them.
    """

    corrections_in_trigger = True

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
                 target_device_idx: int = None,
                 precision: int = None):

        # Set subaperture data before initializing base class
        # because we need to know the number of subapertures
        self.subapdata = subapdata

        super().__init__(sn=sn,
                         filtmat=filtmat,
                         weight_int_pixel_dt=weight_int_pixel_dt,
                         interleave=interleave,
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
        self.two_steps_cog = False
        self.cog_2ndstep_size = 0

        self.exp_weight = exp_weight
        self.window_int_pixel = window_int_pixel
        self.window_int_threshold = window_int_threshold
        # Pixel weights, shape (n_subaps, np_sub*np_sub) like the subaperture pixels
        self._int_pixels_weight = None
        # Weights for the slope computation, see set_xy_weights()
        self._weights = None

        self.accumulated_slopes = Slopes(self.nslopes(), target_device_idx=self.target_device_idx)
        self.set_xy_weights()
        self.outputs['out_subapdata'] = self.subapdata

        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

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
        # pixels are only available then
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
            self.xweights_flat = self.xweights.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)
            self.yweights_flat = self.yweights.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)
            self.mask_weighted_flat = self.mask_weighted.reshape(self.subapdata.np_sub * self.subapdata.np_sub, 1)
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

    def trigger_code(self):
        self.calc_slopes_nofor()
        self.apply_slopes_corrections()

    def calc_slopes_nofor(self):
        """
        Calculate slopes without a for-loop over subapertures.
        GPU operations only, so that it can be part of a CUDA graph.
        """
        if self.subapdata is None:
            self.logger.warning('subapdata is not valid.')
            return

        in_pixels = self.local_inputs['in_pixels'].pixels

        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError("Only one between _thr_value and _thr_ratio_value can be set.")

        # Thresholding logic
        if self.thr_ratio_value > 0:
            # One threshold per subaperture, as a fraction of its brightest pixel
            thr_mode, thr = THR_RATIO, 0
        elif self.thr_pedestal:
            thr_mode, thr = THR_PEDESTAL, self.thr_value
        elif self.thr_value > 0:
            thr_mode, thr = THR_SUBTRACT, self.thr_value
        else:
            thr_mode, thr = THR_SUBTRACT, 0

        if self.mult_factor != 0:
            mult_factor = self.mult_factor
            self.logger.warning("multiplication factor in the slope computer!")
        else:
            mult_factor = 1.0

        # Write the slopes directly into the slopes vector, using views
        n_subaps = self.subapdata.n_subaps
        slopes = self.slopes.slopes
        if self.slopes.interleave:
            sx, sy = slopes[0::2], slopes[1::2]
        else:
            sx, sy = slopes[:n_subaps], slopes[n_subaps:]

        if self.target_device_idx >= 0:
            self._calc_slopes_gpu(in_pixels, thr_mode, thr, mult_factor, sx, sy)
        else:
            self._calc_slopes_cpu(in_pixels, thr_mode, thr, mult_factor, sx, sy)

    def _calc_slopes_gpu(self, in_pixels, thr_mode, thr, mult_factor, sx, sy):
        """GPU version of calc_slopes_nofor(), with two kernel launches"""
        n_subaps = self.subapdata.n_subaps
        np2 = self._weights.shape[1]
        if not in_pixels.flags.c_contiguous:
            in_pixels = self.xp.ascontiguousarray(in_pixels)

        # Columns are flux, denominator, x and y weighted sums
        sums = self.xp.empty((n_subaps, 4), dtype=self.dtype)
        if self.weight_int_pixel:
            pix_weight = self._int_pixels_weight    # Updated by prepare_trigger()
        else:
            pix_weight = self._weights              # Not used
        subap_sums = _sh_kernel('sh_subap_sums', in_pixels.dtype, self.dtype)
        subap_sums((n_subaps,), (_SH_BLOCK,),
                   (in_pixels, self.subap_idx, pix_weight, np.int32(self.weight_int_pixel),
                    self._weights, np.int32(np2),
                    self.dtype(thr), self.dtype(self.thr_ratio_value), np.int32(thr_mode), sums))

        stride = 2 if self.slopes.interleave else 1
        normalize = _sh_kernel('sh_slopes_normalize', self.dtype)
        normalize((1,), (_SH_BLOCK,),
                  (sums, np.int32(n_subaps), self.dtype(mult_factor), sx, sy, np.int32(stride),
                   self.flux_per_subaperture_vector.value, self.total_counts.value,
                   self.subap_counts.value))

    def _calc_slopes_cpu(self, in_pixels, thr_mode, thr, mult_factor, sx, sy):
        """CPU version of calc_slopes_nofor(), same algorithm as the GPU kernels"""
        # Subaperture pixels, shape (n_subaps, np_sub*np_sub)
        pixels = np.take(in_pixels, self.subap_idx).astype(self.dtype)

        if self.weight_int_pixel:
            # Weights are updated by prepare_trigger()
            pixels *= self._int_pixels_weight

        flux_per_subaperture_vector = pixels.sum(axis=1)

        if thr_mode == THR_RATIO:
            thr = self.thr_ratio_value * pixels.max(axis=1, keepdims=True)

        if thr_mode == THR_PEDESTAL:
            pixels[pixels < thr] = 0
        else:
            pixels -= thr
            pixels[pixels < 0] = 0

        # Denominator, x and y weighted sums
        subap_tot, sx_raw, sy_raw = self._weights @ pixels.T

        # Subapertures with too little flux get zero slopes
        with np.errstate(divide='ignore'):
            factor = 1.0 / subap_tot
        factor[factor > 1.0 / (np.mean(subap_tot) * 1e-3)] = 0
        factor *= mult_factor
        sx[:] = sx_raw * factor
        sy[:] = sy_raw * factor

        self.flux_per_subaperture_vector.value[:] = flux_per_subaperture_vector
        self.total_counts.value[0] = np.sum(flux_per_subaperture_vector)
        self.subap_counts.value[0] = np.mean(flux_per_subaperture_vector)

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
