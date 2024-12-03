
import numpy as np

from specula import fuse
from specula.lib import make_mask
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
                 corr_template = None,                
                 target_device_idx: int = None, 
                 precision: int = None ):
        super().__init__(sn=sn, target_device_idx=target_device_idx, precision=precision)
        self.thr_value = thr_value
        self.thr_mask_cube = BaseValue()  
        self.total_counts = BaseValue()
        self.subap_counts = BaseValue()
        self.exp_weight = None
        self.subapdata = None
        self.xweights = None
        self.yweights = None
        self.xcweights = None
        self.ycweights = None
        self.mask_weighted = None
        self.corr_template = corr_template
        self.winMatWindowed = None
        self.vecWeiPixRadT = None
        self.weightedPixRad = 0.0
        self.windowing = False
        self.correlation = False
        self.corrWindowSidePix = 0
        self.thr_ratio_value = 0.0
        self.thr_pedestal = False
        self.mult_factor = 0.0
        self.quadcell_mode = False
        self.two_steps_cog = False
        self.cog_2ndstep_size = 0
        self.dotemplate = False
        self.store_thr_mask_cube = False

        self.exp_weight = exp_weight
        self.subapdata = subapdata
        # TODO replace this resize with an earlier initialization
        self.slopes.resize(subapdata.n_subaps * 2)
        self.accumulated_slopes = Slopes(subapdata.n_subaps * 2)
        self.set_xy_weights()
        self.outputs['out_subapdata'] = self.subapdata

    @property
    def subap_idx(self):
        return self.subapdata.idxs
 
    def set_xy_weights(self):
        if self.subapdata:
            out = self.computeXYweights(self.subapdata.np_sub, self.exp_weight, self.weightedPixRad, 
                                          self.quadcell_mode, self.windowing)
            self.mask_weighted = self.xp.array(out['mask_weighted'], copy=False)
            self.xweights = self.xp.array(out['x'], copy=False)
            self.yweights = self.xp.array(out['y'], copy=False)
            self.xcweights = self.xp.array(out['xc'], copy=False)
            self.ycweights = self.xp.array(out['yc'], copy=False)

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
        x, y = np.meshgrid(np.linspace(-1, 1, np_sub), np.linspace(-1, 1, np_sub))
        
        # Compute weights in quadcell mode or otherwise
        if quadcell_mode:
            x = np.where(x > 0, 1.0, -1.0)
            y = np.where(y > 0, 1.0, -1.0)
            xc, yc = x, y
        else:
            xc, yc = x, y
            # Apply exponential weights if exp_weight is not 1
            x = np.where(x > 0, np.power(x, exp_weight), -np.power(np.abs(x), exp_weight))
            y = np.where(y > 0, np.power(y, exp_weight), -np.power(np.abs(y), exp_weight))

        # Adjust xc, yc for centroid calculations in two steps
        xc = np.where(x > 0, xc, -np.abs(xc))
        yc = np.where(y > 0, yc, -np.abs(yc))

        # Apply windowing or weighted pixel mask
        if weightedPixRad != 0:
            if windowing:
                # Windowing case (must be an integer)
                mask_weighted = make_mask(np_sub, diaratio=(2.0 * weightedPixRad / np_sub), xp=np)
            else:
                # Weighted Center of Gravity (WCoG)
                mask_weighted = self.psf_gaussian(np_sub, 2, [weightedPixRad, weightedPixRad])
                mask_weighted /= np.max(mask_weighted)

            mask_weighted[mask_weighted < 1e-6] = 0.0

            x *= mask_weighted.astype(self.dtype)
            y *= mask_weighted.astype(self.dtype)
        else:
            mask_weighted = np.ones((np_sub, np_sub), dtype=self.dtype)

        return {"x": x, "y": y, "xc": xc, "yc": yc, "mask_weighted": mask_weighted}

    # TODO what is this accumulated flag?
    def trigger_code(self, accumulated=False):
        if self.vecWeiPixRadT is not None:
            time = self.current_time_seconds
            idxW = self.xp.where(time > self.vecWeiPixRadT[:, 1])[-1]
            if len(idxW) > 0:
                self.weightedPixRad = self.vecWeiPixRadT[idxW, 0]
                if self.verbose:
                    print(f'self.weightedPixRad: {self.weightedPixRad}')
                self.set_xy_weights()

        if self.dotemplate or self.correlation or self.two_steps_cog:
            self.calc_slopes_for(accumulated=accumulated)
        else:
            self.calc_slopes_nofor(accumulated=accumulated)

    def calc_slopes_for(self, accumulated=False):
        """
        Calculate slopes using a for loop over subapertures.

        Parameters:
        t (float): The time for which to calculate the slopes.
        accumulated (bool): If True, use accumulated pixels for slope calculation.
        """
        if self.verbose and self.subapdata is None:
            print('subapdata is not valid.')
            return

        in_pixels = self.inputs['in_pixels'].get(self.target_device_idx).pixels
        
        n_subaps = self.subapdata.n_subaps
        np_sub = self.subapdata.np_sub
        pixels = self.accumulated_pixels.pixels if accumulated else in_pixels

        sx = self.xp.zeros(n_subaps, dtype=float)
        sy = self.xp.zeros(n_subaps, dtype=float)

        if self.store_thr_mask_cube:
            thr_mask_cube = self.xp.zeros((np_sub, np_sub, n_subaps), dtype=int)
            thr_mask = self.xp.zeros((np_sub, np_sub), dtype=int)

        flux_per_subaperture = self.xp.zeros(n_subaps, dtype=float)
        max_flux_per_subaperture = self.xp.zeros(n_subaps, dtype=float)

        if self.dotemplate:
            corr_template = self.xp.zeros((np_sub, np_sub, n_subaps), dtype=float)
        elif self.corr_template is not None:
            corr_template = self.corr_template

        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError('Only one between _thr_value and _thr_ratio_value can be set.')

        if self.weight_from_accumulated:
            n_weight_applied = 0

        for i in range(n_subaps):
            idx = self.subap_idx[i, :]
            subap = pixels[idx].reshape(np_sub, np_sub)

            if self.weight_from_accumulated and self.accumulated_pixels is not None and t >= self.accumulation_dt:
                accumulated_pixels_weight = self.accumulated_pixels[idx].reshape(np_sub, np_sub)
                accumulated_pixels_weight -= self.xp.min(accumulated_pixels_weight)
                max_temp = self.xp.max(accumulated_pixels_weight)
                if max_temp > 0:
                    if self.weightFromAccWithWindow:
                        window_threshold = 0.05
                        over_threshold = self.xp.where(
                            (accumulated_pixels_weight >= max_temp * window_threshold) | 
                            (self.xp.rot90(accumulated_pixels_weight, 2) >= max_temp * window_threshold)
                        )
                        if len(over_threshold[0]) > 0:
                            accumulated_pixels_weight.fill(0)
                            accumulated_pixels_weight[over_threshold] = 1.0
                        else:
                            accumulated_pixels_weight.fill(1.0)
                    else:
                        accumulated_pixels_weight *= 1.0 / max_temp

                    subap *= accumulated_pixels_weight
                    n_weight_applied += 1

            if self.winMatWindowed is not None:
                if i == 0 and self.verbose:
                    print("self.winMatWindowed applied")
                subap *= self.winMatWindowed[:, :, i]

            if self.dotemplate:
                corr_template[:, :, i] = subap

            flux_per_subaperture[i] = self.xp.sum(subap)
            max_flux_per_subaperture[i] = self.xp.max(subap)

            thr = 0
            if self.thr_value > 0:
                thr = self.thr_value
            if self.thr_ratio_value > 0:
                thr = self.thr_ratio_value * self.xp.max(subap)

            if self.thr_pedestal:
                thr_idx = self.xp.where(subap < thr)
            else:
                subap -= thr
                thr_idx = self.xp.where(subap < 0)

            if len(thr_idx[0]) > 0:
                subap[thr_idx] = 0

            if self.store_thr_mask_cube:
                thr_mask.fill(0)
                if len(thr_idx[0]) > 0:
                    thr_mask[thr_idx] = 1
                thr_mask_cube[:, :, i] = thr_mask

            if self.correlation:
                if self.corrWindowSidePix > 0:
                    subap = self.xp.convolve(
                        subap[np_sub // 2 - self.corrWindowSidePix // 2: np_sub // 2 + self.corrWindowSidePix // 2],
                        corr_template[np_sub // 2 - self.corrWindowSidePix // 2: np_sub // 2 + self.corrWindowSidePix // 2, i],
                        mode='same'
                    )
                else:
                    subap = self.xp.convolve(subap, corr_template[:, :, i], mode='same')
                thr_idx = self.xp.where(subap < 0)
                if len(thr_idx[0]) > 0:
                    subap[thr_idx] = 0

            # CoG in two steps logic (simplified here)
            if self.two_steps_cog:
                pass  # Further logic for two-step centroid calculation can go here.

            subap_total = self.xp.sum(subap)
            factor = 1.0 / subap_total if subap_total > 0 else 0

            sx[i] = self.xp.sum(subap * self.xweights) * factor
            sy[i] = self.xp.sum(subap * self.yweights) * factor

        if self.weight_from_accumulated:
            print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

        if self.dotemplate:
            self.corr_template = corr_template

        if self.mult_factor != 0:
            sx *= self.mult_factor
            sy *= self.mult_factor
            print("WARNING: multiplication factor in the slope computer!")

        if accumulated:
            self.accumulated_slopes.xslopes = sx
            self.accumulated_slopes.yslopes = sy
            self.accumulated_slopes.generation_time = self.current_time
        else:
            if self.store_thr_mask_cube:
                self.thr_mask_cube.value = thr_mask_cube
                self.thr_mask_cube.generation_time = self.current_time

            self.slopes.xslopes = sx
            self.slopes.yslopes = sy
            if self.slopes.pupdata_tag != self.subapdata.tag:
                self.slopes.pupdata_tag = self.subapdata.tag
            self.slopes.generation_time = self.current_time

            self.flux_per_subaperture_vector.value = flux_per_subaperture
            self.flux_per_subaperture_vector.generation_time = self.current_time
            self.total_counts.value = self.xp.sum(self.flux_per_subaperture_vector.value)
            self.total_counts.generation_time = self.current_time
            self.subap_counts.value = self.xp.mean(self.flux_per_subaperture_vector.value)
            self.subap_counts.generation_time = self.current_time

        if self.verbose:
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    
    def calc_slopes_nofor(self, accumulated=False):
        """
        Calculate slopes without a for-loop over subapertures.
        
        Parameters:
        t (float): The time for which to calculate the slopes.
        accumulated (bool): If True, use accumulated pixels for slope calculation.
        """
        if self.verbose and self.subapdata is None:
            print('subapdata is not valid.')
            return

        in_pixels = self.inputs['in_pixels'].get(self.target_device_idx).pixels

        n_subaps = self.subapdata.n_subaps
        np_sub = self.subapdata.np_sub
        orig_pixels = self.accumulated_pixels.pixels if accumulated else in_pixels

        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError("Only one between _thr_value and _thr_ratio_value can be set.")

        # Reform pixels based on the subaperture index
        idx2d = self.xp.unravel_index(self.subap_idx, orig_pixels.shape)
        pixels = orig_pixels[idx2d].T
        
        if self.weight_from_accumulated:
            raise NotImplementedError('weight_from_accumulated is not implemented')
        
            n_weight_applied = 0
            if self.accumulated_pixels is not None and t >= self.accumulation_dt:
                accumulated_pixels_weight = self.accumulated_pixels[self.subap_idx].T
                accumulated_pixels_weight -= self.xp.min(accumulated_pixels_weight, axis=1, keepdims=True)
                max_temp = self.xp.max(accumulated_pixels_weight, axis=1)
                idx0 = self.xp.where(max_temp <= 0)[0]
                if len(idx0) > 0:
                    accumulated_pixels_weight[:, idx0] = 1.0

                if self.weightFromAccWithWindow:
                    window_threshold = 0.05
                    one_over_max_temp = 1.0 / max_temp[:, self.xp.newaxis]
                    accumulated_pixels_weight *= one_over_max_temp
                    over_threshold = self.xp.where(
                        (accumulated_pixels_weight >= window_threshold) | 
                        (accumulated_pixels_weight[:, ::-1] >= window_threshold)
                    )
                    if len(over_threshold[0]) > 0:
                        accumulated_pixels_weight.fill(0)
                        accumulated_pixels_weight[over_threshold] = 1.0
                    else:
                        accumulated_pixels_weight.fill(1.0)
                    n_weight_applied += self.xp.sum(self.xp.any(accumulated_pixels_weight > 0, axis=1))

                pixels *= accumulated_pixels_weight

        # Calculate flux and max flux per subaperture
        flux_per_subaperture_vector = self.xp.sum(pixels, axis=0)
        max_flux_per_subaperture = self.xp.max(flux_per_subaperture_vector)

        if self.winMatWindowed is not None:
            if self.verbose:
                print("self.winMatWindowed applied")
            pixels *= self.winMatWindowed.reshape(np_sub * np_sub, n_subaps)

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
        subap_tot = self.xp.sum(pixels * self.mask_weighted.reshape(np_sub * np_sub, 1), axis=0)
        mean_subap_tot = self.xp.mean(subap_tot)
        factor = 1.0 / subap_tot

# TEST replacing these three lines with clamp_generic_more
#        idx_le_0 = self.xp.where(subap_tot <= mean_subap_tot * 1e-3)[0]
#        if len(idx_le_0) > 0:
#            factor[idx_le_0] = 0.0
        clamp_generic_more( 1.0 / (mean_subap_tot * 1e-3), 0, factor, xp=self.xp)

        # Compute slopes
        sx = self.xp.sum(pixels * self.xweights.reshape(np_sub * np_sub, 1) * factor[self.xp.newaxis, :], axis=0)
        sy = self.xp.sum(pixels * self.yweights.reshape(np_sub * np_sub, 1) * factor[self.xp.newaxis, :], axis=0)

        if self.weight_from_accumulated:
            print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")

        if self.mult_factor != 0:
            sx *= self.mult_factor
            sy *= self.mult_factor
            print("WARNING: multiplication factor in the slope computer!")

        if accumulated:
            self.accumulated_slopes.xslopes = sx
            self.accumulated_slopes.yslopes = sy
            self.accumulated_slopes.generation_time = self.current_time
        else:
            if self.store_thr_mask_cube:
                self.thr_mask_cube.value = thr_mask_cube
                self.thr_mask_cube.generation_time = self.current_time

            self.slopes.xslopes = sx
            self.slopes.yslopes = sy
            self.slopes.generation_time = self.current_time

            self.flux_per_subaperture_vector.value = flux_per_subaperture_vector
            self.flux_per_subaperture_vector.generation_time = self.current_time
            self.total_counts.value = self.xp.sum(self.flux_per_subaperture_vector.value)
            self.total_counts.generation_time = self.current_time
            self.subap_counts.value = self.xp.mean(self.flux_per_subaperture_vector.value)
            self.subap_counts.generation_time = self.current_time

        if self.verbose:
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    def psf_gaussian(self, np_sub, fwhm):
        x = np.linspace(-1, 1, np_sub)
        y = np.linspace(-1, 1, np_sub)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-4 * np.log(2) * (x ** 2 + y ** 2) / fwhm[0] ** 2, dtype=self.dtype)
        return gaussian
