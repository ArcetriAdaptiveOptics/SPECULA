
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
                 target_device_idx: int = None,
                 precision: int = None):

        # Set subaperture data before initializing base class
        # because we need to know the number of subapertures
        self.subapdata = subapdata

        super().__init__(sn=sn, filtmat=filtmat, weight_int_pixel_dt=weight_int_pixel_dt,
                         target_device_idx=target_device_idx, precision=precision)
        self.thr_value = thr_value
        self.thr_mask_cube = BaseValue(target_device_idx=self.target_device_idx)
        self.xweights = None
        self.yweights = None
        self.xcweights = None
        self.ycweights = None
        self.mask_weighted = None
        self.vecWeiPixRadT = None
        self.weightedPixRad = weightedPixRad
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
            out = self.computeXYweights(self.subapdata.np_sub, self.exp_weight, self.weightedPixRad, 
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
                mask_weighted = self.psf_gaussian(np_sub, [weightedPixRad, weightedPixRad])
                mask_weighted /= np.max(mask_weighted)

            mask_weighted[mask_weighted < 1e-6] = 0.0

            x *= mask_weighted.astype(self.dtype)
            y *= mask_weighted.astype(self.dtype)
        else:
            mask_weighted = np.ones((np_sub, np_sub), dtype=self.dtype)

        return {"x": x, "y": y, "xc": xc, "yc": yc, "mask_weighted": mask_weighted}

    def trigger_code(self):
        if self.vecWeiPixRadT is not None:
            time = self.current_time_seconds
            idxW = self.xp.where(time > self.vecWeiPixRadT[:, 1])[-1]
            if len(idxW) > 0:
                self.weightedPixRad = self.vecWeiPixRadT[idxW, 0]
                if self.verbose:
                    print(f'self.weightedPixRad: {self.weightedPixRad}')
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
                int_pixels_weight = self.int_pixels.pixels[idx2d].T
                int_pixels_weight -= self.xp.min(int_pixels_weight, axis=0, keepdims=True)
                max_temp = self.xp.max(int_pixels_weight, axis=0)

                # Handle subapertures with zero or negative max values
                valid_mask = max_temp > 0

                if not self.xp.any(valid_mask):
                    int_pixels_weight.fill(1.0)
                elif self.window_int_pixel:
                    window_threshold = 0.05
                    # Create a mask for pixels above threshold
                    normalized_weight = self.xp.zeros_like(int_pixels_weight)
                    normalized_weight[:, valid_mask] = int_pixels_weight[:, valid_mask] / max_temp[valid_mask]

                    # Initialize over_threshold mask
                    over_threshold = self.xp.zeros_like(normalized_weight, dtype=bool)

                    # Apply threshold and symmetry condition per subaperture
                    for i in range(self.subapdata.n_subaps):
                        if valid_mask[i]:  # Only process valid subapertures
                            # Extract pixels for the i-th subaperture
                            subap_weights = normalized_weight[:, i].reshape(self.subapdata.np_sub, self.subapdata.np_sub)

                            # Apply symmetry condition
                            subap_weights_flipped = subap_weights[::-1, ::-1]
                            symmetry_mask = (subap_weights >= window_threshold) | (subap_weights_flipped >= window_threshold)

                            # Update the mask for this subaperture
                            over_threshold[:, i] = symmetry_mask.flatten()

                            # PLOT DEBUGGING
                            plot_debug = False
                            if plot_debug:  # pragma: no cover
                                import matplotlib.pyplot as plt
                                from specula import cpuArray
                                plt.figure()
                                plt.imshow(cpuArray(subap_weights))
                                plt.figure()
                                plt.imshow(cpuArray(subap_weights_flipped))
                                plt.figure()
                                plt.imshow(cpuArray(symmetry_mask))
                                plt.show()

                    # Reset weights and apply threshold mask
                    int_pixels_weight.fill(0)
                    int_pixels_weight[over_threshold] = 1.0

                    # Count subapertures where weights were applied
                    n_weight_applied = self.xp.sum(self.xp.any(int_pixels_weight > 0, axis=0))
                else:
                    # Normalize by max value for valid subapertures
                    int_pixels_weight[:, valid_mask] /= max_temp[valid_mask]
                    int_pixels_weight[:, ~valid_mask] = 1.0
                    n_weight_applied = self.xp.sum(valid_mask)

                self.int_pixels_weight[:] = int_pixels_weight.astype(self.dtype)

            # Apply weights to pixels
            pixels *= self.int_pixels_weight

            if self.verbose:  # pragma: no cover
                print(f"Weights mask has been applied to {n_weight_applied} sub-apertures")
                
            # PLOT DEBUGGING
            plot_debug = False
            if plot_debug and self.int_pixels is not None and \
                self.int_pixels.generation_time == self.current_time:  # pragma: no cover
                import matplotlib.pyplot as plt
                from specula import cpuArray

                # 1. Plot diretto di int_pixels.pixels (già 2D)
                int_pixels_cpu = cpuArray(self.int_pixels.pixels)

                # 2. Per int_pixels_weight, usa la stessa logica di come vengono estratti i pixel
                # Crea una copia dell'immagine originale
                int_pixels_weight_2d = cpuArray(self.int_pixels.pixels)  # Start with original shape
                int_pixels_weight_cpu = cpuArray(int_pixels_weight)

                # Reset to zeros and fill with weights
                int_pixels_weight_2d.fill(0)

                # Usa la stessa logica di estrazione pixel per riempire i pesi
                for i in range(self.subapdata.n_subaps):
                    # Ottieni gli indici per la subapertura i-esima
                    subap_indices = cpuArray(self.subap_idx[i])  # Indici dei pixel per questa subapertura

                    # Ottieni i pesi per questa subapertura e convertili in 2D
                    subap_weights = int_pixels_weight_cpu[:, i].reshape(self.subapdata.np_sub, self.subapdata.np_sub)

                    # Converti gli indici lineari in coordinate 2D
                    rows = subap_indices // int_pixels_weight_2d.shape[1]
                    cols = subap_indices % int_pixels_weight_2d.shape[1]

                    # Assegna i pesi usando gli indici
                    for pixel_idx, (r, c) in enumerate(zip(rows, cols)):
                        pixel_r = pixel_idx // self.subapdata.np_sub
                        pixel_c = pixel_idx % self.subapdata.np_sub
                        int_pixels_weight_2d[r, c] = subap_weights[pixel_r, pixel_c]

                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Plot 1: int_pixels.pixels
                im1 = ax1.imshow(int_pixels_cpu, cmap='viridis', origin='lower')
                ax1.set_title(f'int_pixels.pixels (time={self.current_time})')
                ax1.set_xlabel('X pixel')
                ax1.set_ylabel('Y pixel')
                plt.colorbar(im1, ax=ax1)

                # Plot 2: int_pixels_weight (reshaped to 2D)
                im2 = ax2.imshow(int_pixels_weight_2d, cmap='plasma', origin='lower')
                ax2.set_title(f'int_pixels_weight ({"windowed" if self.window_int_pixel else "normalized"})')
                ax2.set_xlabel('X pixel')
                ax2.set_ylabel('Y pixel')
                plt.colorbar(im2, ax=ax2)

                plt.tight_layout()
                plt.savefig(f'int_pixels_debug_t{self.current_time}.png', dpi=150, bbox_inches='tight')
                plt.show()

                # Statistics
                print(f"int_pixels stats: min={int_pixels_cpu.min():.3f}, max={int_pixels_cpu.max():.3f}, mean={int_pixels_cpu.mean():.3f}")
                print(f"int_pixels_weight stats: min={int_pixels_weight_2d.min():.3f}, max={int_pixels_weight_2d.max():.3f}")
                print(f"Number of subapertures with applied weights: {n_weight_applied}")


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

        if self.verbose:
            print(f"Slopes min, max and rms : {self.xp.min(sx)}, {self.xp.max(sx)}, {self.xp.sqrt(self.xp.mean(sx ** 2))}")

    def psf_gaussian(self, np_sub, fwhm):
        x = np.linspace(-1, 1, np_sub)
        y = np.linspace(-1, 1, np_sub)
        x, y = np.meshgrid(x, y)
        gaussian = np.exp(-4 * np.log(2) * (x ** 2 + y ** 2) / fwhm[0] ** 2, dtype=self.dtype)
        return gaussian

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time