
import numpy as np
import matplotlib.pyplot as plt

from specula import xp
from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData


class PixelsDisplay(BaseDisplay):
    def __init__(self,
                 title='Pixels Display',
                 figsize=(6, 6),
                 sh_as_pyr=False, 
                 subapdata: SubapData = None,
                 log_scale=False):

        super().__init__(
            title=title,
            figsize=figsize
        )

        self._sh_as_pyr = sh_as_pyr
        self._subapdata = subapdata
        self._log_scale = log_scale

        # Setup input
        self.input_key = 'pixels'  # Used by base class
        self.inputs['pixels'] = InputValue(type=Pixels)

    def _update_display(self, pixels):
        """Override base method to implement pixels-specific display"""
        # Process image data
        image = cpuArray(pixels.pixels)

        if self._sh_as_pyr and self._subapdata is not None:
            image = self._reformat_as_pyramid(image, self._subapdata)

        if self._log_scale:
            # Avoid log(0) by adding small epsilon
            image = np.log10(np.maximum(image, 1e-10))

        if self.img is None:
            self.img = self.ax.imshow(image)

            if not self._colorbar_added:
                plt.colorbar(self.img, ax=self.ax)
                self._colorbar_added = True
        else:
            self.img.set_data(image)
            self.img.set_clim(image.min(), image.max())

        self._safe_draw()

    def reformat_as_pyramid(self, pixels, subapdata):    
        pupil = subapdata.copyTo(-1).single_mask()
        idx2d = xp.unravel_index(subapdata.idxs, pixels.shape)
        A, B, C, D = pupil.copy(), pupil.copy(), pupil.copy(), pupil.copy()
        pix_idx = subapdata.display_map
        half_sub = subapdata.np_sub // 2
        for i in range(subapdata.n_subaps):
            subap = pixels[idx2d[0][i], idx2d[1][i]].reshape(half_sub*2, half_sub*2)
            A.flat[pix_idx[i]] = subap[:half_sub, :half_sub].sum()
            B.flat[pix_idx[i]] = subap[:half_sub, half_sub:].sum()
            C.flat[pix_idx[i]] = subap[half_sub:, :half_sub].sum()
            D.flat[pix_idx[i]] = subap[half_sub:, half_sub:].sum()
   
        pyr_pixels = np.vstack((np.hstack((A, B)), np.hstack((C, D))))
        return pyr_pixels
