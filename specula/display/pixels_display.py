
import numpy as np
from specula import xp
from specula import cpuArray

import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData


class PixelsDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, wsize=[600, 600], window=25, title='Pixels', sh_as_pyr=False, subapdata: SubapData =None):
        super().__init__()
        self._psf = None
        self._wsize = wsize
        self._window = window
        self._log = False
        self._title = title
        self._opened = False
        self._first = True
        self._disp_factor = disp_factor
        self._sh_as_pyr = sh_as_pyr
        self._subapdata = subapdata
        self.inputs['pixels'] = InputValue(type=Pixels)

    def set_w(self):
        self.fig = plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        self.ax = self.fig.add_subplot(111)

    def trigger_code(self):
        pixels = self.local_inputs['pixels']
        image = cpuArray(pixels.pixels)

        if self._sh_as_pyr:
            image = self.reformat_as_pyramid(image, self._subapdata)
        if self._log:
            image = xp.log10(image)

        if not self._opened:
            self.set_w()
            self._opened = True
        if self._first:
            self.img = self.ax.imshow(image)
            self._first = False
        else:
            self.img.set_data(image)
            self.img.set_clim(image.min(), image.max())
#            plt.colorbar()
        self.fig.canvas.draw()
        plt.pause(0.001)
        
    def reformat_as_pyramid(self, pixels, subapdata):
        pupil = subapdata.copyTo(-1).single_mask()
        idx2d = self.xp.unravel_index(subapdata.idxs, pixels.shape)
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
