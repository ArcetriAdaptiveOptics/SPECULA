import numpy as np

from specula import xp
from specula import cpuArray

import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


class PsfDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, wsize=[600, 600], window=25, title='PSF', log10=False):
        super().__init__()
        self._psf = None
        self._wsize = wsize
        self._window = window
        self._log = log10
        self._image_p2v = 0.0
        self._title = title
        self._opened = False
        self._first = True
        self._disp_factor = disp_factor
        self.inputs['psf'] = InputValue(type=BaseValue)

    def set_w(self):
#        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
#        plt.title(self._title)
        self.fig = plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        self.ax = self.fig.add_subplot(111)

    def trigger_code(self):
        psf = self.local_inputs['psf']
        image = cpuArray(psf.value)

        if self._image_p2v > 0:
            image = np.maximum(image, self._image_p2v**(-1.) * np.max(image))
        
        if self._log:
             image = np.log10(image)


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

