import numpy as np

from specula import xp
from specula import cpuArray

import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField


class PhaseDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, doImage=False, window=24, title='phase'):
        super().__init__()

        self._phase = None
        self._doImage = doImage
        self._window = window
        self._disp_factor = disp_factor
        self._title = title
        self._opened = False
        self._size_frame = (0, 0)
        self._first = True
        self._disp_factor = disp_factor
        self.inputs['phase'] = InputValue(type=ElectricField)

    def set_w(self, size_frame):
        self.fig = plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
        self.ax = self.fig.add_subplot(111)
#        plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
#        plt.title(self._title)

    def trigger_code(self):
        phase = self.local_inputs['phase']

        frame = cpuArray(phase.phaseInNm * (phase.A > 0).astype(float))
        idx = np.where(cpuArray(phase.A) > 0)[0]
        frame[idx] -= np.mean(frame[idx])

        if self._verbose:
            print('removing average phase in phase_display')

        if np.sum(self._size_frame) == 0:
            size_frame = frame.shape
        else:
            size_frame = self._size_frame

        if not self._opened:
            self.set_w(size_frame)
            self._opened = True
        if self._first:
            self.img = self.ax.imshow(frame)
            self._first = False
        else:
            self.img.set_data(frame)
            self.img.set_clim(frame.min(), frame.max())
        self.fig.canvas.draw()
        plt.pause(0.001)

        # plt.figure(self._window)

        # if self._doImage:
        #     plt.imshow(frame, aspect='auto')
        # else:
        #     plt.imshow(np.repeat(np.repeat(frame, self._disp_factor, axis=0), self._disp_factor, axis=1), cmap='gray')
        # plt.draw()
        # plt.pause(0.01)

