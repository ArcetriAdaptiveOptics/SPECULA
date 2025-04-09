import numpy as np
import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.pupdata import PupData
from specula.data_objects.subap_data import SubapData
from specula.data_objects.slopes import Slopes

from specula.connections import InputValue
from specula import cpuArray


class SlopecDisplay(BaseProcessingObj):
    def __init__(self, window=27, disp_factor=1):
        super().__init__(target_device_idx=-1)

        self._disp_factor = disp_factor
        self._window = window
        self._title = ''
        self._opened = False
        self._first = True
        self.fig = self.ax = None
        self.inputs['slopes'] = InputValue(type=Slopes)
        self.inputs['pupdata'] = InputValue(type=PupData, optional=True)
        self.inputs['subapdata'] = InputValue(type=SubapData, optional=True)

    def set_w(self, size_frame):
        self.fig = plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
        self.ax = self.fig.add_subplot(111)

    def trigger_code(self):
        
        slopes_obj = self.local_inputs['slopes']
        pupdata = self.local_inputs['pupdata']
        subapdata = self.local_inputs['subapdata']

        if pupdata:
            frame3d = slopes_obj.get2d(cm=None, pupdata=pupdata)
        else:
            frame3d = slopes_obj.get2d(cm=None, pupdata=subapdata)

        frame2d = np.hstack(cpuArray(frame3d))
        title = self._title if self._title else 'Slope Display'

        if not self._opened:
            self.set_w(frame2d.shape)
            self._opened = True
        if self._first:
            self.ax.set_title(title)
            self.img = self.ax.imshow(frame2d)
            self._first = False
        else:
            self.img.set_data(frame2d)
            self.img.set_clim(frame2d.min(), frame2d.max())
        self.fig.canvas.draw()
        plt.pause(0.001)

