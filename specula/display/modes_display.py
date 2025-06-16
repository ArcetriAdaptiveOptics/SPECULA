import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue


class ModesDisplay(BaseProcessingObj):
    def __init__(self, wsize=(600, 300), window=22, yrange=(-500, 500), oplot=False, color=1, title=''):
        super().__init__(target_device_idx=-1)

        self._wsize = wsize
        self._window = window
        self._yrange = yrange
        self._oplot = oplot
        self._color = color
        self._title = title
        self._opened = False
        self._first = True
        self.inputs['modes'] = InputValue(type=BaseValue)

    def set_w(self):
        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        plt.title(self._title if self._title != '' else 'modes')

    def trigger_code(self):
        m = self.local_inputs['modes']
        if not self._opened and not self._oplot:
            self.set_w()
            self._opened = True

        plt.figure(self._window)
        if self._first:
            self._line = plt.plot(m.value, '.-')
            plt.title(self._title)
            plt.ylim(self._yrange)
            self._first = False
        else:
            self._line[0].set_ydata(m.value)
        plt.draw()
        plt.pause(0.01)
