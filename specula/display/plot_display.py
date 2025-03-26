import numpy as np

from specula import xp

import matplotlib.pyplot as plt
plt.ion()

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.connections import InputList
from specula.base_value import BaseValue


class PlotDisplay(BaseProcessingObj):
    def __init__(self, disp_factor=1, histlen=200, wsize=(600, 400), window=23, yrange=(0, 0), oplot=False, color=1, psym=-4, title=''):
        super().__init__()
        
        self._wsize = wsize
        self._window = window
        self._history = np.zeros(histlen)
        self._count = 0
        self._yrange = yrange
        self._value = None
        self._oplot = oplot
        self._color = color
        self._psym = psym
        self._title = title
        self._opened = False
        self._first = True
        self.line = []
        self._disp_factor = disp_factor
        self.inputs['value'] = InputValue(type=BaseValue,optional=True)
        self.inputs['value_list'] = InputList(type=BaseValue,optional=True)

    def set_w(self):
        self.fig = plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
        self.ax = self.fig.add_subplot(111)
#        plt.figure(self._window, figsize=(self._wsize[0] / 100, self._wsize[1] / 100))
#        plt.title(self._title)

    def trigger(self):
        if not self._opened:
            self.set_w()
            self._opened = True

        # Unify the cases by creating a list with a single element if value_list is empty
        if len(self.local_inputs['value_list']) > 0:
            value_list = self.local_inputs['value_list']
        else:
            value_list = [self.local_inputs['value']]

        nValues = len(value_list)
        n = self._history.shape[0]

        if self._history.ndim == 1:
            self._history = np.zeros((n, nValues))

        if self._count >= n:
            self._history[:-1, :] = self._history[1:, :]
            self._count = n - 1

        x = np.arange(self._count)

        plt.figure(self._window)

        if self._first:
            self.fig.suptitle(self._title)

        xmin = np.zeros(nValues)
        xmax = np.zeros(nValues)
        ymin = np.zeros(nValues)
        ymax = np.zeros(nValues)

        for i in range(nValues):
            v = value_list[i]
            self._history[self._count, i] = v.value
            y = self._history[:self._count, i]

            if self._first:
                self.line.append(self.ax.plot(x, y, marker='.'))
            else:
                self.line[i][0].set_xdata(x)
                self.line[i][0].set_ydata(y)
                xmin[i] = x.min()
                xmax[i] = x.max()
                ymin[i] = y.min()
                ymax[i] = y.max()

        if self._first:
            self._first = False
        else:
            self.ax.set_xlim(np.min(xmin), np.max(xmax))
            if np.sum(np.abs(self._yrange)) > 0:
                self.ax.set_ylim(self._yrange[0], self._yrange[1])
            else:
                self.ax.set_ylim(np.min(ymin), np.max(ymax))

        self.ax.axhline(y=0, color='grey', linestyle='--', dashes=(4, 8), linewidth=0.5)  # add a horizontal line at 0
        self.fig.canvas.draw()
        plt.pause(0.001)
        self._count += 1
        # if self._oplot:
        #     plt.plot(xp.arange(self._count), self._history[:self._count], marker='.', color=self._color)
        # else:
        #     plt.plot(xp.arange(self._count), self._history[:self._count], marker='.')
        #     plt.ylim(self._yrange)
        #     plt.title(self._title)
        # plt.draw()
        # plt.pause(0.01)


