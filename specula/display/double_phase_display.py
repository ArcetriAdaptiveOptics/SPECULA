import numpy as np

from specula import xp
from specula import cpuArray

import matplotlib.pyplot as plt

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField

from symao.turbolence import ft_ft2

class DoublePhaseDisplay(BaseProcessingObj):
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
        self.inputs['phase1'] = InputValue(type=ElectricField)
        self.inputs['phase2'] = InputValue(type=ElectricField)
        self.nframes = 0
        self.psd_statTot1 = None
        self.psd_statTot2 = None

    def set_w(self, size_frame):
        self.fig = plt.figure(self._window, figsize=( 4 * size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))

        self.ax1 = self.fig.add_subplot(141)
        self.ax2 = self.fig.add_subplot(142)
        self.ax3 = self.fig.add_subplot(143)
        self.ax4 = self.fig.add_subplot(144)

#        plt.figure(self._window, figsize=(size_frame[0] * self._disp_factor / 100, size_frame[1] * self._disp_factor / 100))
#        plt.title(self._title)

    def trigger_code(self):
        
        self.nframes += 1

        phase1 = self.local_inputs['phase1']
        frame1 = cpuArray(phase1.phaseInNm * (phase1.A > 0).astype(float))
        idx = np.where(cpuArray(phase1.A) > 0)[0]
        frame1[idx] -= np.mean(frame1[idx])
        psd_stat1 = np.absolute(ft_ft2(frame1, 1))**2
        ss = frame1.shape[0]

        phase2 = self.local_inputs['phase2']
        frame2 = cpuArray(phase2.phaseInNm * (phase2.A > 0).astype(float))
        idx = np.where(cpuArray(phase2.A) > 0)[0]
        frame2[idx] -= np.mean(frame2[idx])
        psd_stat2 = np.absolute(ft_ft2(frame2, 1))**2
        ss = frame2.shape[0]

        if self.psd_statTot1 is None:
            self.psd_statTot1 = np.zeros_like(psd_stat1)
        if self.psd_statTot2 is None:
            self.psd_statTot2 = np.zeros_like(psd_stat2)
        self.psd_statTot1 = (self.psd_statTot1 * (self.nframes-1)  + psd_stat1)/self.nframes
        self.psd_statTot2 = (self.psd_statTot2 * (self.nframes-1)  + psd_stat2)/self.nframes

        if self._verbose:
            print('removing average phase in phase_display')

        if np.sum(self._size_frame) == 0:
            size_frame = frame1.shape
        else:
            size_frame = self._size_frame

        if not self._opened:
            self.set_w(2*size_frame)
            self._opened = True
        if self._first:
            self.img1 = self.ax1.imshow(frame1)
            self.img2 = self.ax2.imshow(frame2)
            self._first = False
        else:
            self.img1.set_data(frame1)
            self.img1.set_clim(frame1.min(), frame1.max())
            self.img2.set_data(frame2)
            self.img2.set_clim(frame2.min(), frame2.max())
        
        self.ax4.clear()
        
        self.ax3.loglog( psd_stat1[ss//2, ss//2+1:] , alpha=0.025, color ='r')
        self.ax3.loglog( psd_stat2[ss//2, ss//2+1:] , alpha=0.025, color ='b')

        self.ax4.loglog( self.psd_statTot1[ss//2, ss//2+1:] , color ='r')
        self.ax4.loglog( self.psd_statTot2[ss//2, ss//2+1:] , color ='b')

        self.fig.canvas.draw()

        
        plt.pause(0.001)

        # plt.figure(self._window)

        # if self._doImage:
        #     plt.imshow(frame, aspect='auto')
        # else:
        #     plt.imshow(np.repeat(np.repeat(frame, self._disp_factor, axis=0), self._disp_factor, axis=1), cmap='gray')
        # plt.draw()
        # plt.pause(0.01)

