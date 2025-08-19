import numpy as np

from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField

from symao.turbolence import ft_ft2


class DoublePhaseDisplay(BaseDisplay):
    def __init__(self,
                 title='Double Phase Display',
                 figsize=(12, 3)):  # 4 subplots side by side

        super().__init__(
            title=title,
            figsize=figsize
        )

        self.img1 = None
        self.img2 = None
        self.nframes = 0
        self.psd_statTot1 = None
        self.psd_statTot2 = None

        # Setup inputs - two phase inputs
        self.inputs['phase1'] = InputValue(type=ElectricField)
        self.inputs['phase2'] = InputValue(type=ElectricField)

    def reset(self):
        """Reset the display"""
        if self._opened:
            self.ax.clear()
            self._safe_draw()
        self.img1 = None
        self.img2 = None
        self.nframes = 0
        self.psd_statTot1 = None
        self.psd_statTot2 = None
        self._colorbar_added = False

    def _process_phase_data(self, phase):
        """Process phase data: mask and remove average"""
        frame = cpuArray(phase.phaseInNm * (phase.A > 0).astype(float))

        # Get valid indices (where amplitude > 0)
        valid_mask = cpuArray(phase.A) > 0

        if np.any(valid_mask):
            # Remove average phase only from valid pixels
            frame[valid_mask] -= np.mean(frame[valid_mask])

            if self._verbose:
                print('Removing average phase in double_phase_display')

        return frame

    def _calculate_psd(self, frame):
        """Calculate power spectral density"""
        return np.absolute(ft_ft2(frame, 1))**2

    def _get_data(self):
        """Get both phases"""
        phase1 = self.local_inputs.get('phase1')
        phase2 = self.local_inputs.get('phase2')

        if phase1 is None or phase2 is None:
            return []  # BaseDisplay will show error

        return [phase1, phase2]

    def _update_display(self, data_list):
        """Override base method - now receives list of [phase1, phase2]"""
        if len(data_list) != 2:
            self._show_error("Need both phase1 and phase2 inputs")
            return

        phase1, phase2 = data_list

        # Process both phases
        frame1 = self._process_phase_data(phase1)
        frame2 = self._process_phase_data(phase2)

        # Calculate PSDs
        psd_stat1 = self._calculate_psd(frame1)
        psd_stat2 = self._calculate_psd(frame2)
        ss = frame1.shape[0]

        # Update frame counter and accumulate PSDs
        self.nframes += 1

        if self.psd_statTot1 is None:
            self.psd_statTot1 = np.zeros_like(psd_stat1)
        if self.psd_statTot2 is None:
            self.psd_statTot2 = np.zeros_like(psd_stat2)

        self.psd_statTot1 = (self.psd_statTot1 * (self.nframes-1) + psd_stat1) / self.nframes
        self.psd_statTot2 = (self.psd_statTot2 * (self.nframes-1) + psd_stat2) / self.nframes

        # Create subplots on first run
        if self.img1 is None:
            # Clear default axis and create 4 subplots
            self.fig.clear()
            self.ax1 = self.fig.add_subplot(141)
            self.ax2 = self.fig.add_subplot(142)
            self.ax3 = self.fig.add_subplot(143)
            self.ax4 = self.fig.add_subplot(144)

            # Create images
            self.img1 = self.ax1.imshow(frame1)
            self.img2 = self.ax2.imshow(frame2)

            # Set titles
            self.ax1.set_title('Phase 1')
            self.ax2.set_title('Phase 2')
            self.ax3.set_title('PSD Instantaneous')
            self.ax4.set_title('PSD Average')
        else:
            # Update existing images
            self.img1.set_data(frame1)
            self.img1.set_clim(frame1.min(), frame1.max())
            self.img2.set_data(frame2)
            self.img2.set_clim(frame2.min(), frame2.max())

        # Clear and update PSD plots
        self.ax3.clear()
        self.ax4.clear()

        # Plot instantaneous PSDs with low alpha
        self.ax3.loglog(psd_stat1[ss//2, ss//2+1:], alpha=0.025, color='r', label='Phase 1')
        self.ax3.loglog(psd_stat2[ss//2, ss//2+1:], alpha=0.025, color='b', label='Phase 2')
        self.ax3.set_title('PSD Instantaneous')
        self.ax3.legend()

        # Plot averaged PSDs
        self.ax4.loglog(self.psd_statTot1[ss//2, ss//2+1:], color='r', label='Phase 1')
        self.ax4.loglog(self.psd_statTot2[ss//2, ss//2+1:], color='b', label='Phase 2')
        self.ax4.set_title(f'PSD Average (n={self.nframes})')
        self.ax4.legend()

        self._safe_draw()

    def trigger_code(self):
        """Override to handle dual phase inputs"""
        try:
            if not self._opened:
                self._create_figure()

            # DoublePhaseDisplay handles dual inputs
            self._update_display()
        except Exception as e:
            self._show_error(f"Double phase display error: {str(e)}")
