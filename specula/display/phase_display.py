
import numpy as np

from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField

class PhaseDisplay(BaseDisplay):
    def __init__(self,
                 title='Phase Display',
                 figsize=(8, 6)):  # Default size in inches
        super().__init__(
            title=title,
            figsize=figsize
        )

        # Setup input
        self.input_key = 'phase'  # Used by base class
        self.inputs['phase'] = InputValue(type=ElectricField)

    def _process_phase_data(self, phase):
        """Process phase data: mask and remove average"""
        frame = cpuArray(phase.phaseInNm * (phase.A > 0).astype(float))

        # Get valid indices (where amplitude > 0)
        valid_mask = cpuArray(phase.A) > 0

        if np.any(valid_mask):
            # Remove average phase only from valid pixels
            frame[valid_mask] -= np.mean(frame[valid_mask])

            if self._verbose:
                print('Removing average phase in phase_display')

        return frame

    def _reset_elements(self):
        """Reset phase-specific elements"""
        self.img = None
        self._colorbar_added = False

    def _update_display(self, phase):
        frame = self._process_phase_data(phase)

        if self.img is None:
            self.img = self.ax.imshow(frame)
            self._add_colorbar_if_needed(self.img)
        else:
            self._update_image_data(self.img, frame)

        self._safe_draw()