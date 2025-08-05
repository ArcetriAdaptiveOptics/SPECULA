import numpy as np
import matplotlib.pyplot as plt

from specula import xp
from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.base_value import BaseValue


class PsfDisplay(BaseDisplay):
    def __init__(self, 
                 window=None, 
                 title='PSF Display',
                 figsize=(6, 6),
                 log_scale=False,
                 image_p2v=0.0):

        super().__init__(
            window=window,
            title=title,
            figsize=figsize
        )

        self._log_scale = log_scale
        self._image_p2v = image_p2v
        self.img = None

        # Setup input
        self.input_key = 'psf'
        self.inputs['psf'] = InputValue(type=BaseValue)

    def _process_psf_data(self, psf):
        """Process PSF data: apply P2V threshold and log scaling"""
        image = cpuArray(psf.value)

        # Apply P2V threshold if specified
        if self._image_p2v > 0:
            threshold = self._image_p2v**(-1.) * np.max(image)
            image = np.maximum(image, threshold)

        # Apply logarithmic scaling if requested
        if self._log_scale:
            # Avoid log(0) by ensuring minimum positive value
            image = np.maximum(image, 1e-10)
            image = np.log10(image)

        return image

    def _update_display(self, psf):
        """Override base method to implement PSF-specific display"""
        image = self._process_psf_data(psf)

        if self.img is None:
            # First time: create image
            self.img = self.ax.imshow(image, aspect='auto')
            self._add_colorbar_if_needed(self.img)
        else:
            # Update existing image
            self._update_image_data(self.img, image)

        self._safe_draw()

    def set_log_scale(self, log_scale: bool):
        """Enable/disable logarithmic scaling"""
        self._log_scale = log_scale

    def set_p2v_threshold(self, p2v_value: float):
        """Set P2V threshold value"""
        self._image_p2v = p2v_value

    def get_display_info(self):
        """Get current display configuration"""
        return {
            'log_scale': self._log_scale,
            'p2v_threshold': self._image_p2v,
            'window': self._window,
            'title': self._title
        }