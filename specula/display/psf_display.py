import numpy as np

from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.base_value import BaseValue


class PsfDisplay(BaseDisplay):
    def __init__(self,
                 title='PSF Display',
                 figsize=(6, 6),
                 log_scale=False,
                 image_p2v=0.0):

        super().__init__(
            title=title,
            figsize=figsize
        )

        self._log_scale = log_scale
        self._image_p2v = image_p2v

        # Setup input
        self.input_key = 'psf'  # Used by base class
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
            self.img = self.ax.imshow(image)
            self._add_colorbar_if_needed(self.img)
        else:
            # Update existing image
            self._update_image_data(self.img, image)

        self._safe_draw()