import numpy as np
import matplotlib.colors as mcolors

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

        return image

    def _update_display(self, psf):
        """Override base method to implement PSF-specific display"""
        image = self._process_psf_data(psf)

        # Apply logarithmic scaling if requested
        norm = None
        if self._log_scale:
            # Ensure image has strictly positive values for LogNorm
            img_min = image.min()
            img_max = image.max()
            if self._image_p2v == 0:
                ratio = 1e-6
            else:
                ratio = 1/self._image_p2v
            if img_max <= 0:
                img_max = 1.0
            if img_min >= img_max*ratio or img_min <= 0:
                img_min = img_max*ratio
            norm = mcolors.LogNorm(vmin=img_min, vmax=img_max)
            # clip image to avoid issues with LogNorm
            image = np.clip(image, img_min, img_max)

        if self.img is None:
            # First time: create image
            self.img = self.ax.imshow(image, norm=norm)
            self._add_colorbar_if_needed(self.img)
        else:
            # Update existing image
            self._update_image_data(self.img, image)
            if norm is not None:
                self.img.set_norm(norm)
        self._safe_draw()
