import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from specula import xp
from specula import cpuArray

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData


class PixelsDisplay(BaseDisplay):
    """
    Display for pixel data (images).
    
    Parameters:
    -----------
    title : str
        Display title
    figsize : tuple
        Figure size (width, height)
    sh_as_pyr : bool
        If True and subapdata is provided, reformat SH data as pyramid
    subapdata : SubapData, optional
        Subaperture data for pyramid reformatting
    log_scale : bool
        If True, display image in log10 scale
    crop : tuple or None, optional
        Crop region as (x_start, x_end, y_start, y_end) or
        (x_center, y_center, half_width, half_height) if crop_mode='center'.
        If None, display full image.
    crop_mode : str, optional
        'slice' (default): crop = (x_start, x_end, y_start, y_end)
        'center': crop = (x_center, y_center, half_width, half_height)
    """

    def __init__(self,
                 title='Pixels Display',
                 figsize=(6, 6),
                 sh_as_pyr=False,
                 subapdata: SubapData = None,
                 log_scale=False,
                 crop=None,
                 crop_mode='slice'):

        super().__init__(
            title=title,
            figsize=figsize
        )

        self._sh_as_pyr = sh_as_pyr
        self._subapdata = subapdata
        self._log_scale = log_scale
        self._crop = crop
        self._crop_mode = crop_mode

        # Validate crop_mode
        if crop_mode not in ['slice', 'center']:
            raise ValueError(f"crop_mode must be 'slice' or 'center', got '{crop_mode}'")

        # Setup input
        self.input_key = 'pixels'  # Used by base class
        self.inputs['pixels'] = InputValue(type=Pixels)

    def _apply_crop(self, image):
        """
        Apply crop to image.
        
        Parameters:
        -----------
        image : np.ndarray
            2D image array
            
        Returns:
        --------
        cropped_image : np.ndarray
            Cropped image
        """
        if self._crop is None:
            return image

        h, w = image.shape

        if self._crop_mode == 'slice':
            # Crop format: (x_start, x_end, y_start, y_end)
            if len(self._crop) != 4:
                raise ValueError(f"crop with mode 'slice' must have 4 values, got {len(self._crop)}")

            x_start, x_end, y_start, y_end = self._crop

            # Handle negative indices (Python-style)
            if x_start < 0:
                x_start = w + x_start
            if x_end < 0:
                x_end = w + x_end
            if y_start < 0:
                y_start = h + y_start
            if y_end < 0:
                y_end = h + y_end

            # Clamp to image bounds
            x_start = max(0, min(x_start, w))
            x_end = max(0, min(x_end, w))
            y_start = max(0, min(y_start, h))
            y_end = max(0, min(y_end, h))

            return image[y_start:y_end, x_start:x_end]

        elif self._crop_mode == 'center':
            # Crop format: (x_center, y_center, half_width, half_height)
            if len(self._crop) != 4:
                raise ValueError(f"crop with mode 'center' must have 4 values, got {len(self._crop)}")

            x_center, y_center, half_w, half_h = self._crop

            x_start = max(0, x_center - half_w)
            x_end = min(w, x_center + half_w)
            y_start = max(0, y_center - half_h)
            y_end = min(h, y_center + half_h)

            return image[y_start:y_end, x_start:x_end]

    def _update_display(self, pixels):
        """Override base method to implement pixels-specific display"""
        # Process image data
        image = cpuArray(pixels.pixels)

        if self._sh_as_pyr and self._subapdata is not None:
            image = self._reformat_as_pyramid(image, self._subapdata)

        # Apply crop before log scale
        image = self._apply_crop(image)

        norm = None
        if self._log_scale:
            # Ensure image has strictly positive values for LogNorm
            img_min = image.min()
            img_max = image.max()
            ratio = 1e-6
            if img_max <= 0:
                img_max = 1.0
            if img_min >= img_max*ratio or img_min <= 0:
                img_min = img_max*ratio
            norm = mcolors.LogNorm(vmin=img_min, vmax=img_max)
            # clip image to avoid issues with LogNorm
            image = np.clip(image, img_min, img_max)

        if self.img is None:
            self.img = self.ax.imshow(image, norm=norm)
            if not self._colorbar_added:
                plt.colorbar(self.img, ax=self.ax)
                self._colorbar_added = True
        else:
            self.img.set_data(image)
            if norm is not None:
                self.img.set_norm(norm)

        self._safe_draw()

    def _reformat_as_pyramid(self, pixels, subapdata):    
        pupil = subapdata.copyTo(-1).single_mask()
        idx2d = xp.unravel_index(subapdata.idxs, pixels.shape)
        A, B, C, D = pupil.copy(), pupil.copy(), pupil.copy(), pupil.copy()
        pix_idx = subapdata.display_map
        half_sub = subapdata.np_sub // 2
        for i in range(subapdata.n_subaps):
            subap = pixels[idx2d[0][i], idx2d[1][i]].reshape(half_sub*2, half_sub*2)
            A.flat[pix_idx[i]] = subap[:half_sub, :half_sub].sum()
            B.flat[pix_idx[i]] = subap[:half_sub, half_sub:].sum()
            C.flat[pix_idx[i]] = subap[half_sub:, :half_sub].sum()
            D.flat[pix_idx[i]] = subap[half_sub:, half_sub:].sum()

        pyr_pixels = np.vstack((np.hstack((A, B)), np.hstack((C, D))))
        return pyr_pixels

    def set_crop(self, crop, crop_mode='slice'):
        """
        Set or update crop region dynamically.
        
        Parameters:
        -----------
        crop : tuple or None
            Crop region specification
        crop_mode : str
            'slice' or 'center'
        """
        if crop_mode not in ['slice', 'center']:
            raise ValueError(f"crop_mode must be 'slice' or 'center', got '{crop_mode}'")

        self._crop = crop
        self._crop_mode = crop_mode

    def clear_crop(self):
        """Remove crop and display full image."""
        self._crop = None
