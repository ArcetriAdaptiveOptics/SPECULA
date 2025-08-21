import numpy as np

from specula import cpuArray
from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.slopes import Slopes


class SlopecDisplay(BaseDisplay):
    def __init__(self,
                 title='Slopes Display',
                 figsize=(6, 6)):

        super().__init__(
            title=title,
            figsize=figsize
        )

        self.img = None

        # Setup input
        self.input_key = 'slopes'  # Used by base class
        self.inputs['slopes'] = InputValue(type=Slopes)

    def _update_display(self, slopes_obj):
        """Override base method to implement slopes-specific display"""
        # Get 2D slopes data and convert to displayable format
        frame3d = slopes_obj.get2d()
        if len(frame3d.shape) == 3:
            frame2d = np.hstack(cpuArray(frame3d))
        else:
            # slopes from intensity case
            frame2d = cpuArray(frame3d)

        if self.img is None:
            # First time: create image
            self.img = self.ax.imshow(frame2d)
            self._add_colorbar_if_needed(self.img)

            # Set axis labels for clarity
            self.ax.set_xlabel('Slope Components')
            self.ax.set_ylabel('Subapertures')
        else:
            # Update existing image
            self._update_image_data(self.img, frame2d)

        self._safe_draw()