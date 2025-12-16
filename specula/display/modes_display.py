import numpy as np

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray


class ModesDisplay(BaseDisplay):
    def __init__(self, 
                 title='Modes Display',
                 figsize=(6, 3),
                 xrange=None,
                 yrange=(-500, 500)):

        super().__init__(
            title=title,
            figsize=figsize
        )

        self._xrange = xrange
        self._yrange = yrange
        self.line = None

        # Setup input
        self.input_key = 'modes' # Used by base class
        self.inputs['modes'] = InputValue(type=BaseValue)

    def _update_display(self, modes):
        """Override base method to implement modes-specific display"""
        # Get the modes vector
        y = cpuArray(modes.value)
        x = np.arange(len(y))

        if self.line is None:
            # First time: create line
            self.line = self.ax.plot(x, y, '.-')[0]

            # Set fixed X range if specified
            if self._xrange is not None:
                self.ax.set_xlim(self._xrange[0], self._xrange[1])
            elif len(x) > 0:
                self.ax.set_xlim(0, len(x) - 1)

            # Set fixed Y range if specified
            if np.sum(np.abs(self._yrange)) > 0:
                self.ax.set_ylim(self._yrange[0], self._yrange[1])
            else:
                # Auto-scale based on data
                self.ax.set_ylim(y.min() * 1.1, y.max() * 1.1)

            # Add reference line at y=0
            self.ax.axhline(y=0, color='grey', linestyle='--',
                          dashes=(4, 8), linewidth=0.5, alpha=0.7)

            # Set labels
            self.ax.set_xlabel('Mode Index')
            self.ax.set_ylabel('Mode Value')
        else:
            # Update existing line
            self.line.set_xdata(x)
            self.line.set_ydata(y)

            # Update X limits if vector size changed or xrange is set
            if self._xrange is not None:
                self.ax.set_xlim(self._xrange[0], self._xrange[1])
            elif len(x) > 0:
                self.ax.set_xlim(0, len(x) - 1)

            # Update Y limits if auto-scaling
            if np.sum(np.abs(self._yrange)) == 0:
                self.ax.set_ylim(y.min() * 1.1, y.max() * 1.1)

        # Draw efficiently
        self._safe_draw()

    def set_x_range(self, xmin, xmax):
        """Set fixed X axis range"""
        self._xrange = (xmin, xmax)
        if self.line is not None:
            self.ax.set_xlim(xmin, xmax)
            self._safe_draw()

    def set_y_range(self, ymin, ymax):
        """Set fixed Y axis range"""
        self._yrange = (ymin, ymax)
        if self.line is not None:
            self.ax.set_ylim(ymin, ymax)
            self._safe_draw()
