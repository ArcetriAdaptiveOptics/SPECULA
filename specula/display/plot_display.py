import numpy as np
import matplotlib.pyplot as plt

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue, InputList
from specula.base_value import BaseValue


class PlotDisplay(BaseDisplay):
    def __init__(self,
                 title='Plot Display',
                 figsize=(8, 6),
                 histlen=200,
                 yrange=(0, 0),
                 x_axis='time'):  # can be time or iteration

        super().__init__(
            title=title,
            figsize=figsize
        )

        self._histlen = histlen
        self._history = np.zeros(histlen)
        self._count = 0
        self._yrange = yrange
        self.lines = None
        self._x_axis = x_axis
        self._time_history = []

        # Setup inputs - can handle both single value and list of values
        self.inputs['value'] = InputValue(type=BaseValue, optional=True)
        self.inputs['value_list'] = InputList(type=BaseValue, optional=True)

    def _get_data(self):
        """Get unified list of values"""
        if len(self.local_inputs['value_list']) > 0:
            return self.local_inputs['value_list']
        elif self.local_inputs['value'] is not None:
            return [self.local_inputs['value']]
        else:
            return []

    def _update_display(self, data_list):
        """Update display with list of data points"""
        nValues = len(data_list)
        n = self._history.shape[0]

        # Reshape history array if needed for multiple values
        if self._history.ndim == 1 and nValues > 1:
            self._history = np.zeros((n, nValues))
        elif self._history.ndim == 2 and self._history.shape[1] != nValues:
            self._history = np.zeros((n, nValues))

        # Scroll history if buffer is full
        if self._count >= n:
            if self._history.ndim == 1:
                self._history[:-1] = self._history[1:]
            else:
                self._history[:-1, :] = self._history[1:, :]
            self._count = n - 1
            self._time_history = self._time_history[1:]

        # X axis for current data
        if self._x_axis == 'time':
            self._time_history.append(self.current_time_seconds)
        else:
            self._time_history.append(self._time_history[-1]+1 if self._time_history else 1)
        x = np.array(self._time_history)

        # Update data and plots
        xmin, xmax, ymin, ymax = [], [], [], []

        for i in range(nValues):
            v = data_list[i]

            # Extract scalar value from potentially array-like value
            if hasattr(v.value, 'item'):
                # For numpy arrays, use .item() to extract scalar
                scalar_value = v.value.item()
            elif hasattr(v.value, '__len__') and len(v.value) == 1:
                # For single-element sequences
                scalar_value = v.value[0]
            else:
                # Already a scalar
                scalar_value = v.value

            # Store new value in history
            if self._history.ndim == 1:
                self._history[self._count] = scalar_value
                y = self._history[:self._count + 1]
            else:
                self._history[self._count, i] = scalar_value
                y = self._history[:self._count + 1, i]

            if self.lines is None:
                # First time: create lines list and reference line
                self.lines = []
                self.ax.axhline(y=0, color='grey', linestyle='--',
                              dashes=(4, 8), linewidth=0.5, alpha=0.7)

            # Create or update line
            if i >= len(self.lines):
                # Create new line for this series
                line = self.ax.plot(x, y, marker='.', 
                                  color=plt.cm.tab10(i % 10))[0]
                self.lines.append(line)
            else:
                # Update existing line
                self.lines[i].set_xdata(x)
                self.lines[i].set_ydata(y)

            # Track limits for axis scaling
            xmin.append(x.min())
            xmax.append(x.max())
            ymin.append(y.min())
            ymax.append(y.max())

        # Update axes limits
        if len(self.lines) > 0:
            if xmin != xmax:
                self.ax.set_xlim(min(xmin), max(xmax))

            if np.sum(np.abs(self._yrange)) > 0:
                self.ax.set_ylim(self._yrange[0], self._yrange[1])
            elif ymin != ymax:
                self.ax.set_ylim(min(ymin), max(ymax))

        # Set x axis label
        if self._x_axis == 'time':
            self.ax.set_xlabel('Time [s]')
        else:
            self.ax.set_xlabel('Iteration')

        self._safe_draw()
        self._count += 1
