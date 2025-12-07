import numpy as np
import matplotlib.pyplot as plt

from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray


class PlotVectorDisplay(BaseDisplay):
    """
    Display for plotting vector elements over time.
    
    Parameters:
    -----------
    title : str
        Display title
    figsize : tuple
        Figure size (width, height)
    histlen : int
        History buffer length
    yrange : tuple
        Fixed Y range (ymin, ymax). Use (0, 0) for auto
    x_axis : str
        'time' or 'iteration'
    indices : int or list of int, optional
        Which vector elements to plot. Can be:
        - Single int: plot that element
        - List of ints: plot those elements
        - None: plot all elements (up to max_elements)
    slice_args : list, optional
        Arguments for slice(start, stop, step). E.g., [0, 10, 2] plots every other element from 0 to 10
        Cannot be used together with indices
    legend_labels : list of str, optional
        Labels for legend. If None, use "Elem. 0", "Elem. 1", etc.
    max_elements : int, optional
        Maximum number of elements to plot. Default 20. Set to None for no limit.
    """

    def __init__(self,
                 title='Vector Plot',
                 figsize=(8, 6),
                 histlen=200,
                 yrange=(0, 0),
                 x_axis='time',
                 indices=None,
                 slice_args=None,
                 legend_labels=None,
                 max_elements=20):

        super().__init__(
            title=title,
            figsize=figsize
        )

        # Validate that indices and slice_args are not both set
        if indices is not None and slice_args is not None:
            raise ValueError("Cannot specify both 'indices' and 'slice_args'")

        self._histlen = histlen
        self._history = None  # Will be initialized when we know vector size
        self._count = 0
        self._yrange = yrange
        self.lines = None
        self._x_axis = x_axis
        self._time_history = []
        self._max_elements = max_elements

        # Normalize indices to list
        if indices is not None:
            if isinstance(indices, int):
                self._indices = [indices]
            else:
                self._indices = list(indices)
        else:
            self._indices = None

        # Create slice object if provided
        if slice_args is not None:
            if not isinstance(slice_args, (list, tuple)) or len(slice_args) == 0:
                raise ValueError("slice_args must be a non-empty list or tuple")
            self._slice_obj = slice(*slice_args)
        else:
            self._slice_obj = None

        self._legend_labels = legend_labels
        self._legend_set = False
        self._n_elements = None

        # Setup input for single vector value
        self.inputs['vector'] = InputValue(type=BaseValue)
        self.input_key = 'vector'

    def _get_data(self):
        """Get vector data from input"""
        vector_obj = self.local_inputs[self.input_key]
        if vector_obj is None:
            self._show_error(f"No {self.input_key} data available")
            return None
        return vector_obj

    def _extract_vector(self, value_obj):
        """
        Extract vector from BaseValue, handling different array types.
        
        Returns:
            vector: 1D numpy array (always a copy)
        """
        vec = cpuArray(value_obj.value).copy()

        # Handle different array-like types
        if hasattr(vec, 'ravel'):
            vec = vec.ravel()
        elif isinstance(vec, (list, tuple)):
            vec = np.array(vec)
        elif np.isscalar(vec):
            vec = np.array([vec])
        else:
            vec = np.array(vec)

        return vec

    def _initialize_history(self, n_elements):
        """Initialize history buffer when we first know the vector size."""
        if self._history is None or self._history.shape[1] != n_elements:
            self._history = np.zeros((self._histlen, n_elements))
            self._count = 0
            self._time_history = []
            self.lines = None
            self._legend_set = False

    def _get_plot_indices(self, n_elements):
        """
        Determine which elements to plot.
        
        Returns:
            list of int: Indices to plot
        """
        if self._slice_obj is not None:
            # Use slice to get indices
            indices = list(range(n_elements)[self._slice_obj])
        elif self._indices is not None:
            # Validate explicit indices
            valid_indices = [i for i in self._indices if 0 <= i < n_elements]
            if len(valid_indices) != len(self._indices):
                invalid = [i for i in self._indices if i < 0 or i >= n_elements]
                print(f"Warning: Indices {invalid} out of range for vector of size {n_elements}")
            indices = valid_indices
        else:
            # Plot all elements
            indices = list(range(n_elements))

        # Apply max_elements limit
        if self._max_elements is not None and len(indices) > self._max_elements:
            print(f"Warning: Plotting only first {self._max_elements} of {len(indices)}"
                  f" selected elements. Use max_elements=None to plot all or specify"
                  f" indices/slice_args to choose which ones.")
            indices = indices[:self._max_elements]

        return indices

    def _get_legend_label(self, idx, position):
        """
        Get legend label for element idx at position in plot.
        
        Parameters:
        -----------
        idx : int
            Original element index in the vector
        position : int
            Position in the list of plotted elements (0, 1, 2, ...)
        """
        if self._legend_labels and position < len(self._legend_labels):
            return self._legend_labels[position]
        else:
            return f'Elem. {idx}'

    def _update_display(self, vector_obj):
        """Update display with new vector data."""
        if vector_obj is None:
            return

        # Extract vector
        vec = self._extract_vector(vector_obj)
        n_elements = len(vec)

        # Initialize history if needed
        if self._history is None:
            self._initialize_history(n_elements)
        elif self._history.shape[1] != n_elements:
            # Vector size change is not supported
            raise ValueError(f"Vector size changed from {self._history.shape[1]} to {n_elements}")

        # Get indices to plot
        plot_indices = self._get_plot_indices(n_elements)
        if not plot_indices:
            self._show_error("No valid indices to plot")
            return

        # Scroll history if buffer is full
        if self._count >= self._histlen:
            self._history[:-1, :] = self._history[1:, :]
            self._count = self._histlen - 1
            self._time_history = self._time_history[1:]

        # Store new values
        self._history[self._count, :] = vec

        # X axis
        if self._x_axis == 'time':
            self._time_history.append(self.current_time_seconds)
        else:
            self._time_history.append(self._time_history[-1] + 1 if self._time_history else 1)

        x = np.array(self._time_history)

        # Initialize lines on first call
        if self.lines is None:
            self.lines = []
            self.ax.axhline(y=0, color='grey', linestyle='--',
                          dashes=(4, 8), linewidth=0.5, alpha=0.7)
            self.ax.grid(True, alpha=0.3)

        # Plot each element
        xmin, xmax, ymin, ymax = [], [], [], []

        for i, idx in enumerate(plot_indices):
            y = self._history[:self._count + 1, idx]

            # Create or update line
            if i >= len(self.lines):
                label = self._get_legend_label(idx, i)
                line = self.ax.plot(x, y, marker='.',
                                  color=plt.cm.tab10(i % 10),
                                  label=label)[0]
                self.lines.append(line)
            else:
                self.lines[i].set_xdata(x)
                self.lines[i].set_ydata(y)

            # Track limits
            xmin.append(x.min())
            xmax.append(x.max())
            ymin.append(y.min())
            ymax.append(y.max())

        # Add legend if not already set
        if not self._legend_set and self.lines:
            self.ax.legend(loc='best', fontsize='small')
            self._legend_set = True

        # Update axes limits
        if xmin and xmax and min(xmin) < max(xmax):
            self.ax.set_xlim(min(xmin), max(xmax))

        if np.sum(np.abs(self._yrange)) > 0:
            self.ax.set_ylim(self._yrange[0], self._yrange[1])
        elif ymin and ymax and min(ymin) < max(ymax):
            margin = 0.05 * (max(ymax) - min(ymin))
            self.ax.set_ylim(min(ymin) - margin, max(ymax) + margin)

        # Set axis labels
        if self._x_axis == 'time':
            self.ax.set_xlabel('Time [s]')
        else:
            self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Value')

        self._safe_draw()
        self._count += 1
