import matplotlib.pyplot as plt
from numbers import Integral

from specula.scalar_values import IntValue
from specula.base_processing_obj import BaseProcessingObj
from specula.base_processing_obj import OutputDesc
from specula.display import display_process

def runningOnNotebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and 'IPKernelApp' in get_ipython().config
    except:
        return False

class BaseDisplay(BaseProcessingObj):

    __plot_completed = {}

    # With --async-displays, updates can be skipped when the display process is busy.
    # Displays that accumulate a history set this to False
    skip_updates = True

    def __new__(cls, *args, **kwargs):
        # Constructor arguments, used to build a replica in the display process
        obj = super().__new__(cls)
        obj._init_args = args
        obj._init_kwargs = kwargs
        return obj

    def __init__(self,
                 title='',
                 window: int=None,
                 subplot: int=111,
                 figsize=(8, 6),
                 window_xy=None):
        super().__init__()

        if isinstance(window, Integral) and not isinstance(window, bool) and window >= 1:
            window = int(window)
            # Displays can share a window, each one in its own subplot.
            # The figure size is set by the first display of the window
            if window in self.__plot_completed:
                if subplot in self.__plot_completed[window]:
                    raise ValueError(f'subplot {subplot} of window {window} already exists')
                figsize = None
        elif window is None:
            # Find an unused window number
            window = max(self.__plot_completed.keys(), default=0) + 1
        else:
            raise ValueError('window must be a positive integer')

        self.window = window
        self.figsize = figsize
        self.colorbar_added = False
        self.input_key = ''
        self.subplot = subplot
        self.onNotebook = runningOnNotebook()

        if window not in self.__plot_completed:
            self.__plot_completed[window] = {}

        self.output_id = IntValue(value=-1)
        self.outputs['out_window_id'] = self.output_id

        # Drawing happens in the display process: no figure here.
        # The display code, including setup() and finalize() of derived classes,
        # runs there: here setup() only checks the inputs
        self.async_mode = display_process.enabled
        if self.async_mode:
            self.fig = self.ax = None
            self.setup = lambda: BaseProcessingObj.setup(self)
            self.finalize = lambda: None
            self.trigger = lambda: display_process.send(self)
            display_process.displays.append(self)
            return

        self.fig = plt.figure(num=self.window, figsize=self.figsize)
        self.ax = self.fig.add_subplot(self.subplot)
        self.__plot_completed[self.window][self.subplot] = False

        if title:
            self.ax.set_title(title)

        if not self.onNotebook:
            self.fig.show()
            if window_xy is not None:
                self._set_window_position(window_xy)
        else:
            from IPython.display import display
            self.handle = display(self.fig, display_id=True)

    def _set_window_position(self, window_xy):
        """Place the GUI window at screen pixel (x, y) if the backend allows it."""
        try:
            x, y = int(window_xy[0]), int(window_xy[1])
        except (TypeError, ValueError, IndexError):
            self.logger.warning(f'Ignoring window_xy={window_xy!r}: expected [x, y] in screen pixels')
            return
        # Non-GUI backends (e.g. Agg) have no window: nothing to move
        win = getattr(self.fig.canvas.manager, 'window', None)
        try:
            if hasattr(win, 'wm_geometry'):     # Tk
                win.wm_geometry(f'+{x}+{y}')
            elif hasattr(win, 'move'):          # Qt, GTK
                win.move(x, y)
        except Exception as e:
            self.logger.debug(f'Could not set window position: {e}')

    @classmethod
    def output_names(cls):
        return {'out_window_id': OutputDesc(IntValue, 'Window ID where the plot has been drawn')}

    @classmethod
    def reset_windows(cls):
        '''Forget all windows and close their figures,
        so that a new simulation can use the same window numbers'''
        for window in cls.__plot_completed:
            plt.close(window)
        cls.__plot_completed.clear()

    def _update_display(self, data):
        """Update the display with new data"""
        raise NotImplementedError("Subclasses should implement this method")

    def _get_data(self):
        """Get data from input. Derived classes can override this method
        in case of complex data"""
        data = self.local_inputs.get(self.input_key)
        if data is None:
            self._show_error(f"No {self.input_key} data available")
            return
        return data

    def trigger_code(self):
        try:
            data = self._get_data()
            self._update_display(data)
            if self.onNotebook:
                self.handle.update(self.fig)
        except Exception as e:
            self._show_error(f"Display error: {str(e)}")

    def post_trigger(self):
        super().post_trigger()

        self.__plot_completed[self.window][self.subplot] = True
        self.output_id.value = self.window
        self.output_id.generation_time = self.current_time

        if self.async_mode:
            return

        # If all subplots in this window have completed drawing,
        # call safe_draw() and reset the plot flags

        if all(self.__plot_completed[self.window].values()):
            self._safe_draw()
            for k in self.__plot_completed[self.window].keys():
                self.__plot_completed[self.window][k] = False

    # ============ UTILITY METHODS ============

    def _add_colorbar_if_needed(self, image_obj, unit=None, **kwargs):
        """Add colorbar if not already present"""
        if not hasattr(self, 'colorbar_added'):
            self.colorbar_added = False

        if not self.colorbar_added and image_obj is not None:
            cbar = plt.colorbar(image_obj, ax=self.ax, **kwargs)
            if unit:
                cbar.ax.set_title(unit)
            self.colorbar_added = True

    def _update_image_data(self, image_obj, data):
        """Standard image update logic"""
        if image_obj is not None:
            image_obj.set_data(data)
            image_obj.set_clim(data.min(), data.max())

    def _show_error(self, message):
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha='center', va='center', 
                    transform=self.ax.transAxes, color='red', fontsize=12)
        self._safe_draw()

    def _has_gui_window(self):
        """True if the figure is shown in an open GUI window"""
        return self.fig.canvas.required_interactive_framework is not None and \
               plt.fignum_exists(self.fig.number)

    def _safe_draw(self):
        """Thread-safe drawing method"""
        # Without a GUI window (non-interactive backend like the Agg fallback
        # when $DISPLAY is not set, or a closed window) nobody will see the
        # figure, so skip the expensive rendering. DisplayRecorder renders
        # the figure itself when it needs the pixels.
        if not self.onNotebook and not self._has_gui_window():
            return
        try:
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        except Exception as e:
            self.logger.error(f"Drawing error: {e}")
