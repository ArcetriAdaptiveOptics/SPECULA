import matplotlib.pyplot as plt

from specula.scalar_values import IntValue
from specula.base_processing_obj import BaseProcessingObj
from specula.base_processing_obj import OutputDesc

def runningOnNotebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and 'IPKernelApp' in get_ipython().config
    except:
        return False

class BaseDisplay(BaseProcessingObj):

    __plot_completed = {}

    def __init__(self,
                 title='',
                 window: int=None,
                 subplot: int=111,
                 figsize=(8, 6)):
        super().__init__()

        if window is None:
            window = id(self)

        self.window = window
        self.figsize = figsize
        self.colorbar_added = False
        self.input_key = ''
        self.subplot = subplot
        self.onNotebook = runningOnNotebook()

        if window not in self.__plot_completed:
            self.__plot_completed[window] = {}

        self.fig = plt.figure(num=self.window, figsize=self.figsize)
        self.ax = self.fig.add_subplot(self.subplot)
        self.__plot_completed[self.window][self.subplot] = False

        if title:
            self.ax.set_title(title)

        if not self.onNotebook:
            self.fig.show()
        else:
            from IPython.display import display
            self.handle = display(self.fig, display_id=True)

        self.output_id = IntValue(value=-1)
        self.outputs['out_window_id'] = self.output_id

    @classmethod
    def output_names(cls):
        return {'out_window_id': OutputDesc(IntValue, 'Window ID where the plot has been drawn')}

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

    def _safe_draw(self):
        """Thread-safe drawing method"""
        try:
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        except Exception as e:
            self.logger.error(f"Drawing error: {e}")
