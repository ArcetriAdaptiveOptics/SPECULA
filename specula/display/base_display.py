from specula.base_processing_obj import BaseProcessingObj
import matplotlib.pyplot as plt


class BaseDisplay(BaseProcessingObj):
    def __init__(self,
                 title='',
                 figsize=(8, 6)):
        super().__init__()
        self._title = title
        self._figsize = figsize
        self._opened = False
        self._colorbar_added = False
        self.input_key = ''
        self.fig = None
        self.ax = None
        self.img = None
        self.line = None

    def _create_figure(self):
        """Create the matplotlib figure and axes"""
        if self._opened:
            return

        self.fig = plt.figure(figsize=self._figsize)
        self.ax = self.fig.add_subplot(111)
        if self._title:
            self.fig.suptitle(self._title)
        self.fig.show()
        self._opened = True

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
            if not self._opened:
                self._create_figure()
            data = self._get_data()
            self._update_display(data)
        except Exception as e:
            self._show_error(f"Display error: {str(e)}")

    # ============ UTILITY METHODS ============

    def reset(self):
        """Reset the display"""
        if self._opened:
            self.ax.clear()
            self._safe_draw()
        self.img = None
        self.line = None
        self._colorbar_added = False

    def close(self):
        """Close the display window"""
        if self.fig:
            plt.close(self.fig)
            self._opened = False
            self.fig = None
            self.ax = None
            self.img = None
            self.line = None
            self._colorbar_added = False

    def set_y_range(self, ymin, ymax):
        """Set fixed Y axis range"""
        if hasattr(self, '_yrange'):
            self._yrange = (ymin, ymax)
            if self._opened and self.ax:
                self.ax.set_ylim(ymin, ymax)
                self._safe_draw()

    def auto_y_range(self):
        """Enable automatic Y axis scaling - override in subclasses for specific logic"""
        if hasattr(self, '_yrange'):
            self._yrange = (0, 0)

    def _add_colorbar_if_needed(self, image_obj):
        """Add colorbar if not already present"""
        if not hasattr(self, '_colorbar_added'):
            self._colorbar_added = False
            
        if not self._colorbar_added and image_obj is not None:
            plt.colorbar(image_obj, ax=self.ax)
            self._colorbar_added = True

    def _update_image_data(self, image_obj, data):
        """Standard image update logic"""
        if image_obj is not None:
            image_obj.set_data(data)
            image_obj.set_clim(data.min(), data.max())

    def _show_error(self, message):
        if not self._opened:
            self._create_figure()
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
            print(f"Drawing error: {e}")