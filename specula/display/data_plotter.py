
import matplotlib
import numpy as np
matplotlib.use('Agg') # Memory backend, no GUI

from matplotlib.figure import Figure

dataplotter_cache = {}

class DataPlotter():
    '''
    Plot any kind of data in a memory backend
    '''
    def __init__(self, disp_factor=1, histlen=200, wsize=(400, 300), yrange=(-10, 10), title=''):
        super().__init__()

        self._wsize = wsize
        self._history = np.zeros(histlen)
        self._count = 0
        self._yrange = yrange
        self._value = None
        self._title = title
        self._w_is_set = False
        self._first = True
        self._disp_factor = disp_factor

    def set_w(self, size_frame=None, nframes=1):
        if self._w_is_set:
            return

        if size_frame is None:
            size_frame = self._wsize
        self.fig = Figure(figsize=(size_frame[0] * self._disp_factor / 100 * nframes, size_frame[1] * self._disp_factor / 100))
        self.ax = []
        for i in range(nframes):
            self.ax.append(self.fig.add_subplot(1, nframes, i+1))
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)  # Adjust these values as needed
  
        self._w_is_set = True

    def multi_plot(self, obj_list):
        '''
        Plot a list of data objects one next to the other.
        
        Returns a matplotlib.Figure object.
        
        Generation of the numpy array to display is delegated to each data object.
        '''

        if obj_list is None:
            return self.plot_text(f'This is None')

        if len(obj_list) < 1:
            return self.plot_text(f'No values to plot')

#        TODO: commented out because it does not work for prop.layer_list,
#        since layers are of different types
#
#        for obj in objrefs[1:]:
#            if type(obj) is not type(objrefs[0]):
#                raise ValueError('All objects in multi_plot() must be of the same type')

        if not hasattr(obj_list[0], 'array_for_display'):
            return self.plot_text(f'Plot not implemented for class {obj_list[0].__class__.__name__}')

        frames = [x.array_for_display() for x in obj_list]

        for f in frames:
            if f is None:
                return self.plot_text(f'Cannot plot None values')

        # 2d images: imshow
        if len(frames[0].shape) == 2:
            return self.imshow(frames)

        # Single vector value: plot_vector
        elif len(frames[0].shape) == 1 and len(frames[0]) > 1 and len(frames) == 1:
            return self.plot_vector(frames[0])

        # Scalar value: plot history
        elif len(frames[0].shape) == 1 and len(frames[0]) == 1 and len(frames) == 1:
            return self.plot_history(frames[0])

        # Another kind of scalar value: plot history
        elif len(frames[0].shape) == 0 and len(frames) == 1:
            return self.plot_history(frames[0])

        else:
            return self.plot_text(f'Cannot plot: data shape is {frames[0].shape} x {len(frames)}')

    def plot_history(self, value):
        n = len(self._history)
        if self._count >= n:
            self._history[:-1] = self._history[1:]
            self._count = n - 1

        self.set_w()
        self._history[self._count] = value
        self._count += 1

        x = np.arange(self._count)
        y = self._history[:self._count]
        if self._first:
            self.fig.suptitle(self._title)
            self.line = self.ax[0].plot(x, y, marker='.')
            self._first = False
        else:
            self.line[0].set_xdata(x)
            self.line[0].set_ydata(y)
            self.ax[0].set_xlim(x.min(), x.max())
            self.ax[0].set_ylim(y.min(), y.max())
        self.fig.canvas.draw()
        return self.fig

    def plot_vector(self, vector):
        self.set_w()

        if self._first:
            self._line = self.ax[0].plot(vector, '.-')
            self.fig.suptitle(self._title)
            self.ax[0].set_ylim([vector.min(), vector.max()])
            self._first = False
        else:
            self._line[0].set_ydata(vector)
        return self.fig

    def imshow(self, frames):
        if np.sum(self._wsize) == 0:
            size_frame = frames[0].shape[0]
        else:
            size_frame = self._wsize

        self.set_w(size_frame, len(frames))

        if self._first:
            self.img = []
            for i, frame in enumerate(frames):
                self.img.append(self.ax[i].imshow(frame))
            self._first = False
        else:
            for i, frame in enumerate(frames):
                self.img[i].set_data(frame)
                self.img[i].set_clim(frame.min(), frame.max())
        self.fig.canvas.draw()
        return self.fig

    def plot_text(self, text):
        self.set_w()

        if self._first:
            self.text = self.ax[0].text(0, 0, text, fontsize=14)
        else:
            del self.text
            self.text = self.ax[0].text(0, 0, text, fontsize=14)
        self.fig.canvas.draw()
        return self.fig


    @staticmethod
    def plot_best_effort(plot_name, dataobj_or_list):
        '''
        Plot a data object or a list of data objects as best as it can be done.
        The plot_name is used to remember the DataPlotter instance and allow
        plot updates instead of expensive re-plots from scratch
        '''

        if plot_name not in dataplotter_cache:
            dataplotter_cache[plot_name] = DataPlotter()

        if isinstance(dataobj_or_list, list):
            for obj in dataobj_or_list:
                obj.xp = np
            fig = dataplotter_cache[plot_name].multi_plot(dataobj_or_list)
        else:
            dataobj_or_list.xp = np  # Supply a numpy instance, sometimes it is needed
            fig = dataplotter_cache[plot_name].multi_plot([dataobj_or_list])
        return fig