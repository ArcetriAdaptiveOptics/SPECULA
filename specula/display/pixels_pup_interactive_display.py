import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from specula import cpuArray
from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData


class PixelsPupInteractiveDisplay(BaseDisplay):

    def __init__(self,
                 title="Pixels + Pupils",
                 figsize=(9,6),
                 log_scale=False):

        super().__init__(title=title, figsize=figsize)

        self._log_scale = log_scale

        self.input_key = "in_pixels"  # Used by base class to identify which input to trigger on
        self.inputs["in_pixels"] = InputValue(type=Pixels)
        self.inputs["in_pupdata"] = InputValue(type=PupData)

        # display objects
        self.circles = []
        self.centers = []
        self.labels = []
        self.grid_lines = []
        self.text_block = None

        # interaction
        self.dragging_idx = None

        # color map for A,B,C,D pupils
        self.pupil_colors = ["red", "lime", "cyan", "yellow"]
        self._events_connected = False

    def _connect_events(self):
        if self.fig is not None and not self._events_connected:
            self.fig.canvas.mpl_connect("button_press_event", self._on_press)
            self.fig.canvas.mpl_connect("button_release_event", self._on_release)
            self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self._events_connected = True

    # ---------------------------------------------------
    # INTERACTIVE DRAGGING
    # ---------------------------------------------------

    def _on_press(self, event):

        if event.inaxes != self.ax:
            return

        for i, circle in enumerate(self.circles):

            cx, cy = circle.center
            r = circle.radius

            dist = np.sqrt((event.xdata-cx)**2 + (event.ydata-cy)**2)

            if dist < r:
                self.dragging_idx = i
                break


    def _on_release(self, event):
        self.dragging_idx = None


    def _on_motion(self, event):

        if self.dragging_idx is None:
            return

        if event.inaxes != self.ax:
            return

        idx = self.dragging_idx

        self.circles[idx].center = (event.xdata, event.ydata)
        self.centers[idx].set_data([event.xdata], [event.ydata])

        if self.local_inputs["in_pupdata"] is not None:
            pup = self.local_inputs["in_pupdata"]
            pup.cx[idx] = event.xdata
            pup.cy[idx] = event.ydata

        self._update_text(self.local_inputs["in_pupdata"])

        self._safe_draw()


    # ---------------------------------------------------
    # GRID
    # ---------------------------------------------------

    def _draw_grid(self, pup):

        cx = cpuArray(pup.cx)
        cy = cpuArray(pup.cy)

        meanx = np.mean(cx)
        meany = np.mean(cy)

        if len(self.grid_lines) == 0:

            vline = self.ax.axvline(meanx, linestyle="--", color="white", alpha=0.4)
            hline = self.ax.axhline(meany, linestyle="--", color="white", alpha=0.4)

            self.grid_lines = [vline, hline]

        else:

            self.grid_lines[0].set_xdata([meanx, meanx])
            self.grid_lines[1].set_ydata([meany, meany])


    # ---------------------------------------------------
    # TEXT INFO
    # ---------------------------------------------------

    def _update_text(self, pup):

        if pup is None:
            return

        cx = cpuArray(pup.cx)
        cy = cpuArray(pup.cy)
        r = cpuArray(pup.radius)

        n = pup.n_pupils

        text = ""

        for i in range(n):
            text += f"P{i}: cx={cx[i]:6.2f} cy={cy[i]:6.2f} r={r[i]:6.2f}\n"

        text += "\nDistances:\n"

        for i in range(n):
            for j in range(i+1, n):

                d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)

                text += f"d({i},{j}) = {d:6.2f}\n"

        if self.text_block is None:

            self.text_block = self.ax.text(
                1.02,
                1.0,
                text,
                transform=self.ax.transAxes,
                verticalalignment="top",
                fontsize=10,
                family="monospace"
            )

        else:
            self.text_block.set_text(text)


    # ---------------------------------------------------
    # PUPIL DRAWING
    # ---------------------------------------------------

    def _draw_pupils(self, pup):

        if pup is None:
            return

        cx = cpuArray(pup.cx)
        cy = cpuArray(pup.cy)
        r = cpuArray(pup.radius)

        n = pup.n_pupils

        if len(self.circles) == 0:

            for i in range(n):

                color = self.pupil_colors[i % len(self.pupil_colors)]

                circle = Circle(
                    (cx[i], cy[i]),
                    r[i],
                    fill=False,
                    linewidth=2,
                    color=color
                )

                self.ax.add_patch(circle)
                self.circles.append(circle)

                pt, = self.ax.plot(cx[i], cy[i], marker="+", color=color)
                self.centers.append(pt)

                label = self.ax.text(
                    cx[i],
                    cy[i],
                    f"{i}",
                    color=color,
                    fontsize=12,
                    weight="bold"
                )

                self.labels.append(label)

        else:

            for i in range(n):

                self.circles[i].center = (cx[i], cy[i])
                self.circles[i].radius = r[i]

                self.centers[i].set_data([cx[i]], [cy[i]])

                self.labels[i].set_position((cx[i], cy[i]))

        self._draw_grid(pup)
        self._update_text(pup)


    # ---------------------------------------------------
    # MAIN UPDATE
    # ---------------------------------------------------

    def _update_display(self, pixels):

        # connect events on first update (after figure is created)
        self._connect_events()

        pixels = self.local_inputs['in_pixels']
        pup = self.local_inputs['in_pupdata']
        image = cpuArray(pixels.pixels)

        norm = None

        if self._log_scale:

            img_min = image.min()
            img_max = image.max()

            ratio = 1e-6

            if img_max <= 0:
                img_max = 1

            if img_min <= 0:
                img_min = img_max * ratio

            norm = mcolors.LogNorm(vmin=img_min, vmax=img_max)

            image = np.clip(image, img_min, img_max)

        if self.img is None:

            self.img = self.ax.imshow(image, norm=norm)

            if not self._colorbar_added:
                plt.colorbar(self.img, ax=self.ax, location='left')
                self._colorbar_added = True

        else:

            self.img.set_data(image)

            if norm is not None:
                self.img.set_norm(norm)

        self._draw_pupils(pup)
        self._safe_draw()