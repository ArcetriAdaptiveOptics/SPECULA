import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle

from specula import cpuArray

from specula.base_value import BaseValue
from specula.display.base_display import BaseDisplay
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData


class PixelsPupDisplay(BaseDisplay):
    """
    Display for pixel data with pupil overlays.

    Shows:
        - image pixels
        - 4 pupil circles
        - pupil centers
        - radius values
        - distances between pupil centers
    """

    def __init__(self,
                 title="Pixels + Pupils",
                 figsize=(8,6),
                 log_scale=False,
                 crop=None,
                 crop_mode="slice"):

        super().__init__(title=title, figsize=figsize)

        self._log_scale = log_scale
        self._crop = crop
        self._crop_mode = crop_mode

        # inputs
        self.input_key = 'in_pixels' # Used by base class to identify which input to trigger on
        self.inputs["in_pixels"] = InputValue(type=Pixels)
        self.inputs["in_pupdata"] = InputValue(type=PupData)
        self.inputs["in_params"] = InputValue(type=BaseValue, optional=True)

        # display objects
        self.circles = []
        self.center_points = []
        self.text_block = None


    def _apply_crop(self, image):

        if self._crop is None:
            return image

        h, w = image.shape

        if self._crop_mode == "slice":
            x0, x1, y0, y1 = self._crop
            return image[y0:y1, x0:x1]

        elif self._crop_mode == "center":
            cx, cy, hw, hh = self._crop

            x0 = max(0, cx-hw)
            x1 = min(w, cx+hw)
            y0 = max(0, cy-hh)
            y1 = min(h, cy+hh)

            return image[y0:y1, x0:x1]


    def _draw_pupils(self, pupdata):

        if pupdata is None:
            return

        cx = cpuArray(pupdata.cx)
        cy = cpuArray(pupdata.cy)
        r  = cpuArray(pupdata.radius)

        n = pupdata.n_pupils

        # create circles once
        if len(self.circles) == 0:
            for i in range(n):
                circle = Circle((cx[i], cy[i]),
                                r[i],
                                fill=False,
                                color="red",
                                linewidth=2)

                self.ax.add_patch(circle)
                self.circles.append(circle)

                pt, = self.ax.plot(cx[i], cy[i], "rx")
                self.center_points.append(pt)

        else:
            for i in range(n):
                self.circles[i].center = (cx[i], cy[i])
                self.circles[i].radius = r[i]
                self.center_points[i].set_data([cx[i]], [cy[i]])


        # distances
        dist_text = ""
        for i in range(n):
            for j in range(i+1, n):

                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]

                d = np.sqrt(dx*dx + dy*dy)

                dist_text += f"d({i},{j}) = {d:.2f}\n"


        # center/radius text
        info = ""

        for i in range(n):
            info += f"P{i}: cx={cx[i]:.1f}  cy={cy[i]:.1f}  r={r[i]:.1f}\n"

        info += "\n" + dist_text
        info += "\n" + f"Generated at t={self.t_to_seconds(pupdata.generation_time):.2f} sec"

        if self.local_inputs['in_params'] is not None:
            for key, value in self.local_inputs['in_params'].value.items():
                info += f"\n{key} = {value}"

        if self.text_block is None:

            self.text_block = self.ax.text(
                1.02,
                1.0,
                info,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                family="monospace"
            )

        else:
            self.text_block.set_text(info)

    def _update_display(self, pixels):

        pupdata = self.local_inputs["in_pupdata"]
        pixels = self.local_inputs['in_pixels']


        image = cpuArray(pixels.pixels)

        image = self._apply_crop(image)

        norm = None

        if self._log_scale:

            img_min = image.min()
            img_max = image.max()

            ratio = 1e-6

            if img_max <= 0:
                img_max = 1

            if img_min >= img_max*ratio or img_min <= 0:
                img_min = img_max*ratio

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


        # draw pupil overlay
        self._draw_pupils(pupdata)
        self._safe_draw()