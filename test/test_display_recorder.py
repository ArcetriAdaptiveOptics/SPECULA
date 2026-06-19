import os
import sys
import tempfile
import unittest
import matplotlib

matplotlib.use("Agg")  # must be set before pyplot import in real CI setups

import matplotlib.pyplot as plt
import imageio.v2 as imageio   # from scikit-image
import numpy as np

import specula
specula.init(0)  # Default target device

from specula.scalar_values import IntValue
from specula.display.display_recorder import DisplayRecorder


@unittest.skipIf(sys.platform == 'darwin' and sys.version_info < (3, 9), reason='Not implemented on MacOSX with python 3.8')
class TestDisplayRecorder(unittest.TestCase):

    def test_recorded_frame_is_not_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test.mp4")

            rec = DisplayRecorder(filename, fps=5)

            fig, ax = plt.subplots()
            ax.imshow(np.random.rand(20, 20))
            fig.canvas.draw()

            rec.local_inputs = {"in_windows": [IntValue(fig.number)]}
            rec.trigger()
            rec.finalize()

            reader = imageio.get_reader(filename)

            try:
                frame = reader.get_data(0)

                self.assertIsNotNone(frame)
                self.assertGreater(frame.mean(), 0)

            finally:
                reader.close()

            plt.close(fig)

    def test_multiple_frames_written(self):
        import tempfile
        import numpy as np
        import matplotlib.pyplot as plt
        import imageio

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/test.mp4"

            rec = DisplayRecorder(filename, fps=5)

            fig, ax = plt.subplots()

            n_frames = 5

            for i in range(n_frames):
                ax.clear()
                ax.plot([0, i], [0, i])
                fig.canvas.draw()

                rec.local_inputs = {"in_windows": [IntValue(fig.number)]}
                rec.trigger()

            rec.finalize()

            reader = imageio.get_reader(filename)

            try:
                count = sum(1 for _ in reader)
            finally:
                reader.close()

            plt.close(fig)

            self.assertEqual(count, n_frames)

    def test_headless_recording(self):

        with tempfile.TemporaryDirectory() as tmpdir:
            filename = f"{tmpdir}/headless.mp4"

            rec = DisplayRecorder(filename, fps=5)

            fig, ax = plt.subplots()
            ax.plot([1, 2, 3])
            fig.canvas.draw()

            rec.local_inputs = {"in_windows": [IntValue(fig.number)]}
            rec.trigger()
            rec.finalize()

            self.assertTrue(os.path.exists(filename))
