import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio   # from scikit-image

from specula.connections import InputList
from specula.scalar_values import IntValue
from specula.base_processing_obj import InputDesc
from specula.base_processing_obj import BaseProcessingObj

class DisplayRecorder(BaseProcessingObj):

    def __init__(self,
                 filename: str,
                 fps: int=10,
                 codec: str='libx264',
                 ):

        super().__init__()
        self.writer = imageio.get_writer(
            filename,
            fps=fps,
            codec=codec,
            )

        self.inputs['in_windows'] = InputList(type=IntValue)

    @classmethod
    def input_names(cls):
        return {'in_windows': InputDesc(IntValue, 'Window IDs of the windows to record')}

    def trigger(self):
        windows = self.local_inputs['in_windows']
        frames = []
        for window in windows:
            fig = plt.figure(num=window.value)
            buffer = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buffer[:, :, :3])

        if len(windows) > 1:
            try:
                frame = np.hstack(frames)
            except ValueError:
                frame = np.vstack(frames)
        else:
            frame = frames[0]

        self.writer.append_data(frame)

    def finalize(self):
        self.writer.close()
