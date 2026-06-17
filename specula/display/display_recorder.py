from specula.display.base_display import BaseDisplay
import matplotlib.pyplot as plt
import imageio.v2 as imageio   # from scikit-image

class DisplayRecorder(BaseDisplay):

    def __init__(self,
                 window,
                 filename,
                 fps: int=10,
                 codec: str='libx264',
                 ):

        super().__init__()
        writer = imageio.get_writer(
                filename,
                fps=fps,
                codec=codec,
                )
        # Video recording is handled in the base class, because
        # we cannot guarantee that this object is triggered last
        self.register_writer(writer, window_id=window)
        self.writer = writer

    def _create_figure(self):
        # Override _create_figure to disable window opening
        pass

    def trigger(self):
        # Do nothing. The video writer will be c
        pass

    def finalize(self):
        self.writer.close()
