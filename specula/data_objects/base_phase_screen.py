from specula.base_data_obj import BaseDataObj

class BasePhaseScreen(BaseDataObj):
    """
    Atmospheric phase screens base data object.
    Defines the standard interface for extracting interpolated patches.
    """
    def __init__(self, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

    def extract_phase(self, shift_step: int, angle_deg: float, output_size: int):
        """
        Extracts a patch of size (output_size, output_size)
        at position (pos_x, pos_y) in continuous pixel coordinates.

        Parameters
        ----------
        shift_step : int
            Step size for shifting the phase screen.
        angle_deg : float
            Rotation angle in degrees for the phase screen.

        output_size : int
            Size of the side (in pixels) of the requested square patch.

        Returns
        -------
        ndarray
            The interpolated patch of the phase screen, ready for the GPU.
        """
        raise NotImplementedError("Subclasses must implement extract_phase")
