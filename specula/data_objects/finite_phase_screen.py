from specula.base_data_obj import BaseDataObj

class FinitePhaseScreen(BaseDataObj):
    """
    Finite phase screen data object based on a pre-generated static map.
    It uses modulo arithmetic to cycle indefinitely over the map
    and pure GPU bilinear interpolation for sub-pixel shifts.
    """
    def __init__(self, full_screen, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Ensure the screen is on the correct device/format based on the inherited self.xp
        self.screen = self.xp.asarray(full_screen, dtype=self.dtype)
        self.height, self.width = self.screen.shape


    def extract_phase(self, shift_step: int, angle_deg: float, output_size: int):
        """
        Extracts a sub-screen by shifting along the X-axis and applying rotation.
        
        Args:
            shift_step: The number of pixels to shift along the specific direction.
            angle_deg: The rotation angle in degrees.
            output_size: The size (height and width) of the extracted square sub-screen.
            
        Returns:
            A 2D array (numpy or cupy) representing the evolved phase screen.
        """
        # --- 1. SHIFT OPERATION ---
        max_shift = self.width - output_size
        if max_shift <= 0:
             raise ValueError("Screen width must be larger than output_size.")

        start_x = shift_step % max_shift
        start_y = (self.height - output_size) // 2

        shifted_sub_screen = self.screen[
            start_y : start_y + output_size,
            start_x : start_x + output_size
        ]

        # --- 2. ROTATION OPERATION ---
        # Uses the inherited ndimage_rotate function dynamically assigned by BaseTimeObj
        rotated_sub_screen = self.ndimage_rotate(
            shifted_sub_screen,
            angle=angle_deg,
            reshape=False,
            order=1
        )

        return rotated_sub_screen
