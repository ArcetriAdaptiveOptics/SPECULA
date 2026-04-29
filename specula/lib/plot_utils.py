import numpy as np
import matplotlib.pyplot as plt
from specula import cpuArray
from specula.data_objects.ifunc import IFunc

def display_ifunc_2d(ifunc_obj: IFunc, m2c_array=None, modal_vector=None,
                     id_mode_starting: int = 0, n_raw_col: int = 10,
                     do_not_show_ticks: bool = False, show_plot: bool = True):
    """
    Displays an influence function or reconstructed modes from an IFunc object.
    
    Args:
        ifunc_obj (IFunc): The Influence Function object containing the modal base and mask.
        m2c_array (array-like, optional): Modal to Command matrix to multiply the base.
        modal_vector (array-like, optional): Vector(s) of modal coefficients to reconstruct shapes.
        id_mode_starting (int, optional): The ID of the first mode to display. Defaults to 0.
        n_raw_col (int, optional): Number of rows/cols for the grid display. Defaults to 10.
        do_not_show_ticks (bool, optional): If True, hides the axes ticks. Defaults to False.
        show_plot (bool, optional): If True, displays the plot. Set to False for testing.
                                    Defaults to True.

    Returns:
        numpy.ndarray: The computed 2D shape or grid of shapes.
    """

    # --- 1. Extract and format data from the IFunc object ---
    # We use cpuArray to safely bring the data to CPU memory as a numpy array,
    # which is required by matplotlib.
    modal_base = cpuArray(ifunc_obj.influence_function).astype(float)
    mask_small = cpuArray(ifunc_obj.mask_inf_func)

    # Apply Modal to Command matrix if provided
    if m2c_array is not None:
        m2c = np.array(cpuArray(m2c_array), dtype=float)
        modal_base = m2c @ modal_base

    # Create a boolean mask for faster array indexing in Python
    mask_small_bool = mask_small > 0
    mask_small_size = mask_small.shape[0]

    # --- 2. Inner helper function for plotting ---
    def show_img(data, title):
        plt.figure(figsize=(12, 9))
        max_abs = np.max(np.abs(data))

        # We transpose the data (data.T) for matplotlib (Y, X) to match IDL's [X, Y] logic.
        # origin='lower' ensures the origin is at the bottom-left, just like in IDL.
        plt.imshow(data.T, origin='lower', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        plt.title(title, fontsize=14)
        plt.colorbar()

        if do_not_show_ticks:
            plt.axis('off') # Hides axes ticks

        if show_plot:
            plt.show()

    shape_out = None

    # --- 3. Visualization Logic ---

    # CASE A: A modal vector is provided, we reconstruct the shape
    if modal_vector is not None:
        modal_vector = np.array(modal_vector, dtype=float)

        # 1D Vector: Single frame reconstruction
        if modal_vector.ndim == 1:
            shape = np.zeros((mask_small_size, mask_small_size), dtype=float)
            n_elements = len(modal_vector)
            mb_subset = modal_base[0:n_elements, :]

            # Vector-Matrix multiplication (replaces IDL's vecmat_multiply)
            shape[mask_small_bool] = modal_vector @ mb_subset

            show_img(shape, 'Reconstructed Shape')
            shape_out = shape

        # 2D Vector: Multiple frames reconstruction (e.g., num_frames x num_modes)
        else:
            num_frames = modal_vector.shape[0]
            num_modes = modal_vector.shape[1]

            shape = np.zeros((mask_small_size, mask_small_size, num_frames), dtype=float)
            mb_subset = modal_base[0:num_modes, :]

            for i in range(num_frames):
                temp = np.zeros((mask_small_size, mask_small_size), dtype=float)
                temp[mask_small_bool] = modal_vector[i, :] @ mb_subset
                shape[:, :, i] = temp

            # Show the very last frame computed (-1 index in Python)
            show_img(shape[:, :, -1], 'Reconstructed Shape (Last Frame)')
            shape_out = shape

    # CASE B: No modal vector provided, display a grid of influence functions
    else:
        id_mode_starting = int(id_mode_starting)
        max_mode = modal_base.shape[0]

        # Initialize a large array for the mosaic grid
        shape_big = np.zeros((n_raw_col * mask_small_size, n_raw_col * mask_small_size),
                             dtype=float)

        for i in range(n_raw_col):
            for j in range(n_raw_col):
                # Calculate the current mode index based on grid position
                id_mode = id_mode_starting + i + (n_raw_col - 1 - j) * n_raw_col

                if id_mode < max_mode:
                    shape = np.zeros((mask_small_size, mask_small_size), dtype=float)
                    shape[mask_small_bool] = modal_base[id_mode, :]

                    # Map the small shape into the correct position in the big grid array
                    # Logic is kept identical to IDL since we maintained the [X, Y] convention
                    shape_big[i * mask_small_size : (i + 1) * mask_small_size,
                              j * mask_small_size : (j + 1) * mask_small_size] = shape

        title_str = f"Mode {id_mode_starting} - {id_mode_starting + n_raw_col**2 - 1}"
        show_img(shape_big, title_str)
        shape_out = shape_big

    return shape_out
