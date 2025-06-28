import numpy as np
from scipy.ndimage import binary_dilation
from specula import cpuArray

def calculate_extrapolation_indices_coeffs(mask):
    """
    Calculates indices and coefficients for extrapolating edge pixels of a mask.

    Parameters:
        mask (ndarray): Binary mask (True/1 inside, False/0 outside).

    Returns:
        tuple: (edge_pixels, reference_indices, coefficients)
            - edge_pixels: Linear indices of the edge pixels to extrapolate.
            - reference_indices: Array of reference pixel indices for extrapolation.
            - coefficients: Coefficients for linear extrapolation.
    """

    # Convert the mask to boolean
    binary_mask = cpuArray(mask).astype(bool)

    # Identify edge pixels (outside but adjacent to the mask) using binary dilation
    dilated_mask = binary_dilation(binary_mask)
    edge_pixels = np.where(dilated_mask & ~binary_mask)

    # No more than 50% of the overall pixel can be edge pixels
    max_edge_pixels = int(0.5 * mask.shape[0] * mask.shape[1])

    # Arrays with fixed size
    edge_pixels_fixed = np.full(max_edge_pixels, -1, dtype=np.int32)
    reference_indices_fixed = np.full((max_edge_pixels, 8), -1, dtype=np.int32)
    coefficients_fixed = np.full((max_edge_pixels, 8), np.nan, dtype=np.float32)

    # Use the first n_edge_pixels to fill the fixed arrays
    n_edge_pixels = len(edge_pixels[0])
    edge_pixels_linear = np.ravel_multi_index(edge_pixels, mask.shape)
    edge_pixels_fixed[:n_edge_pixels] = edge_pixels_linear

    # Directions for extrapolation (y+1, y-1, x+1, x-1)
    directions = [
        (1, 0),  # y+1 (down)
        (-1, 0), # y-1 (up)
        (0, 1),  # x+1 (right)
        (0, -1)  # x-1 (left)
    ]

    # Iterate over each edge pixel
    problem_indices = []
    for i, (y, x) in enumerate(zip(*edge_pixels)):
        valid_directions = 0

        # Examine the 4 directions
        for dir_idx, (dy, dx) in enumerate(directions):
            # Coordinates of reference points at distance 1 and 2
            y1, x1 = y + dy, x + dx
            y2, x2 = y + 2*dy, x + 2*dx

            # Check if the points are valid (inside the image and inside the mask)
            valid_ref1 = (0 <= y1 < mask.shape[0] and 
                          0 <= x1 < mask.shape[1] and 
                          binary_mask[y1, x1])

            valid_ref2 = (0 <= y2 < mask.shape[0] and 
                          0 <= x2 < mask.shape[1] and 
                          binary_mask[y2, x2])

            if valid_ref1:
                # Index of the first reference point (linear index)
                ref_idx1 = y1 * mask.shape[1] + x1
                reference_indices_fixed[i, 2*dir_idx] = ref_idx1

                if valid_ref2:
                    # Index of the second reference point (linear index)
                    ref_idx2 = y2 * mask.shape[1] + x2
                    reference_indices_fixed[i, 2*dir_idx + 1] = ref_idx2

                    # Coefficients for linear extrapolation: 2*P₁ - P₂
                    coefficients_fixed[i, 2*dir_idx] = 2.0
                    coefficients_fixed[i, 2*dir_idx + 1] = -1.0
                    valid_directions += 1
                else:
                    # If the second point is invalid, check if it's the only valid pixel
                    if valid_directions == 0:
                        coefficients_fixed[i, 2*dir_idx] = 1.0
                        valid_directions += 1
                    else:
                        # Set coefficients to 0
                        coefficients_fixed[i, 2*dir_idx] = 0.0
                        coefficients_fixed[i, 2*dir_idx + 1] = 0.0
            else:
                # Set coefficients to 0 if the first reference is invalid
                coefficients_fixed[i, 2*dir_idx] = 0.0
                coefficients_fixed[i, 2*dir_idx + 1] = 0.0

        # Normalize coefficients based on the number of valid directions
        if valid_directions > 1:
            factor = 1.0 / valid_directions
            for dir_idx in range(4):
                if coefficients_fixed[i, 2*dir_idx] != 0:
                    coefficients_fixed[i, 2*dir_idx] *= factor
                    if coefficients_fixed[i, 2*dir_idx + 1] != 0:
                        coefficients_fixed[i, 2*dir_idx + 1] *= factor

    return edge_pixels_fixed, reference_indices_fixed, coefficients_fixed

def apply_extrapolation(data, edge_pixels, reference_indices, coefficients, xp=np):
    """
    Applies linear extrapolation to edge pixels using precalculated indices and coefficients.

    Parameters:
        data (ndarray): Input array to extrapolate.
        edge_pixels (ndarray): Linear indices of edge pixels to extrapolate.
        reference_indices (ndarray): Indices of reference pixels.
        coefficients (ndarray): Coefficients for linear extrapolation.
        xp (np): NumPy or CuPy module for array operations.

    Returns:
        ndarray: Array with extrapolated pixels.
    """
    # Create a copy of the input array
    result = data.copy()
    flat_result = result.ravel()
    flat_data = data.ravel()

    # Mask for valid coefficients (not NaN in the first column)
    valid_edge_mask = (edge_pixels >= 0) & ~xp.isnan(coefficients[:, 0])
    valid_indices = xp.where(valid_edge_mask)[0]

    # Iterate over each valid edge pixel
    for i in valid_indices:
        edge_idx = edge_pixels[i]
        # Initialize the extrapolated value
        extrap_value = 0.0

        # Sum contributions from all references
        for j in range(reference_indices.shape[1]):
            ref_idx = reference_indices[i, j]
            if ref_idx >= 0:  # If the index is valid
                contrib = coefficients[i, j] * flat_data[ref_idx]
                extrap_value += contrib

        # Assign the extrapolated value
        flat_result[edge_idx] = extrap_value

    return result