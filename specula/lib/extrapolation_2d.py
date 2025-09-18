import numpy as np
from scipy.ndimage import binary_dilation
from specula import cpuArray

def calculate_extrapolation_indices_coeffs(mask, threshold=1e-3):
    """
    Calculates indices and coefficients for extrapolating edge pixels of a mask.

    Parameters:
        mask (ndarray): Binary mask (True/1 inside, False/0 outside).
        threshold (float): Threshold below which values are considered 0/False.

    Returns:
        tuple: (edge_pixels, reference_indices, coefficients)
            - edge_pixels: Linear indices of the edge pixels to extrapolate.
            - reference_indices: Array of reference pixel indices for extrapolation.
            - coefficients: Coefficients for linear extrapolation.
    """

    # Convert the mask to boolean with threshold
    binary_mask = cpuArray(mask) >= threshold

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

    # Calculate valid indices here
    valid_edge_mask = (edge_pixels_fixed >= 0) & ~np.isnan(coefficients_fixed[:, 0])
    valid_indices = np.where(valid_edge_mask)[0]

    return edge_pixels_fixed, reference_indices_fixed, coefficients_fixed, valid_indices

def apply_extrapolation(data, edge_pixels, reference_indices, coefficients, valid_indices, out=None, xp=np):
    """
    Applies linear extrapolation to edge pixels using precalculated indices and coefficients.

    Parameters:
        data (ndarray): Input array to extrapolate.
        edge_pixels (ndarray): Linear indices of edge pixels to extrapolate.
        reference_indices (ndarray): Indices of reference pixels.
        coefficients (ndarray): Coefficients for linear extrapolation.ù
        valid_indices (ndarray): Indices of valid edge pixels.
        xp (np): NumPy or CuPy module for array operations.

    Returns:
        ndarray: Array with extrapolated pixels.
    """
    if out is None:
        out = data.copy()
    flat_out = out.ravel()
    flat_data = data.ravel()

    # Vectorized extrapolation for valid edge pixels
    if len(valid_indices) > 0:

        edge_pixels = xp.asarray(edge_pixels)
        reference_indices = xp.asarray(reference_indices)
        coefficients = xp.asarray(coefficients)
        valid_indices = xp.asarray(valid_indices)

        # Extract valid edge pixels, reference indices, and coefficients
        valid_edge_pixels = edge_pixels[valid_indices]
        valid_ref_indices = reference_indices[valid_indices]
        valid_coeffs = coefficients[valid_indices]

        # Create a mask for valid reference indices (>= 0)
        valid_ref_mask = valid_ref_indices >= 0

        # Replace invalid indices with 0 to avoid indexing errors
        safe_ref_indices = xp.where(valid_ref_mask, valid_ref_indices, 0)

        # Get data values for all reference indices at once
        ref_data = flat_data[safe_ref_indices]  # Shape: (n_valid_edges, 8)

        # Zero out contributions from invalid references
        masked_coeffs = xp.where(valid_ref_mask, valid_coeffs, 0.0)

        # Compute all contributions at once and sum across reference positions
        contributions = masked_coeffs * ref_data  # Element-wise multiplication
        extrap_values = xp.sum(contributions, axis=1)  # Sum across reference positions

        # Assign extrapolated values to edge pixels
        flat_out[valid_edge_pixels] = extrap_values

    return out
