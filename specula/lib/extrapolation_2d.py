import warnings

import numpy as np
from scipy.ndimage import binary_dilation

from specula import to_xp, cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.lib.interp2d import Interp2D


def _calculate_extrapolation_indices_coeffs(mask, threshold=1e-3):
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
    n_edge_pixels = len(edge_pixels[0])

    # By default we consider that no more than a fraction (between 100% and 25%)
    # of the overall pixels can be edge pixels. This is used to allocate fixed-size
    # arrays for GPU compatibility. Linear interpolation: 1.0 for side<=3, 0.25 for
    # side>=128
    edge_frac = 1.0 - 0.75 * min(max(max(mask.shape) - 3, 0) / 124, 1)
    max_edge_pixels = int(round(edge_frac * mask.shape[0] * mask.shape[1]/2)*2)
    # this if statement is used to avoid errors with peculiar masks with a very
    # high count of edge pixels
    if n_edge_pixels > max_edge_pixels:
        max_edge_pixels = n_edge_pixels
        warnings.warn(f"Number of edge pixels ({n_edge_pixels}) exceeds"
                      f" the default maximum ({max_edge_pixels}).",
                      RuntimeWarning)

    # Arrays with fixed size
    edge_pixels_fixed = np.full(max_edge_pixels, -1, dtype=np.int32)
    reference_indices_fixed = np.full((max_edge_pixels, 8), -1, dtype=np.int32)
    coefficients_fixed = np.full((max_edge_pixels, 8), np.nan, dtype=np.float32)

    # Use the first n_edge_pixels to fill the fixed arrays
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


def _apply_extrapolation(data, edge_pixels, reference_indices,
                         coefficients, valid_indices, out=None, xp=np):
    """
    Applies linear extrapolation to edge pixels using precalculated indices and coefficients.

    Parameters:
        data (ndarray): Input array to extrapolate.
        edge_pixels (ndarray): Linear indices of edge pixels to extrapolate.
        reference_indices (ndarray): Indices of reference pixels.
        coefficients (ndarray): Coefficients for linear extrapolation.
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


class EFInterpolator():
    '''
    Interpolate the amplitude and phase of an ElectricField object using edge extrapolation.
    '''

    __ef_cache = {}  # Shared cache for ElectricField objects
    __zeros_cache = {}  # Shared cache for all EFInterpolator instances

    def _zeros_common(self, shape, dtype):
        """
        Wrapper around self.xp.zeros to enable reuse cache.
        None of the arrays allocated here should be used in 
        prepare_trigger() or post_trigger().
        
        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : dtype
            Data type
            
        Returns
        -------
        array : ndarray
            Array from cache
        """
        key = (self.target_device_idx, shape, dtype)
        if key not in self.__zeros_cache:
            self.__zeros_cache[key] = self.xp.zeros(shape, dtype=dtype)
        return self.__zeros_cache[key]

    def __init__(self,
                 in_ef: ElectricField,
                 out_shape: int,
                 rotAnglePhInDeg: float=0,
                 xShiftPhInPixel: float=0,
                 yShiftPhInPixel: float=0,
                 mask_threshold: float=1e-3,
                 force_extrapolation: bool=False,
                 use_out_ef_cache: bool=False,
                 target_device_idx: int=None,
                 precision: int=None
                 ):
        '''
        Initialize an EFInterpolator object for interpolating an ElectricField,
        with phase extrapolation to avoid edge effects.

        Parameters
        ----------
        in_ef : ElectricField
            Input ElectricField object to be interpolated.
        out_shape : tuple of int
            Desired shape (rows, cols) of the output ElectricField.
        rotAnglePhInDeg : float, optional
            Rotation angle in degrees to apply to the sampling grid (default: 0).
        xShiftInPixel : float, optional
            Horizontal shift (in pixels) to apply to the sampling grid (default: 0).
        yShiftInPixel : float, optional
            Vertical shift (in pixels) to apply to the sampling grid (default: 0).
        mask_threshold : float, optional
            Threshold below which amplitude values are considered 0 for extrapolation
            (default: 1e-3).
        force_extrapolation : bool, optional
            If True, forces extrapolation even if not strictly needed (default: False).
        use_out_ef_cache : bool, optional
            If True, enables caching of output ElectricField objects to save memory and
            allocation time (default: False).
            WARNING: when enabled, output ElectricField objects are cached and saving them
                     with dataStore (or similar mechanisms) will not work. Only the last
                     output ElectricField created with a specific shape and parameters will
                     be valid, all the others will be overwritten by the cache.
        target_device_idx : int, optional
            Target device index for GPU computation (default: None).
        precision : int, optional
            Precision for GPU computation (default: None).

        Output EF is allocated internally and can be retrieved with the interpolated_ef() method.
        '''

        if out_shape[0] / in_ef.size[0] != out_shape[1] / in_ef.size[1]:
            raise ValueError("Output shape must have the same aspect ratio"
                             " as input ElectricField size.")

        oversampling_factor = out_shape[0] / in_ef.size[0]

        self.debug_output = False
        self.mask_threshold = mask_threshold
        self.in_ef = in_ef
        self.force_extrapolation = force_extrapolation
        self.target_device_idx = target_device_idx
        self.use_out_ef_cache = use_out_ef_cache

        if (in_ef.size == out_shape and
            rotAnglePhInDeg == 0 and
            xShiftPhInPixel == 0 and
            yShiftPhInPixel == 0 and force_extrapolation is False):
            self.do_interpolation = False
            self.out_ef = in_ef
            return

        self.do_interpolation = True

        self.edge_pixels = None
        self.reference_indices = None
        self.coefficients = None
        self.valid_indices = None
        self.amplitude_is_binary = None

        if self.use_out_ef_cache:
            # Cache out_ef by shape and parameters
            ef_key = (out_shape, in_ef.pixel_pitch / oversampling_factor,
                      target_device_idx, precision)
            if ef_key not in self.__ef_cache:
                self.__ef_cache[ef_key] = ElectricField(
                    out_shape[0],
                    out_shape[1],
                    in_ef.pixel_pitch / oversampling_factor,
                    target_device_idx=target_device_idx,
                    precision=precision
                )
            self.out_ef = self.__ef_cache[ef_key]
        else:
            # Create a new ElectricField without caching
            self.out_ef = ElectricField(
                out_shape[0],
                out_shape[1],
                in_ef.pixel_pitch / oversampling_factor,
                target_device_idx=target_device_idx,
                precision=precision
            )

        xp = self.out_ef.xp
        dtype = self.out_ef.dtype

        self.interp = Interp2D(
            in_ef.size,
            out_shape,
            -rotAnglePhInDeg,  # Negative angle for PASSATA compatibility
            xShiftPhInPixel,
            yShiftPhInPixel,
            dtype=dtype,
            xp=xp
        )

        self.xp = xp

        # Use cache for phase_extrapolated
        self.phase_extrapolated = self._zeros_common(
            in_ef.size,
            dtype
        )

        self.extrapolation_initialized = False

    def interpolated_ef(self):
        '''
        Returns the interpolated ElectricField object.
        '''
        return self.out_ef

    def interpolate(self):
        '''
        Perform interpolation with edge extrapolation.
        '''

        if not self.do_interpolation:
            return

        if self.extrapolation_initialized is False:
            # Calculate extrapolation data only once
            (edge_pixels, reference_indices,
             coefficients, valid_indices) = \
                _calculate_extrapolation_indices_coeffs(
                    cpuArray(self.in_ef.A),
                    threshold=self.mask_threshold
                )

            # Convert to xp
            self.edge_pixels = to_xp(self.xp, edge_pixels)
            self.reference_indices = to_xp(self.xp, reference_indices)
            self.coefficients = to_xp(self.xp, coefficients)
            self.valid_indices = to_xp(self.xp, valid_indices)

            # Check if input amplitude is binary (all values close to 0 or 1) with tolerance
            unique_values = self.xp.unique(self.in_ef.A)
            tol = 1e-3
            is_binary = self.xp.all(
                self.xp.logical_or(
                    self.xp.abs(unique_values - 0) < tol,
                    self.xp.abs(unique_values - 1) < tol
                )
            )
            self.amplitude_is_binary = is_binary
            self.extrapolation_initialized = True

        # Amplitude: simple interpolation
        self.interp.interpolate(self.in_ef.A, out=self.out_ef.A)

        # Apply binary threshold if input amplitude was binary
        if self.amplitude_is_binary:
            self.out_ef.A[:] = self.out_ef.A > 0.5

        # Phase: apply an intermediate extrapolation to avoid edge effects
        self.phase_extrapolated[:] = self.in_ef.phaseInNm * \
            (self.in_ef.A >= self.mask_threshold).astype(int)

        _ = _apply_extrapolation(
            self.in_ef.phaseInNm,
            self.edge_pixels,
            self.reference_indices,
            self.coefficients,
            self.valid_indices,
            out=self.phase_extrapolated,
            xp=self.xp
        )

        if self.debug_output:
            # compare input and extrapolated phase
            phase_in_nm = self.in_ef.phaseInNm * (self.in_ef.A >= 1e-3).astype(int)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(20, 5))
            plt.subplot(1, 4, 1)
            plt.imshow(cpuArray(phase_in_nm), origin='lower', cmap='gray')
            plt.title('Input Phase')
            plt.colorbar()
            plt.subplot(1, 4, 2)
            plt.imshow(cpuArray(self.phase_extrapolated), origin='lower', cmap='gray')
            plt.title('Extrapolated Phase')
            plt.colorbar()
            plt.subplot(1, 4, 3)
            plt.imshow(cpuArray(self.phase_extrapolated - phase_in_nm), origin='lower', cmap='gray')
            plt.title('Phase Difference')
            plt.colorbar()
            plt.subplot(1, 4, 4)
            plt.imshow(cpuArray(self.in_ef.A), origin='lower', cmap='gray')
            plt.title('Input Electric Field Amplitude')
            plt.colorbar()
            plt.show()

        self.interp.interpolate(self.phase_extrapolated, out=self.out_ef.phaseInNm)

        # Copy other properties
        self.out_ef.S0 = self.in_ef.S0
