import numpy as np
from specula.lib.make_mask import make_mask


def compute_petal_ifunc(dim, n_petals, xp=np, dtype=np.float32, angle_offset=0,
                        obsratio=0.0, diaratio=1.0, mask=None, spider=False,
                        spider_width=2, add_tilts=False, return_coordinates=False,
                        special_last_petal=False):
    """
    Computes influence functions for a segmented mirror with n_petals segments.
    Optionally adds spider arms shadows between petals and tip/tilt modes.
    
    Parameters:
    -----------
    dim : int
        Dimension of the output array (square array of dim x dim)
    n_petals : int
        Number of petals/segments in the pupil
    xp : module
        Numerical module to use (numpy or cupy)
    dtype : data-type
        Data type of the output array
    angle_offset : float
        Rotation angle offset in degrees
    obsratio : float
        Central obstruction ratio (0 to 1)
    diaratio : float
        Diameter ratio (0 to 1)
    mask : array_like, optional
        Predefined mask to use
    spider : bool
        Whether to add spider arms between petals
    spider_width : int
        Width of spider arms in pixels
    add_tilts : bool
        Whether to add tip and tilt modes for each petal
    return_coordinates : bool
        Whether to return the coordinates array
    special_last_petal : bool
        Whether to use logical_or for the last petal to handle wrap-around angles.
        
    Returns:
    --------
    ifs_2d : array_like
        2D array of influence functions (n_modes x n_points)
        If add_tilts=True, n_modes = 3*n_petals (piston, tip, tilt for each)
        If add_tilts=False, n_modes = n_petals (only piston for each)
    mask : array_like
        Pupil mask
    coordinates : array_like, optional
        Coordinates of the petals centroids
    """

    # Create mask if not provided
    if mask is None:
        # Use updated make_mask function with spider support
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp,
                             spider=spider, spider_width=spider_width,
                             n_petals=n_petals, angle_offset=angle_offset)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)

    # Center coordinates
    center = dim / 2
    y, x = xp.mgrid[:dim, :dim]
    y = y - center
    x = x - center

    # Convert to polar coordinates
    r = xp.sqrt(x**2 + y**2)
    theta = xp.arctan2(y, x) + xp.radians(angle_offset)

    # Wrap angles to [0, 2π]
    theta = (theta + 2*xp.pi) % (2*xp.pi)
    
    # Create petals
    petal_angle = 2 * xp.pi / n_petals
    petal_masks = []

    # Create individual petal masks
    for i in range(n_petals):
        # Determine angle bounds for this petal
        angle_min = i * petal_angle
        angle_max = (i + 1) * petal_angle

        # Create mask for this petal
        if i == n_petals - 1 and special_last_petal:
            petal_mask = xp.logical_or(
                theta >= angle_min,
                theta < angle_max
            ) * mask
        else:
            petal_mask = xp.logical_and(
                theta >= angle_min,
                theta < angle_max
            ) * mask

        petal_masks.append(petal_mask)

    # Determine total number of influence functions
    n_modes = 3 * n_petals if add_tilts else n_petals

    # Create influence functions
    ifs_cube = xp.zeros((n_modes, dim, dim), dtype=dtype)

    # Add piston, tip, and tilt modes for each petal
    for i in range(n_petals):
        # Piston mode
        ifs_cube[i] = petal_masks[i]

        if add_tilts:
            # Create tip mode (x gradient)
            tip = x * petal_masks[i]
            # Get non-zero indices
            nonzero = xp.where(petal_masks[i] > 0)
            if len(nonzero[0]) > 0:
                # Normalize to zero mean and unit std deviation
                tip_values = tip[nonzero]
                tip[nonzero] = (tip_values - xp.mean(tip_values)) / (xp.std(tip_values) or 1.0)
            ifs_cube[n_petals + i] = tip

            # Create tilt mode (y gradient)
            tilt = y * petal_masks[i]
            if len(nonzero[0]) > 0:
                # Normalize to zero mean and unit std deviation
                tilt_values = tilt[nonzero]
                tilt[nonzero] = (tilt_values - xp.mean(tilt_values)) / (xp.std(tilt_values) or 1.0)
            ifs_cube[2 * n_petals + i] = tilt

    # Create 2D array of influence functions
    ifs_2d = xp.array([ifs_cube[i][idx] for i in range(n_modes)], dtype=dtype)

    # Create coordinates array if requested
    if return_coordinates:
        # Calculate centroids for each petal
        coordinates = xp.zeros((2, n_petals))
        for i in range(n_petals):
            where_petal = xp.where(petal_masks[i])
            if len(where_petal[0]) > 0:  # Make sure the petal contains points
                coordinates[0, i] = xp.mean(where_petal[1])  # x coordinate
                coordinates[1, i] = xp.mean(where_petal[0])  # y coordinate

        return ifs_2d, mask, coordinates
    else:
        return ifs_2d, mask