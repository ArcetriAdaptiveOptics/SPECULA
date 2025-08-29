from specula.lib.make_xy import make_xy

def closest(value, array, xp):
    """Find the closest value in the array and return its index."""
    return (xp.abs(array - value)).argmin()

def make_mask(np_size, obsratio=0.0, diaratio=1.0, xc=0.0, yc=0.0, 
              square=False, inverse=False, centeronpixel=False, get_idx=False, xp=None,
              spider=False, spider_width=0, n_petals=0, angle_offset=0):
    """
    Create a mask array with optional spider arms.

    Parameters:
        np_size (int): Frame size [px].
        obsratio (float): Diameter of the obstruction (fraction of pupil diameter). Default is 0.0.
        diaratio (float): Diameter of the pupil (fraction of frame size). Default is 1.0.
        xc (float): X-center of the pupil (fraction of frame size). Default is 0.0.
        yc (float): Y-center of the pupil (fraction of frame size). Default is 0.0.
        square (bool): If True, make a square mask.
        inverse (bool): If True, invert the mask (1->0, 0->1).
        centeronpixel (bool): If True, move the center of the pupil to the nearest pixel.
        get_idx: if True, return a tuple with (mask, idx)
        xp: numerical module to use (numpy or cupy)
        spider (bool): Whether to add spider arms between segments.
        spider_width (int): Width of spider arms in pixels.
        n_petals (int): Number of petals/segments for spider arms.
        angle_offset (float): Rotation angle offset in degrees for spider arms.
    
    Returns:
        mask (numpy.ndarray): Array representing the mask with the specified properties.
        idx (numpy.ndarray): Array of indices inside the pupil. (only if get_idx is True)
    """
    if diaratio is None:
        diaratio = 1.0
    if obsratio is None:
        obsratio = 0.0
    if xp is None:
        import numpy as xp

    # Generate coordinate grids
    x, y = make_xy(sampling=np_size, ratio=1.0, xp=xp)

    # Adjust center if centeronpixel is set
    if centeronpixel:
        idx_x = closest(xc, x[:, 0], xp=xp)
        neighbours_x = [abs(x[idx_x-1, 0] - xc), abs(x[idx_x+1, 0] - xc)]
        idxneigh_x = xp.argmin(xp.array(neighbours_x))
        kx = -0.5 if idxneigh_x == 0 else 0.5
        xc = x[idx_x, 0] + kx * (x[1, 0] - x[0, 0])

        idx_y = closest(yc, y[0, :], xp=xp)
        neighbours_y = [abs(y[0, idx_y-1] - yc), abs(y[0, idx_y+1] - yc)]
        idxneigh_y = xp.argmin(xp.array(neighbours_y))
        ky = -0.5 if idxneigh_y == 0 else 0.5
        yc = y[0, idx_y] + ky * (y[0, 1] - y[0, 0])

    ir = obsratio

    # Generate mask based on the square or circular option
    if square:
        idx = xp.where(
            (xp.abs(x - xc) <= diaratio) & (xp.abs(y - yc) <= diaratio) &
            ((xp.abs(x - xc) >= diaratio * ir) | (xp.abs(y - yc) >= diaratio * ir))
        )
    else:
        idx = xp.where(
            ((x - xc) ** 2 + (y - yc) ** 2 < diaratio ** 2) &
            ((x - xc) ** 2 + (y - yc) ** 2 >= (diaratio * ir) ** 2)
        )

    # Create the mask
    mask = xp.zeros((np_size, np_size), dtype=xp.uint8)
    mask[idx] = 1

    # Add spider arms if requested
    if spider and n_petals > 0 and spider_width > 0:
        # Convert spider arguments
        center = np_size / 2

        # Create centered coordinate system
        y_centered, x_centered = xp.mgrid[:np_size, :np_size]
        y_centered = y_centered - center
        x_centered = x_centered - center

        # Create a float spider mask for better resolution
        spider_mask_float = xp.zeros((np_size, np_size), dtype=float)

        # Calculate angle between petals
        petal_angle = 2 * xp.pi / n_petals

        # Process each spider arm
        for i in range(n_petals):
            # Calculate the angle for this spider arm
            angle = i * petal_angle + xp.radians(angle_offset)
            cos_a, sin_a = xp.cos(angle), xp.sin(angle)

            # Calculate maximum distance to edge (slightly larger than radius)
            max_dist = int(xp.sqrt(2) * np_size / 2)

            # Use finer steps for better coverage
            step_size = 0.5  # Sub-pixel stepping

            # Spider width with buffer
            effective_width = spider_width * 1.2  # 20% buffer for better coverage

            # Trace the spider arm with finer resolution
            for d in xp.arange(0, max_dist, step_size):
                # Keep coordinates as float for better precision
                x_pos = center + d * cos_a
                y_pos = center + d * sin_a

                # Skip if outside the array bounds
                if (x_pos < 0 or x_pos >= np_size or y_pos < 0 or y_pos >= np_size):
                    continue

                # Add spider width as a smooth radial function from the line
                half_width = effective_width / 2

                # Use a grid approach to set all pixels within the width
                for w in xp.arange(-half_width, half_width + 0.5, 0.5):
                    # Calculate perpendicular coordinates (keep as float)
                    x_perp = x_pos + w * -sin_a
                    y_perp = y_pos + w * cos_a

                    # Convert to integer indices only for array access
                    x_idx = int(x_perp)
                    y_idx = int(y_perp)

                    # Skip if outside the array
                    if (0 <= x_idx < np_size and 0 <= y_idx < np_size):
                        # Set value to 1.0 (full coverage)
                        spider_mask_float[y_idx, x_idx] = 1.0

        # Convert the float mask to boolean and apply to the main mask
        spider_mask = spider_mask_float > 0
        mask[spider_mask] = 0

    # Invert the mask if the inverse keyword is set
    if inverse:
        mask = 1 - mask

    if get_idx:
        # Update idx to reflect the complete mask including spider arms
        idx = xp.where(mask)
        return mask, idx
    else:
        return mask