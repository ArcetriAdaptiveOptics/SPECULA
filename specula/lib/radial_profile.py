import numpy as np


def compute_radial_profile(image, center_in_px_y=None, center_in_px_x=None,
                           xp=np, dtype=np.float64, return_counts=False):
    """Compute the azimuthally averaged radial profile of a 2D image.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    center_in_px_y, center_in_px_x : float, optional
        Profile center in pixel coordinates. If not set, the geometric center is used.
    xp : module, optional
        Numpy-like module (`numpy` or `cupy`).
    dtype : data-type, optional
        Accumulation dtype.
    return_counts : bool, optional
        If True, also return the number of pixels in each radial bin.

    Returns
    -------
    profile : ndarray
        Mean image value in each integer radial bin.
    radial_distance : ndarray
        Mean distance of each radial bin in pixels.
    n_px_in_radial_bin : ndarray, optional
        Number of pixels in each radial bin, returned only if `return_counts` is True.
    """
    image = xp.asarray(image)
    if image.ndim != 2:
        raise ValueError('compute_radial_profile expects a 2D image')

    if center_in_px_x is None:
        center_in_px_x = image.shape[1] / 2
    if center_in_px_y is None:
        center_in_px_y = image.shape[0] / 2

    # Coordinates relative to the center, in pixels
    y_coord, x_coord = xp.indices(image.shape, dtype=dtype)
    y_coord = y_coord - dtype(center_in_px_y)
    x_coord = x_coord - dtype(center_in_px_x)
    r_coord = xp.sqrt(x_coord**2 + y_coord**2)

    radial_bin = xp.floor(r_coord).astype(np.int32).ravel()
    image_flat = image.ravel().astype(dtype, copy=False)
    r_coord_flat = r_coord.ravel().astype(dtype, copy=False)

    # Count number of occurrences of each value in radial_bin and
    # sum image values and distances in each bin
    n_px_in_radial_bin = xp.bincount(radial_bin)
    sum_in_radial_bin = xp.bincount(radial_bin, weights=image_flat)
    sum_distance_in_radial_bin = xp.bincount(radial_bin, weights=r_coord_flat)

    # Only keep bins with at least one pixel to avoid division by zero
    valid_bins = n_px_in_radial_bin > 0
    n_px_in_radial_bin = n_px_in_radial_bin[valid_bins]
    # Compute the mean profile in each bin
    profile = sum_in_radial_bin[valid_bins] / n_px_in_radial_bin
    # Compute the mean radial distance in each bin
    radial_distance = sum_distance_in_radial_bin[valid_bins] / n_px_in_radial_bin

    if return_counts:
        return profile, radial_distance, n_px_in_radial_bin
    return profile, radial_distance


def compute_fwhm_from_profile(profile, radial_distance=None, xp=np, dtype=np.float64):
    """Estimate the FWHM from a radial profile using linear interpolation."""
    profile = xp.asarray(profile, dtype=dtype)
    if profile.ndim != 1:
        raise ValueError('profile must be a 1D array')

    if radial_distance is None:
        radial_distance = xp.arange(profile.size, dtype=dtype)
    else:
        radial_distance = xp.asarray(radial_distance, dtype=dtype)

    if radial_distance.ndim != 1 or radial_distance.size != profile.size:
        raise ValueError('radial_distance must be a 1D array with the same size as profile')
    if profile.size == 0:
        return dtype(np.nan)

    peak_value = xp.max(profile)
    if float(peak_value) <= 0.0:
        return dtype(0.0)

    # Find the first bin where the profile drops below half the peak value
    half_maximum = peak_value / dtype(2.0)
    # Find the pixel below and above the half maximum
    below_half = xp.where(profile <= half_maximum)[0]
    below_half = below_half[below_half > 0]
    if below_half.size == 0:
        return dtype(np.nan)

    idx = int(below_half[0])
    r1 = radial_distance[idx - 1]
    r2 = radial_distance[idx]
    p1 = profile[idx - 1]
    p2 = profile[idx]

    if float(p2 - p1) == 0.0:
        half_radius = r1
    else:
        # Linear interpolation to find the radius at half maximum
        half_radius = r1 + (half_maximum - p1) * (r2 - r1) / (p2 - p1)
    return dtype(2.0) * dtype(half_radius)


def compute_encircled_energy(profile, n_px_in_radial_bin=None, radial_distance=None,
                             xp=np, dtype=np.float64, normalize=True):
    """Compute the encircled-energy curve from a radial profile.

    If `n_px_in_radial_bin` is not available, the energy in each bin is approximated
    from the annulus area derived from `radial_distance` (or from equally spaced
    bins if `radial_distance` is not provided).
    """
    profile = xp.asarray(profile, dtype=dtype)
    if profile.ndim != 1:
        raise ValueError('profile must be a 1D array')

    if n_px_in_radial_bin is None:
        if radial_distance is None:
            radial_distance = xp.arange(profile.size, dtype=dtype)
        else:
            radial_distance = xp.asarray(radial_distance, dtype=dtype)
            if radial_distance.shape != profile.shape:
                raise ValueError('radial_distance must have the same shape as profile')

        if profile.size == 0:
            energy_in_radial_bin = profile
        elif profile.size == 1:
            # For a single bin, the annulus area is approximated as a circle with radius equal to
            # the bin's radial distance
            outer_radius = xp.maximum(radial_distance[0], dtype(0.5))
            annulus_weight = xp.asarray([outer_radius**2], dtype=dtype)
            energy_in_radial_bin = profile * annulus_weight
        else:
            # Compute the inner and outer radius of each annulus bin from the radial distance
            # midpoints, assuming the first bin starts at radius 0 and the last bin extends to
            # the next radial distance
            radial_midpoints = dtype(0.5) * (radial_distance[1:] + radial_distance[:-1])
            inner_radius = xp.empty_like(radial_distance)
            outer_radius = xp.empty_like(radial_distance)
            inner_radius[0] = dtype(0.0)
            inner_radius[1:] = radial_midpoints
            outer_radius[:-1] = radial_midpoints
            outer_radius[-1] = radial_distance[-1] + (radial_distance[-1] - inner_radius[-1])
            annulus_weight = xp.maximum(outer_radius**2 - inner_radius**2, 0)
            energy_in_radial_bin = profile * annulus_weight
    else:
        n_px_in_radial_bin = xp.asarray(n_px_in_radial_bin, dtype=dtype)
        if n_px_in_radial_bin.shape != profile.shape:
            raise ValueError('n_px_in_radial_bin must have the same shape as profile')
        # Compute the total energy in each radial bin by multiplying the mean profile value by
        # the number of pixels in that bin
        energy_in_radial_bin = profile * n_px_in_radial_bin

    encircled_energy = xp.cumsum(energy_in_radial_bin, dtype=dtype)
    if normalize and encircled_energy.size > 0:
        total_energy = encircled_energy[-1]
        if float(total_energy) != 0.0:
            encircled_energy = encircled_energy / total_energy
    return encircled_energy


def get_encircled_energy_at_distance(encircled_energy, radial_distance, distance,
                                     xp=np, dtype=np.float64):
    """Return the encircled energy at one or more requested radial distances."""
    encircled_energy = xp.asarray(encircled_energy, dtype=dtype)
    radial_distance = xp.asarray(radial_distance, dtype=dtype)
    query_distance = xp.asarray(distance, dtype=dtype)
    scalar_input = query_distance.ndim == 0
    query_distance = xp.atleast_1d(query_distance)

    if radial_distance.ndim != 1 or encircled_energy.ndim != 1:
        raise ValueError('encircled_energy and radial_distance must be 1D arrays')
    if radial_distance.size != encircled_energy.size:
        raise ValueError('encircled_energy and radial_distance must have the same size')
    if radial_distance.size == 0:
        result = xp.full(query_distance.shape, xp.nan, dtype=dtype)
        return result[0] if scalar_input else result

    idx = xp.searchsorted(radial_distance, query_distance, side='left')
    idx = xp.clip(idx, 0, radial_distance.size - 1)
    result = encircled_energy[idx].astype(dtype, copy=True)

    interior = (idx > 0) & (idx < radial_distance.size)
    idx_clipped = xp.clip(idx, 1, radial_distance.size - 1)
    left_idx = idx_clipped - 1
    r1 = radial_distance[left_idx]
    r2 = radial_distance[idx_clipped]
    ee1 = encircled_energy[left_idx]
    ee2 = encircled_energy[idx_clipped]
    delta_r = r2 - r1
    safe_interior = interior & (delta_r != 0)

    result[safe_interior] = ee1[safe_interior] + (
        (query_distance[safe_interior] - r1[safe_interior]) *
        (ee2[safe_interior] - ee1[safe_interior]) / delta_r[safe_interior]
    )
    result[query_distance <= radial_distance[0]] = encircled_energy[0]
    result[query_distance >= radial_distance[-1]] = encircled_energy[-1]

    if scalar_input:
        return result[0]
    return result
