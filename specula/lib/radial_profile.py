import numpy as np


def computeRadialProfile(image, centerInPxY=None, centerInPxX=None,
                         xp=np, dtype=np.float64, return_counts=False):
    """Compute the azimuthally averaged radial profile of a 2D image.

    Parameters
    ----------
    image : ndarray
        Input 2D image.
    centerInPxY, centerInPxX : float, optional
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
    radialDistance : ndarray
        Mean distance of each radial bin in pixels.
    nPxInRadialBin : ndarray, optional
        Number of pixels in each radial bin, returned only if `return_counts` is True.
    """
    image = xp.asarray(image)
    if image.ndim != 2:
        raise ValueError('computeRadialProfile expects a 2D image')

    if centerInPxX is None:
        centerInPxX = image.shape[1] / 2
    if centerInPxY is None:
        centerInPxY = image.shape[0] / 2

    # Coordinates relative to the center, in pixels
    yCoord, xCoord = xp.indices(image.shape, dtype=dtype)
    yCoord = yCoord - dtype(centerInPxY)
    xCoord = xCoord - dtype(centerInPxX)
    rCoord = xp.sqrt(xCoord**2 + yCoord**2)

    radialBin = xp.floor(rCoord).astype(np.int32).ravel()
    imageFlat = image.ravel().astype(dtype, copy=False)
    rCoordFlat = rCoord.ravel().astype(dtype, copy=False)

    # Count number of occurrences of each value in radialBin and
    # sum image values and distances in each bin
    nPxInRadialBin = xp.bincount(radialBin)
    sumInRadialBin = xp.bincount(radialBin, weights=imageFlat)
    sumDistanceInRadialBin = xp.bincount(radialBin, weights=rCoordFlat)

    # Only keep bins with at least one pixel to avoid division by zero
    validBins = nPxInRadialBin > 0
    nPxInRadialBin = nPxInRadialBin[validBins]
    # Compute the mean profile in each bin
    profile = sumInRadialBin[validBins] / nPxInRadialBin
    # Compute the mean radial distance in each bin
    radialDistance = sumDistanceInRadialBin[validBins] / nPxInRadialBin

    if return_counts:
        return profile, radialDistance, nPxInRadialBin
    return profile, radialDistance


def computeFWHMFromProfile(profile, radialDistance=None, xp=np, dtype=np.float64):
    """Estimate the FWHM from a radial profile using linear interpolation."""
    profile = xp.asarray(profile, dtype=dtype)
    if profile.ndim != 1:
        raise ValueError('profile must be a 1D array')

    if radialDistance is None:
        radialDistance = xp.arange(profile.size, dtype=dtype)
    else:
        radialDistance = xp.asarray(radialDistance, dtype=dtype)

    if radialDistance.ndim != 1 or radialDistance.size != profile.size:
        raise ValueError('radialDistance must be a 1D array with the same size as profile')
    if profile.size == 0:
        return dtype(np.nan)

    peakValue = xp.max(profile)
    if float(peakValue) <= 0.0:
        return dtype(0.0)

    # Find the first bin where the profile drops below half the peak value
    halfMaximum = peakValue / dtype(2.0)
    # Find the pixel below and above the half maximum
    belowHalf = xp.where(profile <= halfMaximum)[0]
    belowHalf = belowHalf[belowHalf > 0]
    if belowHalf.size == 0:
        return dtype(np.nan)

    idx = int(belowHalf[0])
    r1 = radialDistance[idx - 1]
    r2 = radialDistance[idx]
    p1 = profile[idx - 1]
    p2 = profile[idx]

    if float(p2 - p1) == 0.0:
        halfRadius = r1
    else:
        # Linear interpolation to find the radius at half maximum
        halfRadius = r1 + (halfMaximum - p1) * (r2 - r1) / (p2 - p1)
    return dtype(2.0) * dtype(halfRadius)


def computeEncircledEnergy(profile, nPxInRadialBin=None, radialDistance=None,
                           xp=np, dtype=np.float64, normalize=True):
    """Compute the encircled-energy curve from a radial profile.

    If `nPxInRadialBin` is not available, the energy in each bin is approximated
    from the annulus area derived from `radialDistance` (or from equally spaced
    bins if `radialDistance` is not provided).
    """
    profile = xp.asarray(profile, dtype=dtype)
    if profile.ndim != 1:
        raise ValueError('profile must be a 1D array')

    if nPxInRadialBin is None:
        if radialDistance is None:
            radialDistance = xp.arange(profile.size, dtype=dtype)
        else:
            radialDistance = xp.asarray(radialDistance, dtype=dtype)
            if radialDistance.shape != profile.shape:
                raise ValueError('radialDistance must have the same shape as profile')

        if profile.size == 0:
            energyInRadialBin = profile
        elif profile.size == 1:
            # For a single bin, the annulus area is approximated as a circle with radius equal to
            # the bin's radial distance
            outerRadius = xp.maximum(radialDistance[0], dtype(0.5))
            annulusWeight = xp.asarray([outerRadius**2], dtype=dtype)
            energyInRadialBin = profile * annulusWeight
        else:
            # Compute the inner and outer radius of each annulus bin from the radial distance
            # midpoints, assuming the first bin starts at radius 0 and the last bin extends to
            # the next radial distance
            radialMidpoints = dtype(0.5) * (radialDistance[1:] + radialDistance[:-1])
            innerRadius = xp.empty_like(radialDistance)
            outerRadius = xp.empty_like(radialDistance)
            innerRadius[0] = dtype(0.0)
            innerRadius[1:] = radialMidpoints
            outerRadius[:-1] = radialMidpoints
            outerRadius[-1] = radialDistance[-1] + (radialDistance[-1] - innerRadius[-1])
            annulusWeight = xp.maximum(outerRadius**2 - innerRadius**2, 0)
            energyInRadialBin = profile * annulusWeight
    else:
        nPxInRadialBin = xp.asarray(nPxInRadialBin, dtype=dtype)
        if nPxInRadialBin.shape != profile.shape:
            raise ValueError('nPxInRadialBin must have the same shape as profile')
        # Compute the total energy in each radial bin by multiplying the mean profile value by
        # the number of pixels in that bin
        energyInRadialBin = profile * nPxInRadialBin

    encircledEnergy = xp.cumsum(energyInRadialBin, dtype=dtype)
    if normalize and encircledEnergy.size > 0:
        totalEnergy = encircledEnergy[-1]
        if float(totalEnergy) != 0.0:
            encircledEnergy = encircledEnergy / totalEnergy
    return encircledEnergy


def getEncircledEnergyAtDistance(encircledEnergy, radialDistance, distance,
                                 xp=np, dtype=np.float64):
    """Return the encircled energy at one or more requested radial distances."""
    encircledEnergy = xp.asarray(encircledEnergy, dtype=dtype)
    radialDistance = xp.asarray(radialDistance, dtype=dtype)
    queryDistance = xp.asarray(distance, dtype=dtype)
    scalarInput = queryDistance.ndim == 0
    queryDistance = xp.atleast_1d(queryDistance)

    if radialDistance.ndim != 1 or encircledEnergy.ndim != 1:
        raise ValueError('encircledEnergy and radialDistance must be 1D arrays')
    if radialDistance.size != encircledEnergy.size:
        raise ValueError('encircledEnergy and radialDistance must have the same size')
    if radialDistance.size == 0:
        result = xp.full(queryDistance.shape, xp.nan, dtype=dtype)
        return result[0] if scalarInput else result

    idx = xp.searchsorted(radialDistance, queryDistance, side='left')
    idx = xp.clip(idx, 0, radialDistance.size - 1)
    result = encircledEnergy[idx].astype(dtype, copy=True)

    interior = (idx > 0) & (idx < radialDistance.size)
    idxClipped = xp.clip(idx, 1, radialDistance.size - 1)
    leftIdx = idxClipped - 1
    r1 = radialDistance[leftIdx]
    r2 = radialDistance[idxClipped]
    ee1 = encircledEnergy[leftIdx]
    ee2 = encircledEnergy[idxClipped]
    deltaR = r2 - r1
    safeInterior = interior & (deltaR != 0)

    result[safeInterior] = ee1[safeInterior] + (
        (queryDistance[safeInterior] - r1[safeInterior]) *
        (ee2[safeInterior] - ee1[safeInterior]) / deltaR[safeInterior]
    )
    result[queryDistance <= radialDistance[0]] = encircledEnergy[0]
    result[queryDistance >= radialDistance[-1]] = encircledEnergy[-1]

    if scalarInput:
        return result[0]
    return result
