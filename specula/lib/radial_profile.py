import numpy as np

def computeRadialProfile(image, centerInPxY=None, centerInPxX=None, xp=np, dtype=np.float64):
    if centerInPxX is None:
        centerInPxX = image.shape[1]/2
    if centerInPxY is None:
        centerInPxY = image.shape[0]/2
    yCoord, xCoord= xp.indices(image.shape)
    yCoord= (yCoord - centerInPxY)
    xCoord= (xCoord - centerInPxX)
    rCoord=xp.sqrt(xCoord**2 + yCoord**2)
    indexR= xp.argsort(rCoord.flat)
    radialDistancesSorted= rCoord.flat[indexR]
    imageValuesSortedByRadialDistance= image.flat[indexR]
    integerPartOfRadialDistances= radialDistancesSorted.astype(dtype)
    deltaRadialDistance= integerPartOfRadialDistances[1:] - \
        integerPartOfRadialDistances[:-1]
    radialDistanceChanges= xp.where(deltaRadialDistance)[0]
    nPxInBinZero= radialDistanceChanges[0]+ 1
    nPxInRadialBin= radialDistanceChanges[1:] - \
        radialDistanceChanges[:-1]
    imageRadialCumSum= xp.cumsum(imageValuesSortedByRadialDistance,
                                 dtype=dtype)
    imageSumInBinZero= imageRadialCumSum[radialDistanceChanges[0]]
    imageSumInRadialBin= \
        imageRadialCumSum[radialDistanceChanges[1:]] - \
        imageRadialCumSum[radialDistanceChanges[:-1]]
    profileInZero= imageSumInBinZero / nPxInBinZero
    profileFromOne= imageSumInRadialBin / nPxInRadialBin
    profile= xp.hstack([profileInZero, profileFromOne])

    distanceRadialCumSum= xp.cumsum(radialDistancesSorted)
    distanceSumInBinZero= distanceRadialCumSum[radialDistanceChanges[0]]
    distanceSumInRadialBin= \
        distanceRadialCumSum[radialDistanceChanges[1:]] - \
        distanceRadialCumSum[radialDistanceChanges[:-1]]
    distanceInZero= distanceSumInBinZero / nPxInBinZero
    distanceFromOne= distanceSumInRadialBin / nPxInRadialBin
    radialDistance= xp.hstack([distanceInZero, distanceFromOne])
    return profile, radialDistance