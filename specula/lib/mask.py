'''
Copied from arte.types.mask
Commenting out unused imports
'''

from specula import xp
from scipy import optimize
import warnings
import matplotlib.pyplot as plt


class BaseMask():
    '''
    TODO this is code copied from Arte.
    Should become a SPECULA DataObject if it is used.
    '''

    def __init__(self, mask_array):
        self._shape = mask_array.shape
        self._mask = mask_array
    
    def mask(self):
        '''
        Boolean mask of the mask

        Returns
        -------
        mask: boolean `~numpy.array`
            mask of the pupil. Array is True outside the pupil,
            and False inside the pupil
        '''
        return self._mask
    
    def as_masked_array(self, xp=xp):
        return xp.ma.array(xp.ones(self._shape, dtype=self.dtype),
                           mask=self.mask())
        
    def shape(self):
        '''
        Array shape

        Returns
        -------
        shape: list (2,)
            shape of the mask array
        '''
        return self._shape
    
    @staticmethod
    def from_masked_array(numpy_masked_array):
        return BaseMask(numpy_masked_array)
    
    def __eq__(self, other):
        if isinstance(other, BaseMask):
            return xp.array_equal(self.mask(), other.mask())
        return False

    def __hash__(self):
        return hash(self.mask().tobytes())

    # TODO: funzione per verificare che tutti i punti di questa maschera stanno
    # dentro una maschera passata


class CircularMask(BaseMask):
    '''
    TODO this is code copied from Arte.
    Should become a SPECULA DataObject if it is used.
    
    Represent a circular mask

    A `~numpy.array` representing a circular pupil. Frame shape, pupil radius
    and center can be specified.

    Use `mask` method to access the mask as boolean mask (e.g. to be used
    in a `~numpy.ma.masked_array` object) with False values where the frame is 
    not masked (i.e. within the pupil) and True values outside

    Use `asTransmissionValue` method to acces the mask as transmission mask, i.e. 
    1 within the pupil and 0 outside. Fractional transmission for edge pixels
    is not implemented

    If a `~numpy.ma.masked_array` having a circular mask is available, the static
    method `fromMaskedArray` can be used to create a `CircularMask` object having
    the same shape of the masked array and the same pupil center and radius


    Parameters
    ----------
        frameShape: tuple (2,)
            shape of the returned array

        maskRadius: real
            pupil radius in pixel

        maskCenter: list (2,) or `~numpy.array`
            Y-X coordinates of the pupil center in pixel
    '''

    FITTING_METHOD_IMAGE_MOMENTS = "ImageMoments"
    FITTING_METHOD_RANSAC = "RANSAC"
    FITTING_METHOD_CENTER_OF_GRAVITY = "COG"
    FITTING_METHOD_CORRELATION = "correlation"

    def __init__(self,
                 frameShape,
                 maskRadius=None,
                 maskCenter=None,
                 xp=xp):
        self._shape = frameShape
        self._maskRadius = maskRadius
        self._maskCenter = maskCenter
        self._mask = None
        self._computeMask(xp=xp)

    def __repr__(self):
        return "shape %s, radius %f, center %s" % (
            self._shape, self._maskRadius, self._maskCenter)

    def _computeMask(self,xp=xp):
        if self._maskRadius is None:
            self._maskRadius = min(self._shape) / 2.
        if self._maskCenter is None:
            self._maskCenter = 0.5 * xp.array([self._shape[0],
                                               self._shape[1]])

        r = self._maskRadius
        cc = self._maskCenter
        y, x = xp.mgrid[0.5: self._shape[0] + 0.5:1,
                        0.5: self._shape[1] + 0.5:1]
        self._mask = xp.where(
            ((x - cc[1]) ** 2 + (y - cc[0]) ** 2) <= r ** 2, False, True)

    def asTransmissionValue(self):
        '''
        Mask as a transmission mask: 1 for non-masked elements,
        0 for masked elements.

        Returns
        -------
        transmission_value: ndarray[int]
            transmission mask as a numpy array with dtype int
        '''
        return xp.logical_not(self._mask).astype(int)
    
    #def as_masked_array(self):
    #    return xp.ma.array(xp.array(self.asTransmissionValue(), dtype=self.dtype,
    #                       mask=self.mask())

    def radius(self):
        '''
        Radius of the mask 

        Returns
        -------
        radius: real
            mask radius in pixel

        '''
        return self._maskRadius

    def center(self):
        '''
        Y, X coordinates of the mask center

        Returns
        -------
        center: `~numpy.array` of shape (2,)
            Y, X coordinate of the mask center in the array reference system

        '''
        return self._maskCenter

    @staticmethod
    def fromMaskedArray(maskedArray, method='ImageMoments', **keywords):
        '''
        Creates a `CircularMask` roughly corresponding to the mask of a masked
        array
        
        Returns a `CircularMask` object having radius and center guessed from
        the passed mask using `ImageMoments` centroid and semiAxes as default.
        
        Important note: the created `CircularMask` object using "ImageMoments" 
        is not guaranteed to have the same set of valid points of the 
        passed mask, but it is included in the passed mask, i.e. all the 
        valid points of the created mask are also valid points of the passed 
        masked array. For the other methods this last property is not guaranteed.  

        Other valid circle estimation methods are:
        - "RANSAC" : circle fitting using RANSAC algorithm on Canny edge detection
        - "COG" : circle fitting using Center of gravity of the mask
        - "correlation" : circle fitting using correlation of the mask with a disk template

        
        Parameters
        ----------
        maskedArray: `~numpy.ma.MaskedArray` 
            a masked array with a circular mask
        method: string
            method for circle estimation. Default is "ImageMoments"

        Returns
        -------
        circular_mask: `CircularMask`
            a circular mask included in the mask of `maskedArray`

        '''
        raise NotImplementedError('CircularMask.fromMaskedArray is not implemented')
    
        # TODO declarations intended to remove warnings
        class ImageMoments:
            pass
        feature = object()
        measure = object()
        draw = object()

        assert isinstance(maskedArray, xp.ma.masked_array)
        shape = maskedArray.shape
        if method == CircularMask.FITTING_METHOD_IMAGE_MOMENTS:
            again = 0.995
            while again:
                im = ImageMoments(maskedArray.mask.astype(int) * -1 + 1)
                centerYX = xp.roll(im.centroid(), 1)
                radius = again * xp.min(im.semiAxes())
                circularMask = CircularMask(shape, radius, centerYX)
                if xp.in1d(circularMask.in_mask_indices(),
                           xp.argwhere(maskedArray.mask.flatten() == False)).all():
                    again = False
                if radius < 1:
                    raise Exception("Couldn't estimate a CircularMask")
                else:
                    again *= 0.9
        elif method == CircularMask.FITTING_METHOD_RANSAC:
            img = xp.asarray(maskedArray.mask.astype(float) * -1 + 1, dtype=float)
            img[img > 0] = 128
            edge = img.copy()
            edge = feature.canny(img, keywords.pop('sigmaCanny', 2))

            coords = xp.column_stack(xp.nonzero(edge))

            model, inliers = measure.ransac(
                coords, measure.CircleModel,
                min_samples = keywords.pop('min_samples', 20), 
                residual_threshold = keywords.pop('residual_threshold',0.001),
                max_trials = keywords.pop('max_trials',10000))
            cx, cy, r = model.params

            if  keywords.pop('display', False) is True:
                print(r"Cx-Cy {:.2f}-{:.2f}, R {:.2f}".format(cx, cy, r))
                rr, cc = draw.disk((model.params[0], model.params[1]),
                                model.params[2],
                                shape=img.shape)
                img[rr, cc] += 512
                CircularMask._dispnobl(img)

            cy+=0.5
            cx+=0.5
            circularMask = CircularMask(img.shape, r, [cx,cy])

        elif method == CircularMask.FITTING_METHOD_CENTER_OF_GRAVITY:
            img = xp.asarray(maskedArray.mask.astype(int) * -1 + 1, dtype=int)
            regions = measure.regionprops(img)
            bubble = regions[0]
            y0, x0 = bubble.centroid
            y0+=0.5
            x0+=0.5
            r = bubble.major_axis_length / 2.
            circularMask = CircularMask(img.shape, r, [y0,x0])
            
        elif method == CircularMask.FITTING_METHOD_CORRELATION:
            
            img = xp.asarray(maskedArray.mask.astype(int) * -1 + 1, dtype=int)
            regions = measure.regionprops(img)
            bubble = regions[0]

            x0, y0 = bubble.centroid
            r = bubble.major_axis_length / 2.

            initial_guess = (x0, y0, r)
            display = keywords.pop('display', False)

            if display:
                fign=plt.figure()
                
            def _cost_disk(params):
                x0, y0, r = params
                coords = draw.disk((x0, y0), r, shape=img.shape)
                template = xp.zeros_like(img)
                template[coords] = 1
                if display:
                    CircularMask._dispnobl(template + img, fign)
                return -xp.sum((template > 0) & (img > 0))

            res = optimize.minimize(_cost_disk, initial_guess,
                                    method='Nelder-Mead', **keywords)
            x0, y0, r = res.x
            y0+=0.5
            x0+=0.5
            if res.success is True:
                circularMask = CircularMask(img.shape, r, [x0,y0])

            else:
                raise Exception("Fit circle didn't converge %s" % res)
        
        else:
            raise ValueError("Unknown method %s" % method)

        if not xp.in1d(circularMask.in_mask_indices(),
                       xp.argwhere(maskedArray.mask.flatten() == False)).all():
            warnings.warn(
                "The generated CircularMask is not completely included in the passed masked array")
        return circularMask

    def regionOfInterest(self):
        raise NotImplementedError('CircularMask.regionOfInterest is not implemented')

    def in_mask_indices(self):
        return self.asTransmissionValue().flatten().nonzero()[0]

    def _dispnobl(img, fign=None, **kwargs):
        if fign is not None:
            plt.figure(fign.number)
        plt.clf()
        plt.imshow(img, aspect='auto', **kwargs)
        plt.colorbar()
        plt.draw()
        plt.ion()
        plt.show()
        plt.pause(0.001)
        plt.ioff()


class AnnularMask(CircularMask):
    '''
    TODO this is code copied from Arte.
    Should become a SPECULA DataObject if it is used.

    Inheritance of CircularMask class to provide an annular mask

    Added inRadius parameter, radius of central obstruction.
    Default inRadius values is 0 with AnnularMask converging to CircularMask
    '''

    def __init__(self,
                 frameShape,
                 maskRadius=None,
                 maskCenter=None,
                 inRadius=0,
                 xp=xp):
        self._inRadius = inRadius
        super().__init__(frameShape, maskRadius, maskCenter, xp=xp)

    def __repr__(self):
        return "shape %s, radius %f, center %s, inradius %f" % (
            self._shape, self._maskRadius, self._maskCenter, self._inRadius)

    def inRadius(self):
        return self._inRadius

    def _computeMask(self, xp=xp):

        if self._maskRadius is None:
            self._maskRadius = min(self._shape) / 2.
        if self._maskCenter is None:
            self._maskCenter = 0.5 * xp.array([self._shape[0],
                                               self._shape[1]], dtype=self.dtype)

        r = self._maskRadius
        cc = self._maskCenter
        y, x = xp.mgrid[0.5: self._shape[0] + 0.5:1,
                        0.5: self._shape[1] + 0.5:1]

        tmp = ((x - cc[1]) ** 2 + (y - cc[0]) ** 2) <= r ** 2
        if self._inRadius == 0:
            self._mask = xp.where(tmp, False, True)
        else:
            cc = CircularMask(self._shape, self._inRadius, self._maskCenter)
            tmp[cc.asTransmissionValue() > 0] = False
            self._mask = xp.where(tmp, False, True)

    @staticmethod
    def fromMaskedArray(maskedArray):
        raise Exception("Not implemented yet. ")
