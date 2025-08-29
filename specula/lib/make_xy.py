###############################################################################
#  who           when        what
#  --------      ---------   ------------
#  A. Riccardi   April 1995  Written
#  A. Riccardi   2014-03-01  extended to odd Sampling. Rewritten to simplify
#                            the code. Ratio can be any positive number,
#                            no longer constrained to be larger than Sampling.
#  A. Puglisi    2019-09-08  ported to Python, added dtype parameter
###############################################################################


def make_xy(sampling, ratio, xp, dtype=None, polar=False, vector=False,
            zero_sampled=False, quarter=False, fft=False):
    '''
    Generates zero-centered domains in a cartesian plane.

    Generates a zero-centered domain on a cartesian plane or axis,
    using cartesian or polar coordinates, tipically for pupil sampling
    and FFT usage.

    Calling sequence::

       x,y = make_xy(sampling, ratio)
       r, theta = make_xy(sampling, ratio, polar=True)

    Parameters
    ----------
    sampling: int
              Number of sampling points per dimension of the domain, greater
              or equal to 2.
    ratio: float
           Extension of sampled domain: -Ratio <= x [,y] <= +Ratio

    Other Parameters
    ----------------
    polar: bool, optional
           If True, return domain sampling in polar coordinates. Default
           value is False (use cartesian coordinates)
    dtype: xp.dtype, optional
           If set, the result will have this dtype. Otherwise, it will be
           inferred by the dtype of *sampling* and *ratio*
    vector: bool, optional
            If True, 1-dimensional domain is sampled instead of 2d.
            If *polar* is True, the value of *vector* is ignored.
    zero_sampled: bool, optional
                  If True, origin of the domain is sampled. This
                  flag is useful to force the zero to be sampled
                  when sampling is even. When sampling is odd, the
                  zero is always sampled
    quarter: bool, optional
             If True, only 1st quadrant is returned with (X>=0 AND Y>=0).
             The array returned has:
             * if Sampling is even: Sampling/2 X Sampling/2 elements
             * if Sampling is odd:  (Sampling+1)/2 X (Sampling+1)/2 elements
    fft: bool, optional
         If True, order the output values for FFT purposes. For example:

         * Sampling=4, Ratio=1 vector -3/4,-1/4,+1/4,+3/4:
             returned as +1/4,+3/4,-3/4,-1/4.
         * Sampling=4, Ratio=1, zero_sampled=True, vector -1,-1/2,0,+1/2:
             returned as 0,1/2,-1,-1/2
         * Sampling=5, Ratio=1 vector -4/5,-2/5,0,+2/5,+4/5:
             returned as 0,+2/5,+4/5,-4/5,-2/5 

    Returns
    -------
      (X, Y): tuple of numpy arrays
            with no special options, a 2-elements tuple with the X and Y
            values of sampled points
      (R, Angle): tuple of numpy arrays
            if *polar* is True, a 2-elements tuple with radial and
            angular values (in radians) of sampled points
      X: numpy array
            if *vector* is True, numpy array with X values of sampled points.

    Raises
    ------
    ValueError
        If the sampling parameter is lower than 2.

    Notes
    -----
    HOW THE DOMAIN IS SAMPLED

    The concept is the following: considering an array of
    `sampling` x `sampling pixels, the bottom-left corner of the bottom-left
    pixel has coordinates (-`ratio`,-`ratio`) and the top-right cornet of the
    to-left pixel has coordinates (+`ratio`,+`ratio`). The procedure
    returns the coordinates of the centers of the pixels. When `sampling` is
    even and `zero_sampled` is True, the coordinates of the bottom-left corners
    of the pixels are returned.

    * `sampling` is even and `zero_sampled` is False: the edge of the domain
      is not sampled and the sampling is symmetrical respect to origin::

               Ex: Sampling=4, Ratio=1.
                   -1   -0.5    0    0.5    1    Domain (Ex. X axis)
                    |     |     |     |     |
                       *     *     *     *       Sampling points
                     -0.75 -0.25  0.25  0.75     Returned vector


    * `sampling` is even and `zero_sampled` is True: the lower edge is sampled
      and the sampling is not symmetrical respect to the origin::

              Ex: Sampling=4, Ratio=1.
                   -1   -0.5    0    0.5    1    Domain (Ex. X axis)
                    |     |     |     |     |
                    *     *     *     *          Sampling points
                   -1   -0.5    0    0.5         Returned vector

    * `sampling`  is odd (`zero_sampled`  is ignored): the zero is always
      sampled::

               Ex: Sampling=5, Ratio=1.
                   -1   -3/5  -1/5   1/5   3/5    1    Domain (Ex. X axis)
                    |     |     |     |     |     |
                       *     *     *     *     *       Sampling points
                     -4/5  -2/5    0    2/5   4/5      Returned vector

    * If `fft` is True, output values are ordered for FFT purposes::

           Ex: 2-dimensional domain: N = Sampling
           X or Y(0:N/2-1, 0:N/2-1)   1st quadrant (including origin
                                      if ZERO_SAMPLEDis set)
           X or Y(N/2:N, 0:N/2-1)     2nd quadrant
           X or Y(N/2:N, N/2:N)       3rd quadrant
           X or Y(0:N/2-1, N/2:N)     2nd quadrant

    Examples
    --------
    Compute a tilt plane on a round pupil

    >>> x, y = make_xy(256, 1.0)
    >>> pupil = x*0
    >>> pupil[(x*x + y*y)<=1] =1
    >>> plt.imshow(pupil)

    '''

    if sampling <= 1:
        raise ValueError('make_xy: sampling must be larger than 1')
    even_sampling = ((sampling % 2) == 0)

    if quarter:
        if even_sampling:
            size = sampling // 2
            if zero_sampled:
                x0 = 0.0
            else:
                x0 = -0.5
        else:
            size = (sampling + 1) // 2
            x0 = 0.0
    else:
        size = sampling
        x0 = (sampling - 1) / 2.0
        if even_sampling and zero_sampled:
            x0 += 0.5

    x = (xp.arange(size, dtype=dtype) - x0) / (sampling / 2.0) * ratio

    if fft and not quarter:
        if even_sampling:
            x = xp.roll(x, -sampling // 2)
        else:
            x = xp.roll(x, -(sampling - 1) // 2)

    if vector:
        return x

    x = xp.tile(x, (size, 1))
    y = xp.transpose(x).copy()
    if polar:
        x, y = _xy_to_polar(x, y, xp)

    return x, y


def _xy_to_polar(x, y, xp):

    r = xp.sqrt(x * x + y * y)
    y = xp.arctan2(y, x)
    return r, y

# ___oOo___
