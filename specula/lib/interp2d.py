import numpy as np
from specula import cp, to_xp
from scipy.interpolate import RegularGridInterpolator

class Interp2D():

    if cp: # pragma: no cover
        # Definition of bilinear interpolation device function used by both kernels
        bilinear_interp_device = r'''
            __device__ TYPE bilinear_interp(TYPE *g_in, int in_dx, int in_dy, TYPE xcoord, TYPE ycoord) {
                int xin = floor(xcoord);
                int yin = floor(ycoord);
                int xin2 = xin + 1;
                int yin2 = yin + 1;

                TYPE xdist = xcoord - xin;
                TYPE ydist = ycoord - yin;

                int idx_a = yin * in_dx + xin;
                int idx_b = yin * in_dx + xin2;
                int idx_c = yin2 * in_dx + xin;
                int idx_d = yin2 * in_dx + xin2;

                TYPE value;
                if (yin2 < in_dy) {
                    value = g_in[idx_a] * (1 - xdist) * (1 - ydist) +
                            g_in[idx_b] * xdist * (1 - ydist) +
                            g_in[idx_c] * ydist * (1 - xdist) +
                            g_in[idx_d] * xdist * ydist;
                } else {
                    value = g_in[idx_a] * (1 - xdist) * (1 - ydist) +
                            g_in[idx_b] * xdist * (1 - ydist);
                }
                return value;
            }
            '''

        interp2_kernel = bilinear_interp_device + r'''
            extern "C" __global__
            void interp2_kernel_TYPE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, int in_dy, TYPE *xx, TYPE *yy) {
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int x = blockIdx.x * blockDim.x + threadIdx.x;

                if ((y < out_dy) && (x < out_dx)) {
                    TYPE xcoord = xx[y * out_dx + x];
                    TYPE ycoord = yy[y * out_dx + x];
                    g_out[y * out_dx + x] = bilinear_interp(g_in, in_dx, in_dy, xcoord, ycoord);
                }
            }
            '''
        interp2_kernel_float = \
            cp.RawKernel(interp2_kernel.replace('TYPE', 'float'), name='interp2_kernel_float')
        interp2_kernel_double = \
            cp.RawKernel(interp2_kernel.replace('TYPE', 'double'), name='interp2_kernel_double')

        interp2_kernel_onthefly = bilinear_interp_device + r'''
            extern "C" __global__
            void interp2_kernel_onthefly_TYPE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, int in_dy,
                                            TYPE scale_x, TYPE scale_y, TYPE shift_x, TYPE shift_y,
                                            TYPE cos_angle, TYPE sin_angle, TYPE center_x, TYPE center_y) {
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int x = blockIdx.x * blockDim.x + threadIdx.x;

                if ((y < out_dy) && (x < out_dx)) {
                    // Compute coordinates on-the-fly
                    TYPE xcoord = x * scale_x;
                    TYPE ycoord = y * scale_y;
                    
                    // Apply rotation if necessary
                    if (cos_angle != 1.0 || sin_angle != 0.0) {
                        TYPE xx_centered = xcoord - center_x;
                        TYPE yy_centered = ycoord - center_y;
                        TYPE xcoord_rot = xx_centered * cos_angle - yy_centered * sin_angle;
                        TYPE ycoord_rot = xx_centered * sin_angle + yy_centered * cos_angle;
                        xcoord = xcoord_rot + center_x;
                        ycoord = ycoord_rot + center_y;
                    }
                    
                    // Apply shift
                    xcoord += shift_x;
                    ycoord += shift_y;
                    
                    // Clamp to limits
                    if (xcoord < 0) xcoord = 0;
                    if (ycoord < 0) ycoord = 0;
                    if (xcoord > in_dx - 1) xcoord = in_dx - 1;
                    if (ycoord > in_dy - 1) ycoord = in_dy - 1;
                    
                    // Call bilinear interpolation
                    g_out[y * out_dx + x] = bilinear_interp(g_in, in_dx, in_dy, xcoord, ycoord);
                }
            }
            '''
        interp2_kernel_onthefly_float = \
            cp.RawKernel(interp2_kernel_onthefly.replace('TYPE', 'float'),
                         name='interp2_kernel_onthefly_float')
        interp2_kernel_onthefly_double = \
            cp.RawKernel(interp2_kernel_onthefly.replace('TYPE', 'double'),
                         name='interp2_kernel_onthefly_double')

    def __init__(self, input_shape, output_shape,
                 rotInDeg=0, rowShiftInPixels=0, colShiftInPixels=0,
                 yy=None, xx=None, dtype=np.float32, xp=np):
        '''
        Initialize an Interp2D object for 2D interpolation between arrays.

        Parameters
        ----------
        input_shape : tuple of int
            Shape (rows, cols) of the input array to be interpolated.
        output_shape : tuple of int
            Desired shape (rows, cols) of the output (interpolated) array.
        rotInDeg : float, optional
            Rotation angle in degrees to apply to the sampling grid (default: 0).
        rowShiftInPixels : float, optional
            Vertical shift (in pixels) to apply to the sampling grid (default: 0).
        colShiftInPixels : float, optional
            Horizontal shift (in pixels) to apply to the sampling grid (default: 0).
        yy : array-like, optional
            Precomputed y-coordinates for the output grid (same shape as output_shape).
        xx : array-like, optional
            Precomputed x-coordinates for the output grid (same shape as output_shape).
        dtype : data-type, optional
            Data type for interpolation (default: np.float32).
        xp : module, optional
            Array module to use (default: numpy).

        Notes
        -----
        If `xx` and `yy` are not provided, they are generated to map the output grid
        to the input grid, with optional rotation and shift applied.
        '''
        self.xp = xp
        self.dtype = dtype
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.do_interp = True

        # Check if interpolation is actually needed
        if (input_shape == output_shape and
            rotInDeg == 0 and
            rowShiftInPixels == 0 and
            colShiftInPixels == 0 and
            xx is None and yy is None):
            # If not, it will be skipped later
            self.do_interp = False
            return

        # Decide whether to use on-the-fly or precomputed coordinates
        # Use on-the-fly ONLY when:
        # 1. On GPU (xp is cp)
        # 2. No custom coordinates provided (xx, yy are None)
        use_onthefly = (self.xp is cp and xx is None and yy is None)

        if use_onthefly:
            self.use_precomputed = False
            self.scale_x = self.dtype((input_shape[1] - 1) / output_shape[1])
            self.scale_y = self.dtype((input_shape[0] - 1) / output_shape[0])
            self.xx = None
            self.yy = None
        else:
            self.use_precomputed = True
            if xx is None or yy is None:
                yy, xx = map(self.dtype, np.mgrid[0:output_shape[0], 0:output_shape[1]])
                # This -1 appears to be correct by comparing with IDL code
                # It is not used in propagation, where xx and yy are set from the caller code
                yy *= (input_shape[0]-1) / output_shape[0]
                xx *= (input_shape[1]-1) / output_shape[1]
            else:
                if yy.shape != output_shape or xx.shape != output_shape:
                    raise ValueError(f'yy and xx must have shape {output_shape}')
                else:
                    yy = xp.array(yy, dtype=dtype)
                    xx = xp.array(xx, dtype=dtype)

            if rotInDeg != 0:
                yc = input_shape[0] / 2 - 0.5
                xc = input_shape[1] / 2 - 0.5
                cos_ = np.cos(rotInDeg * 3.1415 / 180.0)
                sin_ = np.sin(rotInDeg * 3.1415 / 180.0)
                xxr = (xx-xc)*cos_ - (yy-yc)*sin_
                yyr = (xx-xc)*sin_ + (yy-yc)*cos_
                xx = xxr + xc
                yy = yyr + yc

            if rowShiftInPixels != 0 or colShiftInPixels != 0:
                yy += rowShiftInPixels
                xx += colShiftInPixels

            yy[np.where(yy < 0)] = 0
            xx[np.where(xx < 0)] = 0
            yy[np.where(yy > input_shape[0] - 1)] = input_shape[0] - 1
            xx[np.where(xx > input_shape[1] - 1)] = input_shape[1] - 1

            self.yy = to_xp(self.xp, yy, dtype=dtype).ravel()
            self.xx = to_xp(self.xp, xx, dtype=dtype).ravel()

            self.scale_x = None
            self.scale_y = None

        self.shift_x = self.dtype(colShiftInPixels)
        self.shift_y = self.dtype(rowShiftInPixels)
        self.rot_angle = rotInDeg * np.pi / 180.0
        self.cos_angle = self.dtype(np.cos(self.rot_angle))
        self.sin_angle = self.dtype(np.sin(self.rot_angle))
        self.center_x = self.dtype(input_shape[1] / 2 - 0.5)
        self.center_y = self.dtype(input_shape[0] / 2 - 0.5)

    def interpolate(self, value, out=None):
        """
        Interpolates the input array to the output grid defined by the interpolator.

        Parameters
        ----------
        value : array-like
            The input array to be interpolated. Must have shape `input_shape`.
        out : array-like, optional
            Optional output array to store the result. If not provided, a new array is created.

        Returns
        -------
        out : array-like
            The interpolated array with shape `output_shape`.

        Raises
        ------
        ValueError
            If the input array does not have the expected shape.

        Notes
        -----
        For CPU arrays, uses scipy's RegularGridInterpolator.
        For GPU arrays (cupy), uses a custom CUDA kernel.
        """
        if value.shape != self.input_shape:
            raise ValueError(f'Array to be interpolated must have shape'
                             f' {self.input_shape} instead of {value.shape}')

        # Skip interpolation if not needed
        if not self.do_interp:
            if out is None:
                return value
            else:
                out[:] = value
                return out

        if out is None:
            out = self.xp.empty(shape=self.output_shape, dtype=self.dtype)

        if self.xp == cp: # pragma: no cover
            block = (16, 16)
            # Calculate grid size for non-square arrays correctly
            grid_x = (self.output_shape[1] + block[0] - 1) // block[0]
            grid_y = (self.output_shape[0] + block[1] - 1) // block[1]
            grid = (grid_x, grid_y)

            if not self.use_precomputed:
                # Use on-the-fly coordinate calculation kernel
                if self.dtype == cp.float32:
                    self.interp2_kernel_onthefly_float(grid, block, (
                        value, out,
                        self.output_shape[1], self.output_shape[0],
                        self.input_shape[1], self.input_shape[0],
                        self.scale_x, self.scale_y,
                        self.shift_x, self.shift_y,
                        self.cos_angle, self.sin_angle,
                        self.center_x, self.center_y))
                elif self.dtype == cp.float64:
                    self.interp2_kernel_onthefly_double(grid, block, (
                        value, out,
                        self.output_shape[1], self.output_shape[0],
                        self.input_shape[1], self.input_shape[0],
                        self.scale_x, self.scale_y,
                        self.shift_x, self.shift_y,
                        self.cos_angle, self.sin_angle,
                        self.center_x, self.center_y))
                else:
                    raise ValueError(f'Unsupported dtype {self.dtype}')
            else:
                if self.dtype == cp.float32:
                    self.interp2_kernel_float(grid, block,
                        (value, out, self.output_shape[1], self.output_shape[0],
                         self.input_shape[1], self.input_shape[0], self.xx, self.yy))
                elif self.dtype == cp.float64:
                    self.interp2_kernel_double(grid, block,
                        (value, out, self.output_shape[1], self.output_shape[0],
                         self.input_shape[1], self.input_shape[0], self.xx, self.yy))
                else:
                    raise ValueError('Unsupported dtype {self.dtype}')

            return out

        else:
            points = (self.xp.arange( self.input_shape[0], dtype=self.dtype),
                      self.xp.arange( self.input_shape[1], dtype=self.dtype))
            interp = RegularGridInterpolator(points,value, method='linear')
            out[:] = interp((self.yy, self.xx)).reshape(self.output_shape)
            return out
