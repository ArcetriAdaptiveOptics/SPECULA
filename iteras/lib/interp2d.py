import numpy as np
from functools import cache

import iteras
from iteras import to_xp
from scipy.interpolate import RegularGridInterpolator

class Interp2D():

    # CUDA sources of the kernels, created at the first GPU call (see _interp2_kernel())

    # Definition of bilinear interpolation device functions used by both kernels.
    # TYPE (float, double) and MODE (the outside rule) are replaced in the source:
    # one compiled kernel per type and rule, so that each kernel contains only the
    # code of its rule. Outside the array, the input is extended by a virtual
    # 1-pixel ring (see interpolate()): fetch() reads an element or a ring value
    # with the rule MODE = 0 (edge value), 1 (zero) or 2 (linear extrapolation
    # 2 a[0] - a[1] along each axis, separable at the corners).
    bilinear_interp_device = r'''
        // Products and sums with explicit rounding: the compiler cannot fuse them
        // into FMAs, so bilinear_interp() gives the same bits in every kernel that
        // includes it (e.g. the geometric propagation of AtmoPropagation)
        __device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
        __device__ __forceinline__ double mul_rn(double a, double b) { return __dmul_rn(a, b); }
        __device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
        __device__ __forceinline__ double add_rn(double a, double b) { return __dadd_rn(a, b); }

        __device__ TYPE fetch(TYPE *g_in, int in_dx, int in_dy, int x, int y) {
            bool out_x = (x < 0) || (x >= in_dx);
            bool out_y = (y < 0) || (y >= in_dy);
            if (MODE == 1 && (out_x || out_y)) return 0;
            int x0 = min(max(x, 0), in_dx - 1);
            int y0 = min(max(y, 0), in_dy - 1);
            if (MODE != 2 || !(out_x || out_y)) return g_in[y0 * in_dx + x0];
            // Linear: P = 2 a[edge] - a[next to the edge], along each axis out of the array
            int x1 = out_x ? (x < 0 ? 1 : in_dx - 2) : x0;
            int y1 = out_y ? (y < 0 ? 1 : in_dy - 2) : y0;
            TYPE cx0 = out_x ? 2 : 1;
            TYPE cx1 = out_x ? -1 : 0;
            TYPE cy0 = out_y ? 2 : 1;
            TYPE cy1 = out_y ? -1 : 0;
            return cy0 * (cx0 * g_in[y0 * in_dx + x0] + cx1 * g_in[y0 * in_dx + x1]) +
                   cy1 * (cx0 * g_in[y1 * in_dx + x0] + cx1 * g_in[y1 * in_dx + x1]);
        }

        // Coordinates already clamped to [0, N-1] (MODE 0) or [-1, N] (MODE 1, 2)
        __device__ TYPE bilinear_interp(TYPE *g_in, int in_dx, int in_dy, TYPE xcoord, TYPE ycoord) {
            // At the last row/column (coordinate N-1, or N on the ring) keep the
            // first neighbour on it, with weight 1 (or 0 for the second one)
            int xin = min((int)floor(xcoord), in_dx - 1);
            int yin = min((int)floor(ycoord), in_dy - 1);

            TYPE xdist = xcoord - xin;
            TYPE ydist = ycoord - yin;

            TYPE a, b, c, d;
            if (xin >= 0 && yin >= 0 && xin + 1 < in_dx && yin + 1 < in_dy) {
                // The 4 neighbours inside the array (almost all the pixels): direct reads
                int idx = yin * in_dx + xin;
                a = g_in[idx];
                b = g_in[idx + 1];
                c = g_in[idx + in_dx];
                d = g_in[idx + in_dx + 1];
            } else {
                a = fetch(g_in, in_dx, in_dy, xin, yin);
                b = fetch(g_in, in_dx, in_dy, xin + 1, yin);
                c = fetch(g_in, in_dx, in_dy, xin, yin + 1);
                d = fetch(g_in, in_dx, in_dy, xin + 1, yin + 1);
            }
            TYPE wx = 1 - xdist;
            TYPE wy = 1 - ydist;
            // a * wx * wy + b * xdist * wy + c * ydist * wx + d * xdist * ydist
            return add_rn(add_rn(add_rn(mul_rn(mul_rn(a, wx), wy),
                                        mul_rn(mul_rn(b, xdist), wy)),
                                 mul_rn(mul_rn(c, ydist), wx)),
                          mul_rn(mul_rn(d, xdist), ydist));
        }

        __device__ TYPE clamp_coord(TYPE c, int n) {
            TYPE lo = (MODE == 0) ? 0 : -1;
            TYPE hi = (MODE == 0) ? n - 1 : n;
            return min(max(c, lo), hi);
        }
        '''

    interp2_kernel = bilinear_interp_device + r'''
        extern "C" __global__
        void interp2_kernel_TYPE_MODE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, int in_dy, TYPE *xx, TYPE *yy) {
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int x = blockIdx.x * blockDim.x + threadIdx.x;

            if ((y < out_dy) && (x < out_dx)) {
                TYPE xcoord = clamp_coord(xx[y * out_dx + x], in_dx);
                TYPE ycoord = clamp_coord(yy[y * out_dx + x], in_dy);
                g_out[y * out_dx + x] = bilinear_interp(g_in, in_dx, in_dy, xcoord, ycoord);
            }
        }
        '''

    interp2_kernel_onthefly = bilinear_interp_device + r'''
        extern "C" __global__
        void interp2_kernel_onthefly_TYPE_MODE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, int in_dy,
                                        TYPE scale_x, TYPE scale_y, TYPE offset_x, TYPE offset_y,
                                        TYPE shift_x, TYPE shift_y,
                                        TYPE cos_angle, TYPE sin_angle, TYPE center_x, TYPE center_y, 
                                        TYPE magnification) {
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            int x = blockIdx.x * blockDim.x + threadIdx.x;

            if ((y < out_dy) && (x < out_dx)) {
                // Compute coordinates on-the-fly
                TYPE xcoord = x * scale_x + offset_x;
                TYPE ycoord = y * scale_y + offset_y;
                
                // Center coordinates for rotation and magnification
                TYPE xx_centered = xcoord - center_x;
                TYPE yy_centered = ycoord - center_y;
                
                // Apply magnification
                if (magnification != 1.0) {
                    xx_centered /= magnification;
                    yy_centered /= magnification;
                }
                
                // Apply rotation if necessary
                if (cos_angle != 1.0 || sin_angle != 0.0) {
                    TYPE xcoord_rot = xx_centered * cos_angle - yy_centered * sin_angle;
                    TYPE ycoord_rot = xx_centered * sin_angle + yy_centered * cos_angle;
                    xx_centered = xcoord_rot;
                    yy_centered = ycoord_rot;
                }
                
                // Restore center
                xcoord = xx_centered + center_x;
                ycoord = yy_centered + center_y;
                
                // Apply shift
                xcoord += shift_x;
                ycoord += shift_y;
                
                // Clamp to the array (mode 0) or to the virtual ring around it
                xcoord = clamp_coord(xcoord, in_dx);
                ycoord = clamp_coord(ycoord, in_dy);

                // Call bilinear interpolation
                g_out[y * out_dx + x] = bilinear_interp(g_in, in_dx, in_dy, xcoord, ycoord);
            }
        }
        '''

    def __init__(self, input_shape, output_shape,
                 rotInDeg=0, rowShiftInPixels=0,
                 colShiftInPixels=0, magnification=1.0,
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
        magnification : float, optional
            Magnification factor to apply to the sampling grid (default: 1.0).
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
        to the input grid, with optional rotation and shift applied. Pixel j is
        centred at coordinate j, so output pixel j maps to input coordinate
        (j + 0.5) * N_in / N_out - 0.5 and the centre (N_out - 1) / 2 maps to
        (N_in - 1) / 2, which is also the centre of rotation and magnification.
        Coordinates outside [0, N_in - 1] take the edge value, or see the
        outside argument of interpolate().
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
            magnification == 1.0 and
            xx is None and yy is None):
            # If not, it will be skipped later
            self.do_interp = False
            self.shift_x = 0.0
            self.shift_y = 0.0
            self.rot_angle = 0.0
            self.magnification = 1.0
            return

        # Decide whether to use on-the-fly or precomputed coordinates
        # Use on-the-fly ONLY when:
        # 1. On GPU (xp is cp)
        # 2. No custom coordinates provided (xx, yy are None)
        use_onthefly = (self.xp is iteras.cp and xx is None and yy is None)

        if use_onthefly:
            self.use_precomputed = False
            scale_x = input_shape[1] / output_shape[1]
            scale_y = input_shape[0] / output_shape[0]
            self.scale_x = self.dtype(scale_x)
            self.scale_y = self.dtype(scale_y)
            self.offset_x = self.dtype(0.5 * scale_x - 0.5)
            self.offset_y = self.dtype(0.5 * scale_y - 0.5)
            self.xx = None
            self.yy = None
        else:
            self.use_precomputed = True
            if xx is None or yy is None:
                yy, xx = map(self.dtype, np.mgrid[0:output_shape[0], 0:output_shape[1]])
                # Pixel-centre mapping: pixel j is centred at j, so output pixel j
                # maps to input coordinate (j + 0.5) * N_in / N_out - 0.5.
                # It is not used in propagation, where xx and yy are set from the caller code
                yy = (yy + 0.5) * (input_shape[0] / output_shape[0]) - 0.5
                xx = (xx + 0.5) * (input_shape[1] / output_shape[1]) - 0.5
            else:
                if yy.shape != output_shape or xx.shape != output_shape:
                    raise ValueError(f'yy and xx must have shape {output_shape}')
                else:
                    yy = xp.array(yy, dtype=dtype)
                    xx = xp.array(xx, dtype=dtype)

            if rotInDeg != 0 or magnification != 1.0:
                yc = input_shape[0] / 2 - 0.5
                xc = input_shape[1] / 2 - 0.5

                xx_centered = (xx - xc) / magnification
                yy_centered = (yy - yc) / magnification

                if rotInDeg != 0:
                    cos_ = np.cos(rotInDeg * np.pi / 180.0)
                    sin_ = np.sin(rotInDeg * np.pi / 180.0)
                    xxr = xx_centered * cos_ - yy_centered * sin_
                    yyr = xx_centered * sin_ + yy_centered * cos_
                    xx_centered = xxr
                    yy_centered = yyr

                xx = xx_centered + xc
                yy = yy_centered + yc

            if rowShiftInPixels != 0 or colShiftInPixels != 0:
                yy += rowShiftInPixels
                xx += colShiftInPixels

            # Clamped to the virtual 1-pixel ring around the input (see interpolate());
            # with outside='edge' they are clamped again to [0, N-1]
            yy = np.clip(yy, -1, input_shape[0])
            xx = np.clip(xx, -1, input_shape[1])

            self.yy = to_xp(self.xp, yy, dtype=dtype).ravel()
            self.xx = to_xp(self.xp, xx, dtype=dtype).ravel()

            self.scale_x = None
            self.scale_y = None
            self.offset_x = None
            self.offset_y = None

        self.shift_x = self.dtype(colShiftInPixels)
        self.shift_y = self.dtype(rowShiftInPixels)
        self.rot_angle = rotInDeg * np.pi / 180.0
        self.magnification = self.dtype(magnification)
        self.cos_angle = self.dtype(np.cos(self.rot_angle))
        self.sin_angle = self.dtype(np.sin(self.rot_angle))
        self.center_x = self.dtype(input_shape[1] / 2 - 0.5)
        self.center_y = self.dtype(input_shape[0] / 2 - 0.5)

    # Integer codes of the outside rules in the CUDA kernels
    OUTSIDE_MODES = {'edge': 0, 'zero': 1, 'linear': 2}

    def interpolate(self, value, out=None, outside='edge'):
        """
        Interpolates the input array to the output grid defined by the interpolator.

        Parameters
        ----------
        value : array-like
            The input array to be interpolated. Must have shape `input_shape`.
        out : array-like, optional
            Optional output array to store the result. If not provided, a new array is created.
        outside : str, optional
            Value for the coordinates outside the input array [0, N - 1]:

            - 'edge' (default): the value of the nearest edge pixel, however far;
            - 'zero': the array is extended by a ring of zeros, so the value goes
              linearly from the edge pixel to 0 within one pixel and is 0 beyond.
              Suited to an amplitude: outside the array there is no light;
            - 'linear': the array is extended by a ring with the linear
              extrapolation 2 a[0] - a[1] (along each axis), so a linear field
              continues for one pixel beyond the edge, and is then held.
              Suited to a phase.

            With 'zero' and 'linear', the coordinates are clamped to [-1, N]
            instead of [0, N - 1].

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
        if outside not in self.OUTSIDE_MODES:
            raise ValueError(f"outside must be one of {list(self.OUTSIDE_MODES)}, got {outside!r}")
        if outside == 'linear' and min(self.input_shape) < 2:
            raise ValueError("outside='linear' needs at least 2 pixels along each axis")
        mode = self.OUTSIDE_MODES[outside]

        # Skip interpolation if not needed
        if not self.do_interp:
            if out is None:
                return value
            else:
                out[:] = value
                return out

        if out is None:
            out = self.xp.empty(shape=self.output_shape, dtype=self.dtype)

        if self.xp == iteras.cp: # pragma: no cover
            cp = iteras.cp
            if self.dtype == cp.float32:
                type_name = 'float'
            elif self.dtype == cp.float64:
                type_name = 'double'
            else:
                raise ValueError(f'Unsupported dtype {self.dtype}')
            block = (16, 16)
            # Calculate grid size for non-square arrays correctly
            grid_x = (self.output_shape[1] + block[0] - 1) // block[0]
            grid_y = (self.output_shape[0] + block[1] - 1) // block[1]
            grid = (grid_x, grid_y)

            if not self.use_precomputed:
                # Use on-the-fly coordinate calculation kernel
                kernel = _interp2_kernel('interp2_kernel_onthefly', type_name, mode)
                kernel(grid, block, (
                    value, out,
                    self.output_shape[1], self.output_shape[0],
                    self.input_shape[1], self.input_shape[0],
                    self.scale_x, self.scale_y,
                    self.offset_x, self.offset_y,
                    self.shift_x, self.shift_y,
                    self.cos_angle, self.sin_angle,
                    self.center_x, self.center_y,
                    self.magnification))
            else:
                kernel = _interp2_kernel('interp2_kernel', type_name, mode)
                kernel(grid, block,
                    (value, out, self.output_shape[1], self.output_shape[0],
                     self.input_shape[1], self.input_shape[0], self.xx, self.yy))

            return out

        else:
            if outside == 'edge':
                ny, nx = self.input_shape
                points = (np.arange(ny, dtype=self.dtype), np.arange(nx, dtype=self.dtype))
                coords = (np.clip(self.yy, 0, ny - 1), np.clip(self.xx, 0, nx - 1))
            else:
                # The virtual ring of the GPU kernels, as an explicit 1-pixel padding
                value = _pad_one_pixel(value, outside)
                ny, nx = value.shape
                points = (np.arange(-1, ny - 1, dtype=self.dtype),
                          np.arange(-1, nx - 1, dtype=self.dtype))
                coords = (self.yy, self.xx)
            interp = RegularGridInterpolator(points, value, method='linear')
            out[:] = interp(coords).reshape(self.output_shape)
            return out


def _pad_one_pixel(value, outside):
    '''
    The input extended by a 1-pixel ring: zeros ('zero'), or the linear
    extrapolation 2 a[0] - a[1] along each axis ('linear'; first along the rows,
    then along the columns, so the corners are extrapolated along both axes).
    '''
    if outside == 'zero':
        return np.pad(value, 1)
    rows = np.concatenate([2 * value[:1] - value[1:2], value, 2 * value[-1:] - value[-2:-1]], axis=0)
    return np.concatenate([2 * rows[:, :1] - rows[:, 1:2], rows,
                           2 * rows[:, -1:] - rows[:, -2:-1]], axis=1)


@cache
def _interp2_kernel(kernel_name, type_name, mode):  # pragma: no cover
    '''
    One interpolation kernel ('interp2_kernel' or 'interp2_kernel_onthefly') for
    a type ('float', 'double') and an outside rule (0, 1, 2; see Interp2D.OUTSIDE_MODES),
    compiled at its first GPU call, so that this module can be imported before init()
    '''
    source = getattr(Interp2D, kernel_name).replace('TYPE', type_name).replace('MODE', str(mode))
    return iteras.cp.RawKernel(source, name=f'{kernel_name}_{type_name}_{mode}')
