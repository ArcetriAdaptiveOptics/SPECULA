import numpy as np
from specula import cp

class Interp2D():
    
    if cp:
        interp2_kernel = r'''
            extern "C" __global__
            void interp2_kernel_TYPE(TYPE *g_in, TYPE *g_out, int out_dx, int out_dy, int in_dx, TYPE *xx, TYPE *yy) {

                int y = blockIdx.y * blockDim.y + threadIdx.y;
                int x = blockIdx.x * blockDim.x + threadIdx.x;

                if ((y<out_dy) && (x<out_dx)) {
                    TYPE xcoord = xx[y*out_dx+x];
                    TYPE ycoord = yy[y*out_dx+x];
                    int xin = floor(xcoord);
                    int yin = floor(ycoord);
                    int xin2 = xin+1;
                    int yin2 = yin+1;

                    TYPE xdist = xcoord-xin;
                    TYPE ydist = ycoord-yin;

                    int idx_a = yin*in_dx + xin;
                    int idx_b = yin*in_dx + xin2;
                    int idx_c = yin2*in_dx + xin;
                    int idx_d = yin2*in_dx + xin2;

                    TYPE value = g_in[idx_a]*(1-xdist)*(1-ydist) +
                                g_in[idx_b]*xdist*(1-ydist) +
                                g_in[idx_c]*ydist*(1-xdist) +
                                g_in[idx_d]*xdist*ydist;

                    g_out[y*out_dx + x] = value;
                    }
                }
            '''
        interp2_kernel_float = cp.RawKernel(interp2_kernel.replace('TYPE', 'float'), name='interp2_kernel_float')
        interp2_kernel_double = cp.RawKernel(interp2_kernel.replace('TYPE', 'double'), name='interp2_kernel_double')

    def __init__(self, input_shape, output_shape, rotInDeg=0, rowShiftInPixels=0, colShiftInPixels=0, yy=None, xx=None, dtype=np.float32, xp=np):
        '''
        Setup a resampling matrix from <input_shape> to <output_shape>,
        with optional rotation and shift.
        If xx and yy are given, they have to be the same shape as <output_shape>
        and they will be used as sampling points over <input_shape>
        '''
        self.xp = xp
        self.dtype = dtype
        self.input_shape = input_shape
        self.output_shape = output_shape

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
            xxr = (xx-xc)*cos_ - (yy-yc)*sin_ + xc
            yyr = (xx-xc)*sin_ + (yy-yc)*cos_ + yc
            xx = xxr + xc
            yy = yyr + yc
            
        if rowShiftInPixels != 0 or colShiftInPixels != 0:
            yy += rowShiftInPixels
            xx += colShiftInPixels

        yy[np.where(yy < 0)] = 0
        xx[np.where(xx < 0)] = 0
        yy[np.where(yy > input_shape[0]-1)] = input_shape[0]-1
        xx[np.where(xx > input_shape[1]-1)] = input_shape[1]-1

        self.yy = self.xp.array(yy, dtype=dtype).ravel()
        self.xx = self.xp.array(xx, dtype=dtype).ravel()

    def interpolate(self, value, out=None):
        if value.shape != self.input_shape:
            raise ValueError(f'Array to be interpolated must have shape {self.input_shape} instead of {value.shape}')
        
        if out is None:
            out = self.xp.empty(shape=self.output_shape, dtype=self.dtype)
        
        if self.xp == cp:
            block = (16, 16)
            numBlocks2d = int(self.output_shape[0] // block[0])
            if self.output_shape[0] % block[0]:
                numBlocks2d += 1
            grid = (numBlocks2d, numBlocks2d)
            
            if self.dtype == cp.float32:
                self.interp2_kernel_float(grid, block, (value, out, self.output_shape[1],  self.output_shape[0],  self.input_shape[1], self.xx, self.yy))
            elif self.dtype == cp.float64:
                self.interp2_kernel_double(grid, block, (value, out, self.output_shape[1],  self.output_shape[0],  self.input_shape[1], self.xx, self.yy))
            else:
                raise ValueError('Unsupported dtype {self.dtype}')
            return out

        else:
            from scipy.interpolate import RegularGridInterpolator
            points = (self.xp.arange( self.input_shape[0], dtype=self.dtype), self.xp.arange( self.input_shape[1], dtype=self.dtype))
            interp = RegularGridInterpolator(points,value, method='linear')
            out[:] = interp((self.yy, self.xx)).reshape(self.output_shape).astype(self.dtype)
            return out
