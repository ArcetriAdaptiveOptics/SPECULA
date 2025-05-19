# -*- coding: utf-8 -*-
#########################################################
# PySimul project.
#
# who       when        what
# --------  ----------  ---------------------------------
# apuglisi  2019-09-28  Created
#
#########################################################


from specula import cp
from specula.lib.rebin import rebin2d


def gcd(a,b):
    '''
    Returns the greatest common divisor of a and b.
    '''
    while b:
        a, b = b, a % b

    return a


def lcm(a,b):
    '''
    Returns the least common multiple of a and b.
    '''
    return (a*b) // gcd(a,b)


def toccd(a, newshape, set_total=None, xp=None):
    '''
    Clone of oaalib's toccd() function, using least common multiple
    to rebin an array similar to openvc's INTER_AREA interpolation.

    If a GPU is available, calculation is delegated to toccd_gpu()
    '''
    if a.shape == newshape:
        return a

    if xp == cp:
        return toccd_gpu(a, newshape, set_total=set_total)

    if len(a.shape) != 2:
        raise ValueError('Input array has shape %s, cannot continue' % str(a.shape))

    if len(newshape) != 2:
        raise ValueError('Output shape is %s, cannot continue' % str(newshape))

    if set_total is None:
        set_total = a.sum()

    mcmx = lcm(a.shape[0], newshape[0])
    mcmy = lcm(a.shape[1], newshape[1])

    temp = rebin2d(a, (mcmx, a.shape[1]), sample=True, xp=xp)
    temp = rebin2d(temp, (newshape[0], a.shape[1]), xp=xp)
    temp = rebin2d(temp, (newshape[0], mcmy), sample=True, xp=xp)
    rebinned = rebin2d(temp, newshape, xp=xp)

    return rebinned / rebinned.sum() * set_total


def toccd_gpu(a, newshape, set_total=None):
    '''
    toccd GPU code adapted from IDL PASSATA (gpu_simul.cu)
    - python code replicates the C function doCudaToCcdOptimized()
    - CUDA kernels are unchanged, except for removal of unused arguments

    Iterates directly on the resulting array (or an intermediate array in step 1)
    '''
    iny, inx = a.shape
    outy, outx = newshape

    mcmx = lcm(inx, outx)
    mcmy = lcm(iny, outy)

    dx_in = int(mcmx / inx)
    dy_in = int(mcmy / iny)
    dx_out = int(mcmx / outx)
    dy_out = int(mcmy / outy)
    f = 1.0 / (dx_out * dy_out)
    oneOverDxIn = 1.0 / dx_in

    block = (16, 16)
    numBlocks2d = int(outx // block[1])
    if outx % block[1]:
        numBlocks2d += 1
    grid = (numBlocks2d, numBlocks2d)

    numBlocks2d_tmp = int(inx // block[0])
    if inx % block[0]:
        numBlocks2d_tmp += 1
    grid_tmp = (numBlocks2d, numBlocks2d_tmp)  # Note second element is different

    tmp = cp.empty_like(a, shape=(iny, outx))  # TODO this is a reallocation and could give problems with streams
    out = cp.empty_like(a, shape=(outy, outx))

    if a.dtype == cp.float32:
        _rebin2D_step1_float(grid_tmp, block, (a, tmp, inx, iny, outx, outy, dx_out, cp.float32(oneOverDxIn)))
        _rebin2D_step2_float(grid, block, (tmp, out, outx, outy, dy_in, dy_out, cp.float32(f)))
    elif a.dtype == cp.float64:
        _rebin2D_step1_double(grid_tmp, block, (a, tmp, inx, iny, outx, outy, dx_out, cp.float64(oneOverDxIn)))
        _rebin2D_step2_double(grid, block, (tmp, out, outx, outy, dy_in, dy_out, cp.float64(f)))
    else:
        raise TypeError(f'toccd_gpu(): unsupported dtype {a.dtype}. Valid dtypes are float32 and float64')

    out /= out.sum()
    if set_total is not None:
        out *= set_total
    else:
        out *= a.sum()
    return out


# only define kernels if cupy has been loaded
if cp:
    kernel_step1 = r'''
extern "C" __global__
void rebin2D_step1_TYPE(TYPE *g_in, TYPE *g_tmp, int inx, int iny, int outx, int outy,
                   int dx_out, TYPE oneOverDxIn) {

   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int i, pos, prev_pos;
   TYPE value;
   TYPE res=0;

   if ((y<iny) && (x<outx)) {
       i = x*dx_out;
       prev_pos = i * oneOverDxIn;
       value = g_in[y*inx + prev_pos];

       for ( ; i<(x+1)*dx_out; i++) {
          pos = i * oneOverDxIn;
          if (pos != prev_pos) {
             value = g_in[y*inx + pos];
             prev_pos = pos;
          }
          res += value;
       }
   g_tmp[y*outx+x] =res;
   }
}
'''
    kernel_step2  = r'''
extern "C" __global__
void rebin2D_step2_TYPE(TYPE *g_tmp, TYPE* g_out, int outx, int outy,
                   int dy_in, int dy_out, TYPE f) {

   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int j;

   if ((y<outy) && (x<outx)) {
       g_out[y*outx+x]=0;
       for (j=y*dy_out; j<(y+1)*dy_out; j++)
          g_out[y*outx+x] += g_tmp[(j/dy_in)*outx + x];
       g_out[y*outx+x] *= f;
    }
}
'''
    _rebin2D_step1_float = cp.RawKernel(kernel_step1.replace('TYPE', 'float'), name='rebin2D_step1_float')
    _rebin2D_step2_float = cp.RawKernel(kernel_step2.replace('TYPE', 'float'), name='rebin2D_step2_float')
    _rebin2D_step1_double = cp.RawKernel(kernel_step1.replace('TYPE', 'double'), name='rebin2D_step1_double')
    _rebin2D_step2_double = cp.RawKernel(kernel_step2.replace('TYPE', 'double'), name='rebin2D_step2_double')
