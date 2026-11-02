'''
Geometric propagation of several sources through several layers in a single
GPU kernel (see AtmoPropagation).

The field of a source on the pupil is the product of the amplitudes and the
sum of the phases of the layers, each interpolated on the footprint of the
source on that layer. Done layer by layer, each interpolation writes a
temporary array, which is then read again to update the output, and reads two
arrays of coordinates as large as the pupil: the propagation is limited by the
memory traffic. Here each thread computes one pixel of one source, through all
the layers, and writes the output once.

The footprint of a source on a layer is one of:
* Copy: the layer is on the pupil grid, the pixels are read at an offset;
* Separable: one coordinate per column and one per row (no rotation);
* Grid: one coordinate per pixel (rotated layers).
The coordinates are those computed by Interp2D, and the interpolation is its
own bilinear_interp(), in the same order of the layers: the results are the
same as with Interp2D.
'''
from collections import namedtuple
from functools import cache

import numpy as np

import iteras
from iteras.lib.interp2d import Interp2D

Copy = namedtuple('Copy', 'row col')        # offset of the footprint on the layer
Separable = namedtuple('Separable', 'x y')  # coordinates of the columns and of the rows
Grid = namedtuple('Grid', 'x y')            # coordinates of each pixel, flattened

# Footprint modes in the kernel
_COPY, _SEPARABLE, _GRID = 0, 1, 2

_kernel_source = Interp2D.bilinear_interp_device + f'''
    #define COPY {_COPY}
    #define SEPARABLE {_SEPARABLE}
''' + r'''
    extern "C" __global__
    void geometric_propagation_TYPE(
            const TYPE * const *layers,     // field of each layer: amplitude, then phase
            const int *layer_dx, const int *layer_dy, int n_layers,
            const int *mode,                // footprint of each (source, layer)
            const long long *offset_x, const long long *offset_y,
            const TYPE *coords,             // coordinates of the footprints
            TYPE * const *outputs,          // field of each source: amplitude, then phase
            int n) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int source = blockIdx.z;
        if (x >= n || y >= n)
            return;

        TYPE amplitude = 1;
        TYPE phase = 0;
        for (int l = 0; l < n_layers; l++) {
            int k = source * n_layers + l;
            int dx = layer_dx[l];
            int dy = layer_dy[l];
            const TYPE *layer_amplitude = layers[l];
            const TYPE *layer_phase = layers[l] + (long long)dx * dy;
            if (mode[k] == COPY) {
                long long idx = (y + offset_y[k]) * dx + x + offset_x[k];
                amplitude *= layer_amplitude[idx];
                phase += layer_phase[idx];
            } else {
                TYPE xcoord, ycoord;
                if (mode[k] == SEPARABLE) {
                    xcoord = coords[offset_x[k] + x];
                    ycoord = coords[offset_y[k] + y];
                } else {
                    xcoord = coords[offset_x[k] + y * n + x];
                    ycoord = coords[offset_y[k] + y * n + x];
                }
                amplitude *= bilinear_interp((TYPE *)layer_amplitude, dx, dy, xcoord, ycoord);
                phase += bilinear_interp((TYPE *)layer_phase, dx, dy, xcoord, ycoord);
            }
        }
        TYPE *out = outputs[source];
        out[y * n + x] = amplitude;
        out[(long long)n * n + y * n + x] = phase;
    }
    '''


@cache
def _kernel(type_name):  # pragma: gpu
    return iteras.cp.RawKernel(_kernel_source.replace('TYPE', type_name),
                               name=f'geometric_propagation_{type_name}')


def footprint(interpolator, layer, n):
    '''
    Footprint on *layer* of a source whose field has n x n pixels, from the
    Interp2D *interpolator* of AtmoPropagation (None: the central part of the layer)
    '''
    if interpolator is None:
        return Copy(row=(layer.size[0] - n) // 2, col=(layer.size[1] - n) // 2)
    xx = interpolator.xx.reshape(n, n)
    yy = interpolator.yy.reshape(n, n)
    if bool((xx == xx[0]).all()) and bool((yy == yy[:, :1]).all()):
        return Separable(x=xx[0].copy(), y=yy[:, 0].copy())
    return Grid(x=interpolator.xx, y=interpolator.yy)


class GeometricPropagation:  # pragma: gpu
    '''
    Fields of the sources (ElectricField *outputs*, n x n) through the *layers*
    (in propagation order), with the *footprints* of each source on each layer
    (one list per source, see footprint()). The layers and the outputs must
    keep their arrays (updated in place), whose addresses are read here.
    '''

    def __init__(self, layers, outputs, footprints, dtype):
        cp = iteras.cp
        self.n = outputs[0].A.shape[0]
        self.n_sources = len(outputs)
        self.n_layers = len(layers)
        self.dtype = dtype

        self.layers = cp.array([layer.field.data.ptr for layer in layers], dtype=cp.uint64)
        self.layer_dx = cp.array([layer.field.shape[2] for layer in layers], dtype=cp.int32)
        self.layer_dy = cp.array([layer.field.shape[1] for layer in layers], dtype=cp.int32)
        self.outputs = cp.array([ef.field.data.ptr for ef in outputs], dtype=cp.uint64)

        # One entry per (source, layer), sources first; the coordinates of all
        # the footprints in one array
        mode, offset_x, offset_y, coords = [], [], [], []
        size = 0
        for source_footprints in footprints:
            for fp in source_footprints:
                if isinstance(fp, Copy):
                    mode.append(_COPY)
                    offset_x.append(fp.col)
                    offset_y.append(fp.row)
                else:
                    mode.append(_SEPARABLE if isinstance(fp, Separable) else _GRID)
                    offset_x.append(size)
                    offset_y.append(size + fp.x.size)
                    coords += [fp.x.ravel(), fp.y.ravel()]
                    size += fp.x.size + fp.y.size
        self.mode = cp.array(mode, dtype=cp.int32)
        self.offset_x = cp.array(offset_x, dtype=cp.int64)
        self.offset_y = cp.array(offset_y, dtype=cp.int64)
        self.coords = cp.concatenate([cp.asarray(c, dtype=dtype) for c in coords]) if coords \
            else cp.zeros(1, dtype=dtype)

    def run(self):
        '''Compute the fields of all the sources, on the current stream'''
        block = (16, 16, 1)
        grid = ((self.n + 15) // 16, (self.n + 15) // 16, self.n_sources)
        type_name = 'float' if self.dtype == np.float32 else 'double'
        _kernel(type_name)(grid, block, (
            self.layers, self.layer_dx, self.layer_dy, np.int32(self.n_layers),
            self.mode, self.offset_x, self.offset_y, self.coords,
            self.outputs, np.int32(self.n)))
