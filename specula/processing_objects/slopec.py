
from specula import cp, np
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat

if cp is not None:
    @cp.fuse(kernel_name='sum_product')
    def _sum_product_gpu(a, b):
        return cp.sum(a * b, axis=-1)


def sum_product(a, b, xp):
    '''
    xp.sum(a * b, axis=-1), with broadcasting, without allocating a * b.
    Used instead of matrix products inside CUDA graphs, since CuPy does not
    allow cuBLAS calls during stream capture.
    '''
    if xp is np:
        return np.einsum('...i,...i->...', a, b)
    return _sum_product_gpu(a, b)


class Slopec(BaseProcessingObj):
    """
    Slope Computer abstract processing object.
    Base class for processing objects that compute slopes from pixel data, 
    such as Shack-Hartmann or Pyramid slopes.
    """
    def __init__(self,
                 sn: Slopes=None,
                 recmat: Recmat=None,
                 filt_intmat: Intmat=None,
                 filt_recmat: Recmat=None,
                 filtmat=None,
                 weight_int_pixel_dt: float=0,
                 interleave: bool=False,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.sn = sn
        self.slopes = Slopes(
            self.nslopes(), interleave=interleave,
            target_device_idx=self.target_device_idx
        )
        self.flux_per_subaperture_vector = BaseValue(
            value=self.xp.zeros(self.nsubaps(), dtype=self.dtype),
            target_device_idx=self.target_device_idx,
            precision=precision
        )

        self.total_counts = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                      target_device_idx=self.target_device_idx,
                                      precision=precision)
        self.subap_counts = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                      target_device_idx=self.target_device_idx,
                                      precision=precision)
        # 2d view of the slopes, e.g. shape (2, size_x, size_y) for a single
        # subaperture-sized x/y slope map. Allocated lazily on the first
        # apply_slopes_corrections() call, once a derived class has set self.slopes.single_mask
        # and self.slopes.display_map (see Slopes.get2d()). Its exact shape depends
        # on those, and it is not duplicated data: it is recomputed from self.slopes
        # at every step, not accumulated separately.
        self.slopes_map = BaseValue(target_device_idx=self.target_device_idx,
                                    precision=precision)
        self._slopes_map_unavailable = False
        # Flat index into slopes_map.value of each slope, see update_slopes_map()
        self._slopes_map_idx = None
        self.recmat = recmat
        if filtmat is not None:
            if filt_intmat:
                raise ValueError('filt_intmat must not be set if "filtmat" is set')
            if filt_recmat:
                raise ValueError('filt_recmat must not be set if "filtmat" is set')
            self.filt_intmat = Intmat(filtmat[0], target_device_idx=self.target_device_idx)
            self.filt_recmat = Recmat(filtmat[1], target_device_idx=self.target_device_idx)
        else:
            if bool(filt_intmat) != bool(filt_recmat):
                missing = 'filt_intmat' if filt_intmat is None else 'filt_recmat'
                raise ValueError(
                    'Both filt_intmat and filt_recmat must be set for slopes filtering. '
                    f'You provided one, but missing: {missing}'
                )
            self.filt_intmat = filt_intmat
            self.filt_recmat = filt_recmat

        self.weight_int_pixel_dt = self.seconds_to_t(weight_int_pixel_dt)
        if self.weight_int_pixel_dt > 0:
            self.weight_int_pixel = True
        else:
            self.weight_int_pixel = False
        self.int_pixels = None
        self.do_reset_accumulation = False

        self.inputs['in_pixels'] = InputValue(type=Pixels)
        self.outputs['out_slopes'] = self.slopes
        self.outputs['out_flux_per_subaperture'] = self.flux_per_subaperture_vector
        self.outputs['out_total_counts'] = self.total_counts
        self.outputs['out_subap_counts'] = self.subap_counts
        self.outputs['out_slopes_map'] = self.slopes_map

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Input pixel data from detector')}

    @classmethod
    def output_names(cls):
        return {'out_slopes': OutputDesc(Slopes, 'Computed wavefront slopes'),
                'out_flux_per_subaperture': OutputDesc(BaseValue, 'Flux per subaperture'),
                'out_total_counts': OutputDesc(BaseValue, 'Total photon counts'),
                'out_subap_counts': OutputDesc(BaseValue, 'Counts per subaperture'),
                'out_slopes_map': OutputDesc(BaseValue, '2d view of the slopes '
                                              '(e.g. shape (2, size_x, size_y)), see Slopes.get2d()')}

    # Derived classes must implement this method
    def nsubaps(self):
        raise NotImplementedError

    # Derived classes must implement this method
    def nslopes(self):
        raise NotImplementedError

    def do_accumulation(self, t):
        """
        Perform pixel accumulation based on the IDL version.
        This method should be called in trigger_code of derived classes.
        """
        if self.weight_int_pixel_dt <= 0:
            return

        current_pixels = self.inputs['in_pixels'].get(self.target_device_idx).pixels

        # Initialize accumulated pixels if not exists
        if self.int_pixels is None:
            self.int_pixels = Pixels(
                current_pixels.shape[0],
                current_pixels.shape[1],
                target_device_idx=self.target_device_idx
            )
            self.int_pixels.pixels = self.xp.zeros_like(current_pixels)

        # Check if we're at the start of a new accumulation period
        if self.do_reset_accumulation:
            # Reset accumulation
            self.int_pixels.pixels *= 0
            self.do_reset_accumulation = False

        # Add to existing accumulation
        self.int_pixels.pixels += current_pixels.astype(self.dtype, copy=False)

        if (t % self.weight_int_pixel_dt) == 0 and t >= self.weight_int_pixel_dt:
            # Update generation time
            self.int_pixels.generation_time = t
            self.do_reset_accumulation = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if 'trigger_code' in cls.__dict__:
            raise TypeError(f'{cls.__name__}: Slopec-derived classes must implement compute_slopes()'
                            f' instead of trigger_code(), so that the slope corrections'
                            f' (slope null, filtering) are applied')

    def trigger_code(self):
        '''
        Computes the slopes with compute_slopes(), implemented by derived
        classes, then applies the slope corrections. Derived classes must not
        override this method. If a derived class uses a CUDA graph, both are
        part of the graph.
        '''
        self.compute_slopes()
        self.apply_slopes_corrections()

    def compute_slopes(self):
        '''
        Derived classes must implement this method, computing self.slopes
        and the flux outputs from the input pixels.
        '''
        raise NotImplementedError(f'{self.__class__.__name__}: please implement compute_slopes() in your derived class!')

    def vecmat(self, v, m):
        '''
        Vector-matrix product v @ m. cuBLAS cannot be used in a CUDA graph,
        so objects using one use sum_product() instead.
        '''
        if self.stream is not None:
            return sum_product(m.T, v, xp=self.xp)
        return v @ m

    def apply_slopes_corrections(self):
        '''
        Slope null subtraction, reconstruction, filtering and 2d slopes map.
        GPU operations only, so that it can be part of a CUDA graph.
        '''
        if self.sn:
            n = self.slopes.size
            if self.sn.interleave == self.slopes.interleave and self.sn.size == n and n % 2 == 0:
                # Same layout: a single subtraction, without gathers and scatters
                self.slopes.slopes -= self.sn.slopes
            else:
                self.slopes.xslopes -= self.sn.xslopes
                self.slopes.yslopes -= self.sn.yslopes

        if self.recmat:
            m = self.vecmat(self.slopes.slopes, self.recmat.recmat)
            self.slopes.slopes[:] = m

        if self.filt_intmat and self.filt_recmat:
            m = self.vecmat(self.slopes.slopes, self.filt_recmat.recmat)
            sl0 = self.vecmat(m, self.filt_intmat.intmat.T)
            self.slopes.slopes -= sl0

        # Not duplicated storage: recomputed from self.slopes at every step
        # via the existing single_mask/display_map machinery (see Slopes.get2d()).
        # Some Slopec subclasses set up single_mask/display_map in a way that is
        # not (yet) compatible with get2d(); rather than crashing the whole
        # simulation, out_slopes_map is simply left unset for those, and a
        # warning is logged once.
        if self.slopes.single_mask is not None and self.slopes.display_map is not None:
            if not self._slopes_map_unavailable:
                self.update_slopes_map()

    def update_slopes_map(self):
        '''
        Update slopes_map from self.slopes. The first call uses Slopes.get2d(),
        the following ones scatter the slopes into the existing map with a
        single operation, using the same geometry. Geometries not handled
        here keep using get2d().
        '''
        if self._slopes_map_idx is not None:
            self.xp.put(self.slopes_map.value, self._slopes_map_idx, self.slopes.slopes)
            return
        try:
            self.slopes_map.set_value(self.slopes.get2d())
        except (IndexError, ValueError) as e:
            self._slopes_map_unavailable = True
            self.logger.warning(
                f'{self.__class__.__name__}: out_slopes_map could not be computed '
                f'({e}); this output will stay empty for this object.'
            )
            return

        # Same geometry as Slopes.get2d()
        idx = self.slopes.display_map
        mask_shape = self.slopes.single_mask.shape
        flat_idx = idx if len(idx.shape) == 1 else idx[0] * mask_shape[1] + idx[1]
        if self.slopes.slopes.size == len(idx):
            # slopes from intensity case
            if len(idx.shape) == 1:
                self._slopes_map_idx = flat_idx
        elif self.slopes.slopes.size == 2 * len(flat_idx):
            map_idx = self.xp.zeros(self.slopes.slopes.size, dtype=flat_idx.dtype)
            map_idx[self.slopes.indx()] = flat_idx
            map_idx[self.slopes.indy()] = flat_idx + mask_shape[0] * mask_shape[1]
            self._slopes_map_idx = map_idx

    def post_trigger(self):
        super().post_trigger()

        if self.slopes_map.value is not None and not self._slopes_map_unavailable:
            self.outputs['out_slopes_map'].generation_time = self.current_time

        self.outputs['out_slopes'].generation_time = self.current_time
        self.outputs['out_flux_per_subaperture'].generation_time = self.current_time
        self.outputs['out_total_counts'].generation_time = self.current_time
        self.outputs['out_subap_counts'].generation_time = self.current_time

        #rms = self.xp.sqrt(self.xp.mean(self.slopes.slopes**2))
        #self.logger.info('Slopes have been filtered. '
        #      'New slopes min, max and rms: '
        #      f'{self.slopes.slopes.min()}, {self.slopes.slopes.max()}, {rms}')
