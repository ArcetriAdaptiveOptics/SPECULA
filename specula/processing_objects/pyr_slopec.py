from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.data_objects.pupdata import PupData
from specula.data_objects.slopes import Slopes


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)


class PyrSlopec(Slopec):
    def __init__(self,
                 pupdata: PupData,
                 sn: Slopes=None,
                 shlike: bool=False,
                 norm_factor: float=None,   # TODO =1.0,
                 thr_value: float=0,
                 slopes_from_intensity: bool=False,
                 target_device_idx: int=None,
                 precision: int=None,
                **kwargs): # is this needed??

        # Set subaperture data before initializing base class
        # because we need to know the number of subapertures
        self.pupdata = pupdata
        self.slopes_from_intensity = slopes_from_intensity

        super().__init__(sn=sn, target_device_idx=target_device_idx, precision=precision, **kwargs)

        if shlike and slopes_from_intensity:
            raise ValueError('Both SHLIKE and SLOPES_FROM_INTENSITY parameters are set. Only one of these should be used.')

        if shlike and norm_factor != 0:
            raise ValueError('Both SHLIKE and NORM_FACTOR parameters are set. Only one of these should be used.')

        self.shlike = shlike
        self.norm_factor = norm_factor
        self.threshold = thr_value
        self.slopes_from_intensity = slopes_from_intensity
        if self.slopes_from_intensity:
            self.pupdata.set_slopes_from_intensity(slopes_from_intensity)   # TODO we should not modify an external object,
                                                                            # since it could be used elsewhere
        pupil_idx = self.pupdata.pupil_idx
        all_idx = self.xp.concatenate([pupil_idx(i) for i in range(4)]).astype(self.xp.int64)
        self.pup_idx  = all_idx[all_idx >= 0] # Exclude -1 padding
        self.pup_idx0 = pupil_idx(0)[pupil_idx(0) >= 0]  # Exclude -1 padding
        self.pup_idx1 = pupil_idx(1)[pupil_idx(1) >= 0]   # Exclude -1 padding
        self.pup_idx2 = pupil_idx(2)[pupil_idx(2) >= 0]   # Exclude -1 padding
        self.pup_idx3 = pupil_idx(3)[pupil_idx(3) >= 0]   # Exclude -1 padding
        self.outputs['out_pupdata'] = self.pupdata
        
        if self.slopes_from_intensity:
            self.slopes.single_mask = self.pupdata.complete_mask()
        else:
            self.slopes.single_mask = self.pupdata.single_mask()
        self.slopes.display_map = self.pupdata.display_map

    def nsubaps(self):
        return self.pupdata.n_subap

    def nslopes(self):
        if self.slopes_from_intensity:
            return len(self.pupdata.pupil_idx(0)) * 4
        else:
            return len(self.pupdata.pupil_idx(0)) * 2

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.local_inputs['in_pixels'].pixels.flatten()

    def _compute_pyr_slopes(self, A, B, C, D, factor):
        sx = (A+B-C-D) * factor
        sy = (B+C-A-D) * factor
        return sx, sy

    def trigger_code(self):

        self.total_counts.value[0] = self.xp.sum(self.flat_pixels[self.pup_idx])
        self.subap_counts.value[0] = self.total_counts.value[0] / self.nsubaps()

        self.flat_pixels -= self.threshold

        clamp_generic_less(0,0,self.flat_pixels, xp=self.xp)
        A = self.flat_pixels[self.pup_idx0].astype(self.xp.float32)
        B = self.flat_pixels[self.pup_idx1].astype(self.xp.float32)
        C = self.flat_pixels[self.pup_idx2].astype(self.xp.float32)
        D = self.flat_pixels[self.pup_idx3].astype(self.xp.float32)

        # Compute total intensity
        self.total_intensity = self.xp.sum(self.flat_pixels[self.pup_idx])

        # Use 1-length array to allow clamp() on both GPU arrays and CPU scalars
        inv_factor = self.xp.zeros(1, dtype=self.dtype)

        if self.slopes_from_intensity:
            inv_factor[0] = self.total_intensity / (4 * self.nsubaps())
            factor = 1.0 / inv_factor[0]
            self.sx = factor * self.xp.concatenate([A, B])
            self.sy = factor * self.xp.concatenate([C, D])
        else:
            if self.norm_factor is not None:
                inv_factor[0] = self.norm_factor
                factor = 1.0 / inv_factor[0]
            elif not self.shlike:
                inv_factor[0] = self.total_intensity /  self.nsubaps()
                factor = 1.0 / inv_factor
            else:
                inv_factor[0] = self.xp.sum(self.flat_pixels[self.pup_idx])
                factor = 1.0 / inv_factor[0]

            self.sx, self.sy = self._compute_pyr_slopes(A, B, C, D, factor)

        clamp_generic_more(0, 1, inv_factor, xp=self.xp)
        self.sx *= inv_factor[0]
        self.sy *= inv_factor[0]

        self.slopes.xslopes = self.sx
        self.slopes.yslopes = self.sy

    def post_trigger(self):
        super().post_trigger()

        self.outputs['out_pupdata'].generation_time = self.current_time

