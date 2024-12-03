from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.base_value import BaseValue
from specula.data_objects.pupdata import PupData


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)

@fuse(kernel_name='clamp_generic_less1')
def clamp_generic_less1(x, c, y, xp):
    y = xp.where(y < x, c, y)


@fuse(kernel_name='clamp_generic_more')
def clamp_generic_more(x, c, y, xp):
    y[:] = xp.where(y > x, c, y)


@fuse(kernel_name='clamp_generic_more1')
def clamp_generic_more1(x, c, y, xp):
    y = xp.where(y > x, c, y)


class PyrSlopec(Slopec):
    def __init__(self, pupdata: PupData=None, shlike=False, norm_factor=None, thr_value=0, slopes_from_intensity=False, 
                 target_device_idx=None, 
                 precision=None,
                **kwargs):
        super().__init__(target_device_idx=target_device_idx, precision=precision, **kwargs)
        self.shlike = shlike
        self.norm_factor = norm_factor
        self.thr_value = int(thr_value)
        self.threshold = self.thr_value if self.thr_value != -1 else None
        self.slopes_from_intensity = slopes_from_intensity
        if pupdata is not None:
            self.pupdata = pupdata  # Property set
            self.pup_idx  = self.pupdata.ind_pup.flatten().astype(self.xp.int64)
            self.pup_idx0 = self.pupdata.ind_pup[:, 0]
            self.pup_idx1 = self.pupdata.ind_pup[:, 1]
            self.pup_idx2 = self.pupdata.ind_pup[:, 2]
            self.pup_idx3 = self.pupdata.ind_pup[:, 3]
            self.n_pup = self.pupdata.ind_pup.shape[1]
            self.n_subap = self.pupdata.ind_pup.shape[0]

        self.total_counts = BaseValue()
        self.subap_counts = BaseValue()
        self.outputs['out_pupdata'] = self.pupdata
        self.outputs['total_counts'] = self.total_counts
        self.outputs['subap_counts'] = self.subap_counts

    @property
    def thr_value(self):
        return self._thr_value

    @thr_value.setter
    def thr_value(self, value):
        self._thr_value = int(value)

    @property
    def pupdata(self):
        return self._pupdata

    @pupdata.setter
    def pupdata(self, p):
        if p is not None:
            self._pupdata = p
            # TODO replace this resize with an earlier initialization
            if self.slopes_from_intensity:
                self.slopes.resize(len(self.pupdata.ind_pup) * 4)
            else:
                self.slopes.resize(len(self.pupdata.ind_pup) * 2)
            self.accumulated_slopes.resize(len(self.pupdata.ind_pup) * 2)

    def run_check(self, time_step, errmsg=''):
        self.prepare_trigger(0)
        #super().build_stream()
        if self.use_sn and not self.sn:
            errmsg += 'Slopes null are not valid'
        if self.weight_from_accumulated and self.accumulate:
            errmsg += 'weightFromAccumulated and accumulate must not be set together'
        if errmsg != '':
            print(errmsg)
        return not (self.weight_from_accumulated and self.accumulate) and self.local_inputs['in_pixels'] and self.slopes and ((not self.use_sn) or (self.use_sn and self.sn))

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.local_inputs['in_pixels'].pixels.flatten()

    def trigger_code(self):
        # outpus:
        # total_counts : computed here
        # subap_counts : computed in post_trigger
        # slopes : computed here

#        if not self.pupdata:
#            return
#        if self.verbose:
#            print('Average pixel counts:', self.xp.sum(pixels) / len(self.pupdata.ind_pup))

        self.total_counts.value = self.xp.sum(self.flat_pixels[self.pup_idx])

        if self.threshold is not None:
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
            inv_factor[0] = self.total_intensity / (4 * self.n_subap)
            factor = 1.0 / inv_factor[0]
            self.sx = factor * self.xp.concatenate([A, B])
            self.sy = factor * self.xp.concatenate([C, D])
        else:
            if self.norm_factor is not None:
                inv_factor[0] = self.norm_factor
                factor = 1.0 / inv_factor[0]
            elif not self.shlike:
                inv_factor[0] = self.total_intensity /  self.n_subap
                factor = 1.0 / inv_factor
            else:
                inv_factor[0] = self.xp.sum(self.flat_pixels[self.pup_idx])
                factor = 1.0 / inv_factor[0]

            self.sx = (A+B-C-D) * factor
            self.sy = (B+C-A-D) * factor

        clamp_generic_more(0, 1, inv_factor, xp=self.xp)
        self.sx *= inv_factor[0]
        self.sy *= inv_factor[0]

        self.slopes.xslopes = self.sx
        self.slopes.yslopes = self.sy 

        
    def post_trigger(self):
        # super().post_trigger()
        self.subap_counts.value = self.total_counts.value / self.pupdata.n_subap
        self.total_counts.generation_time = self.current_time
        self.subap_counts.generation_time = self.current_time
        self.slopes.generation_time = self.current_time


    def run_check(self, time_step, errmsg=''):
        if self.shlike and self.slopes_from_intensity:
            errmsg += 'Both SHLIKE and SLOPES_FROM_INTENSITY parameters are set. Only one of these should be used.'
            return False

        if self.shlike and self.norm_factor != 0:
            errmsg += 'Both SHLIKE and NORM_FACTOR parameters are set. Only one of these should be used.'
            return False

        if not self.pupdata:
            errmsg += 'Pupil data is not valid'
            return False

        return super().run_check(time_step, errmsg=errmsg)

