from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.data_objects.slopes import Slopes
from specula.data_objects.pupdata import PupData

from specula.lib.make_mask import make_mask


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


# @fuse(kernel_name='clamp_generic_more')
# def clamp_generic_more(x, c, y, xp):
#     y[:] = xp.where(y > x, c, y)


class ZwfsSlopec(Slopec):
    def __init__(self,
                 radius: int,
                 ccd_size: tuple,
                 obsratio:float = None,
                 sn: Slopes=None,
                 target_device_idx: int=None,
                 thr_value: float=0,
                 precision: int=None):

        cx = ccd_size[1]/2
        cy = ccd_size[0]/2

        _,ind_pup=make_mask(np_size=ccd_size[0], diaratio = 2*radius/float(ccd_size[0]), obsratio=obsratio, get_idx=True)

        self.pupdata = PupData(
            ind_pup=ind_pup,
            radius=radius,
            cx=cx,cy=cy,
            framesize=ccd_size,
            target_device_idx=target_device_idx
        )

        super().__init__(sn=sn, target_device_idx=target_device_idx, precision=precision)
        self.threshold = thr_value

        self.outputs['out_pupdata'] = self.pupdata
        
        self.slopes.single_mask = self.pupdata.single_mask()
        self.slopes.display_map = self.pupdata.display_map

        all_idx = self.pupdata.pupil_idx(0).astype(self.xp.int64)
        self.pup_idx  = all_idx[all_idx >= 0]

    def nsubaps(self):
        return self.pupdata.n_subap

    def nslopes(self):
        return len(self.pupdata.pupil_idx(0))

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.local_inputs['in_pixels'].pixels.flatten()

    def trigger_code(self):

        self.flat_pixels -= self.threshold

        clamp_generic_less(0,0,self.flat_pixels, xp=self.xp)
        metaintensity = self.flat_pixels[self.pup_idx].astype(self.xp.float32)
        self.flux_per_subaperture_vector.value[:] = metaintensity

        # Compute total intensity
        self.total_intensity = self.xp.sum(metaintensity)
        self.total_counts.value[0] = self.total_intensity
        self.subap_counts.value[0] = self.total_intensity / self.nsubaps()

        self.slopes.slopes = metaintensity

    def post_trigger(self):
        super().post_trigger()

        self.outputs['out_pupdata'].generation_time = self.current_time
