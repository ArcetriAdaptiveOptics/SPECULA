from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
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
    """
    Zernike WFS slopes computer processing object. 
    Computes Zernike WFS slopes from pixel data using the pupil intensity.
    """
    
    def __init__(self,
                 pup_diam: float,
                 ccd_size: int,
                 obsratio:float = None,
                 sn: Slopes=None,
                 target_device_idx: int=None,
                 thr_value: float=0.0,
                 precision: int=None):

        cx = ccd_size/2
        cy = ccd_size/2

        _,ids = make_mask(np_size=ccd_size, diaratio = pup_diam/float(ccd_size), obsratio=obsratio,get_idx=True)
        mask_ids = ids[0]*ccd_size+ids[1]

        self.pupdata = PupData(
            ind_pup=mask_ids,
            radius=pup_diam/2,
            cx=cx,cy=cy,
            framesize=[ccd_size,ccd_size],
            target_device_idx=target_device_idx
        )
        self.pupdata.set_slopes_from_intensity()

        super().__init__(sn=sn, target_device_idx=target_device_idx, precision=precision)
        self.threshold = thr_value

        self.outputs['out_pupdata'] = self.pupdata
        
        self.slopes.single_mask = self.pupdata.single_mask() # unsure if this is needed
        self.slopes.display_map = self.pupdata.display_map # unsure if this is needed

        all_idx = self.pupdata.pupil_idx(0).astype(self.xp.int64)
        self.pup_idx  = all_idx[all_idx >= 0]

    @classmethod
    def input_names(cls):
        return {'in_pixels': InputDesc(Pixels, 'Input pixel data from detector')}

    @classmethod
    def output_names(cls):
        return {'out_slopes': OutputDesc(Slopes, 'Computed wavefront slopes'),
                'out_flux_per_subaperture': OutputDesc(BaseValue, 'Flux per subaperture'),
                'out_total_counts': OutputDesc(BaseValue, 'Total photon counts'),
                'out_subap_counts': OutputDesc(BaseValue, 'Counts per subaperture'),
                'out_pupdata': OutputDesc(PupData, 'Pupil data with subaperture geometry')}

    def nsubaps(self):
        return self.pupdata.n_subap

    def nslopes(self):
        return len(self.pupdata.pupil_idx(0))

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.to_xp(self.local_inputs['in_pixels'].pixels).flatten().astype(self.dtype)

    def trigger_code(self):
        self.flat_pixels -= self.threshold

        # clamp_generic_less(0,0,self.flat_pixels, xp=self.xp) # unsure wheter this is required
        metaintensity = self.flat_pixels[self.pup_idx].astype(self.dtype)
        self.flux_per_subaperture_vector.value[:] = metaintensity

        # Compute total intensity
        self.total_intensity = self.xp.sum(metaintensity)
        self.total_counts.value[0] = self.total_intensity
        self.subap_counts.value[0] = self.total_intensity / self.nsubaps()

        norm_factor = self.xp.sum(metaintensity) / self.nsubaps()
        self.slopes.slopes = metaintensity / norm_factor

    def post_trigger(self):
        super().post_trigger()

        self.outputs['out_pupdata'].generation_time = self.current_time
