from specula import fuse
from specula.processing_objects.slopec import Slopec
from specula.base_processing_obj import OutputDesc
from specula.data_objects.slopes import Slopes
from specula.data_objects.pupdata import PupData

from specula.lib.make_mask import make_mask


@fuse(kernel_name='clamp_generic_less')
def clamp_generic_less(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


class ZwfsSlopec(Slopec):
    """
    Zernike WFS slopes computer processing object. 
    Computes Zernike WFS slopes from pixel data using the pupil intensity.

    clamp_negative (see Slopec): the negative pixels are set to zero after
    the subtraction of thr_value. If the total intensity is zero or negative
    (possible with clamp_negative=False), the slopes are zero.

    Parameters
    ----------
    pup_diam : float [pixels]
        Pupil diameter on the detector.
    ccd_size : int [pixels]
        Detector size (square); the pupil is centred on it.
    obsratio : float, optional
        Central obstruction, as a fraction of the pupil diameter.
    sn : Slopes, optional
        Slope null (see Slopec).
    target_device_idx : int, optional
        Device index (-1 for CPU). Default: global setting.
    thr_value : float [counts], optional
        Subtracted from all the pixels. Default 0.
    clamp_negative : bool, optional
        Set the negative pixels to zero (see Slopec). Default True.
    precision : int, optional
        0 for double, 1 for single precision. Default: global setting.

    The output signal is the intensity of each pupil pixel divided by the
    mean intensity per pupil pixel.
    """
    
    def __init__(self,
                 pup_diam: float,
                 ccd_size: int,
                 obsratio:float = None,
                 sn: Slopes=None,
                 target_device_idx: int=None,
                 thr_value: float=0.0,
                 clamp_negative: bool=True,
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

        super().__init__(sn=sn, clamp_negative=clamp_negative,
                         target_device_idx=target_device_idx, precision=precision)
        self.threshold = thr_value

        self.outputs['out_pupdata'] = self.pupdata
        
        # One value per valid pixel, displayed on the whole frame
        self.slopes.layout = self.pupdata.slopes_layout(from_intensity=True)

        all_idx = self.pupdata.pupil_idx(0).astype(self.xp.int64)
        self.pup_idx  = all_idx[all_idx >= 0]

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({
            'out_pupdata': OutputDesc(PupData, 'Pupil data with subaperture geometry')
        })
        return result    

    def nsubaps(self):
        return self.pupdata.n_subap

    def nslopes(self):
        return len(self.pupdata.pupil_idx(0))

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.flat_pixels = self.to_xp(self.local_inputs['in_pixels'].pixels).flatten().astype(self.dtype)

    def compute_slopes(self):
        self.flat_pixels -= self.threshold

        if self.clamp_negative:
            clamp_generic_less(0, 0, self.flat_pixels, xp=self.xp)
        metaintensity = self.flat_pixels[self.pup_idx].astype(self.dtype)
        self.flux_per_subaperture_vector.value[:] = metaintensity

        # Compute total intensity
        self.total_intensity = self.xp.sum(metaintensity)
        self.total_counts.value[0] = self.total_intensity
        self.subap_counts.value[0] = self.total_intensity / self.nsubaps()

        norm_factor = self.xp.sum(metaintensity) / self.nsubaps()
        # zero slopes if the total intensity is not positive
        self.slopes.slopes = metaintensity / self.xp.where(norm_factor > 0, norm_factor, self.xp.inf)

    def post_trigger(self):
        super().post_trigger()

        self.outputs['out_pupdata'].generation_time = self.current_time
