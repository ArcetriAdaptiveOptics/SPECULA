from specula import RAD2ASEC
from specula.lib.make_mask import make_mask
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.abstract_coronagraph import Coronagraph

class FocalPlaneFilter(Coronagraph):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 fov: float,
                 fov_errinf: float = 0.1,
                 fov_errsup: float = 10,
                 fft_res: float = 3.0,
                 fp_obs: float = 0.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        self.fp_obs = fp_obs
        self.fp_masking = fov / (wavelengthInNm * 1e-9 / simul_params.pixel_pitch * RAD2ASEC)
        super().__init__(simul_params=simul_params,
                 wavelengthInNm=wavelengthInNm,
                 fov=fov,
                 fov_errinf=fov_errinf,
                 fov_errsup=fov_errsup,
                 fft_res=fft_res,
                 target_device_idx=target_device_idx,
                 precision=precision)


    def make_pupil_plane_mask(self):
        return 1.0
    
    def make_focal_plane_mask(self):
        fp_masking = self.fp_masking / self.fov_res
        fp_obsratio = self.fp_obs * fp_masking if self.fp_obs else 0
        fp_mask = make_mask(self.fft_totsize, diaratio=fp_masking, obsratio=fp_obsratio, xp=self.xp)
        return fp_mask

