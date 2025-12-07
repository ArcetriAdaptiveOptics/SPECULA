from specula.processing_objects.abstract_coronograph import Coronograph
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
# from specula import RAD2ASEC


class FourQuadrantCoronograph(Coronograph):

    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 innerStopAsRatioOfPupil: float = 0.0,
                 outerStopAsRatioOfPupil: float = 1.0,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        
        if min(innerStopAsRatioOfPupil,outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')
        
        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil
        super().__init__(simul_params=simul_params,
                         wavelengthInNm=wavelengthInNm,
                         fft_res=fft_res,
                         target_device_idx=target_device_idx, 
                         precision=precision)

        
    def make_focal_plane_mask(self):
        """ Make a quadrant mask, where 2 opposite quadrants apply a pi phase delay """
        left_mask = make_mask(self.fft_totsize, diaratio=1.0, xc=1.0, xp=self.xp)
        bottom_mask = make_mask(self.fft_totsize, diaratio=1.0, yc=1.0, xp=self.xp)
        quad_mask = self.xp.logical_xor(left_mask,bottom_mask)
        fp_mask = self.xp.exp(1j*quad_mask*self.xp.pi, dtype=self.xp.complex64)
        return fp_mask
    
    def make_pupil_plane_mask(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop, obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask
        

    