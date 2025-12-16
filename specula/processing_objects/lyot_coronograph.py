from specula.processing_objects.abstract_coronograph import Coronograph
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula import RAD2ASEC


class LyotCoronograph(Coronograph):

    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 iwaInLambdaOverD: float,
                 owaInLambdaOverD: float = None,
                 innerStopAsRatioOfPupil: float = 0.0,
                 outerStopAsRatioOfPupil: float = 1.0,
                 knife_edge: bool = False,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        
        if min(innerStopAsRatioOfPupil,outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')
        
        if knife_edge is True and owaInLambdaOverD is not None:
            raise ValueError('OWA cannot be defined for the knife-edge focal plane mask')
        
        fov = wavelengthInNm * 1e-9 / simul_params.pixel_pitch * RAD2ASEC
            
        self._knife_edge = knife_edge
        if knife_edge:
            self._fedge = iwaInLambdaOverD
        else:
            self._iwa = iwaInLambdaOverD
            self._owa = owaInLambdaOverD

        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil
        super().__init__(simul_params=simul_params,
                         wavelengthInNm=wavelengthInNm,
                         fov = fov,
                         fft_res = fft_res,
                         target_device_idx=target_device_idx, 
                         precision=precision)

        
    def make_focal_plane_mask(self):
        if self._knife_edge:
            xc = 1.0 + (self._fedge * self.fft_res * self.fov_res + 1.5)/ self.fft_totsize
            fp_mask = make_mask(self.fft_totsize, diaratio=1.0, xc=xc, xp=self.xp, square=True)
        else:
            owa_oversampled = self._owa * self.fft_res * self.fov_res if self._owa is not None else self.fft_totsize
            fp_obsratio = self._iwa * self.fft_res * self.fov_res / owa_oversampled * 2
            fp_diaratio = owa_oversampled / self.fft_totsize 
            fp_mask = make_mask(self.fft_totsize, diaratio=fp_diaratio, obsratio=fp_obsratio, xp=self.xp, xc=1/self.fft_totsize, yc=1/self.fft_totsize)
        return fp_mask
    
    def make_pupil_plane_mask(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop, obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask
        

    