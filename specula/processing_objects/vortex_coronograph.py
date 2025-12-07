from specula.processing_objects.abstract_coronograph import Coronograph
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
# from specula import RAD2ASEC


class VortexCoronograph(Coronograph):

    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 vortexCharge: float,
                 innerStopAsRatioOfPupil: float = 0.0,
                 outerStopAsRatioOfPupil: float = 1.0,
                 addInVortex:bool=False,
                 inVortexRadInLambdaOverD:float=None,
                 inVortexCharge:int=None,
                 inVortexShift:float=None,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        
        if min(innerStopAsRatioOfPupil,outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')
        
        self._charge = vortexCharge
        self._inVortex = addInVortex
        if addInVortex: # default inner vortex: same charge as outer/master vortex, pi shift, 0.621 lambda/D diameter
            self._innerRadInLambdaOverD = 0.62 if inVortexRadInLambdaOverD is None else inVortexRadInLambdaOverD
            self._innerCharge = vortexCharge if inVortexCharge is None else inVortexCharge
            self._innerShift = self.xp.pi if inVortexShift is None else inVortexShift
        
        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil
        super().__init__(simul_params=simul_params,
                         wavelengthInNm=wavelengthInNm,
                         fft_res=fft_res,
                         target_device_idx=target_device_idx, 
                         precision=precision)

        
    def make_focal_plane_mask(self):
        """ Make a 'vortex' mask, where the phase delay changes azimuthally
         from 0 to 2 pi a number of times equal to vortexCharge """
        N = self.fft_totsize
        c = N//2
        X, Y = self.xp.mgrid[0:N, 0:N]   
        theta = self.xp.arctan2((X - c), (Y - c))
        theta = (theta + 2 * self.xp.pi) % (2 * self.xp.pi)
        vortex = self._charge * theta
        if self._inVortex is True:
            rho = self.xp.sqrt((X-c)**2+(Y-c)**2)
            inVortex = self._innerCharge * theta + self._innerShift
            inRho = self._innerRadInLambdaOverD * self.fft_res
            vortex[rho<=inRho] = inVortex[rho<=inRho]
        fp_mask = self.xp.exp(1j*vortex)
        return fp_mask
    
    def make_pupil_plane_mask(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop, obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask
        

    