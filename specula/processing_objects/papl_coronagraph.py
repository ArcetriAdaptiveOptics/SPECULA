
from specula.processing_objects.app_coronagraph import APPCoronagraph
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask


class PAPLCoronagraph(APPCoronagraph):
    """
    Phase-apodized-pupil Lyot (PAPL) coronagraph class.
    This class implements a PAPL coronagraph, which uses a phase-only mask in the pupil plane
    to create a dark hole in the focal plane.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 pupil,
                 contrastInDarkHole:float,
                 iwaInLambdaOverD:float,
                 owaInLambdaOverD:float,
                 fpmIWAInLambdaOverD:float,
                 fpmOWAInLambdaOverD:float=None,
                 knife_edge:bool=True,
                 outerStopAsRatioOfPupil:float=1.0,
                 innerStopAsRatioOfPupil:float=0.0,
                 fft_res: float = 3.0,
                 make_symmetric: bool = False,
                 beta: float = 0.9,
                 target_device_idx: int = None,
                 precision: int = None
                ):

        if min(innerStopAsRatioOfPupil, outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')

        if knife_edge is True and fpmOWAInLambdaOverD is not None:
            raise ValueError('OWA cannot be defined for the knife-edge focal plane mask')

        self._knife_edge = knife_edge
        if knife_edge:
            self._fedge = fpmIWAInLambdaOverD
        else:
            self._iwa = fpmIWAInLambdaOverD
            self._owa = fpmOWAInLambdaOverD

        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil

        super().__init__(simul_params=simul_params,
                        wavelengthInNm=wavelengthInNm,
                        pupil=pupil,
                        contrastInDarkHole=contrastInDarkHole,
                        iwaInLambdaOverD=iwaInLambdaOverD,
                        owaInLambdaOverD=owaInLambdaOverD,
                        fft_res=fft_res,
                        make_symmetric=make_symmetric,
                        beta=beta,
                        target_device_idx=target_device_idx,
                        precision=precision)

    def make_focal_plane_mask(self):
        if self._knife_edge:
            xc = 2*(self._fedge * self.fft_res + self.fft_totsize//2)/ self.fft_totsize
            fp_mask = make_mask(self.fft_totsize, diaratio=1.0, xc=xc, xp=self.xp, square=True)
        else:
            owa_oversampled = self._owa * self.fft_res if self._owa is not None else self.fft_totsize
            fp_obsratio = self._iwa / owa_oversampled
            fp_diaratio = owa_oversampled / self.fft_totsize
            fp_mask = make_mask(self.fft_totsize, diaratio=fp_diaratio,
                                obsratio=fp_obsratio, xp=self.xp)
        return fp_mask

    def make_pupil_stop(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop,
                            obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask
