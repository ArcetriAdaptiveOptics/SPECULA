from specula.processing_objects.abstract_coronagraph import Coronagraph
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula import RAD2ASEC, np


class FourQuadrantCoronagraph(Coronagraph):
    """
    Focal plane mask processing object. Generates a mask with four quadrants.
    """
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 innerStopAsRatioOfPupil: float = 0.0,
                 outerStopAsRatioOfPupil: float = 1.0,
                 phase_delay: float = np.pi,
                 fft_res: float = 3.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):

        if min(innerStopAsRatioOfPupil,outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')

        fov = wavelengthInNm * 1e-9 / simul_params.pixel_pitch * RAD2ASEC
        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil
        self._phase_delay = phase_delay
        super().__init__(simul_params=simul_params,
                         wavelengthInNm=wavelengthInNm,
                         fov=fov,
                         fft_res=fft_res,
                         center_on_pixel=False,
                         target_device_idx=target_device_idx,
                         precision=precision)


    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Input electric field from the telescope pupil')}

    @classmethod
    def output_names(cls):
        return {'out_ef': OutputDesc(ElectricField, 'Output electric field after four-quadrant coronagraph mask application')}

    def make_focal_plane_mask(self):
        """ Make a quadrant mask, where 2 opposite quadrants apply a pi phase delay """
        # left_mask = make_mask(self.fft_totsize, diaratio=1.0, xc=1.0, xp=self.xp, square=True)
        # bottom_mask = make_mask(self.fft_totsize, diaratio=1.0, yc=1.0, xp=self.xp, square=True)
        left_mask = make_mask(self.fft_totsize, diaratio=1.0,
                              xc=1.0+1./self.fft_totsize, xp=self.xp,
                              square=True, yc = 1./self.fft_totsize)
        bottom_mask = make_mask(self.fft_totsize, diaratio=1.0,
                                yc=1.0+1./self.fft_totsize, xp=self.xp,
                                square=True, xc = 1./self.fft_totsize)
        quad_mask = self.xp.logical_xor(left_mask,bottom_mask)
        iu = self.complex_dtype(1j)
        phase_delay = self.dtype(self._phase_delay)
        fp_mask = self.xp.exp(iu * quad_mask * phase_delay, dtype=self.complex_dtype)
        return fp_mask

    def make_pupil_plane_mask(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop,
                            obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask
    