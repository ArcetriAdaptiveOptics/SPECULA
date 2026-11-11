from specula import RAD2ASEC
from specula.lib.make_mask import make_mask
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.abstract_coronagraph import Coronagraph

class FocalPlaneFilter(Coronagraph):
    """
    Basic focal plane filter processing object. Consists of a round pupil with a given
    diameter and optional obstruction in the center.

    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters containing pixel_pupil and pixel_pitch.
    wavelengthInNm : float [nm]
        Wavelength.
    fov : float [arcsec]
        Diameter of the field stop.
    fov_errinf : float, optional
        Relative error allowed on the inner part of the FoV (default: 0.1).
    fov_errsup : float, optional
        Relative error allowed on the outer part of the FoV (default: 10).
    fft_res : float [pixels per lambda/D], optional
        Minimum resolution of the focal plane (default: 3.0).
    fp_obs : float [lambda/D], optional
        Diameter of the central obstruction (default: 0.0, no obstruction).
        It must be smaller than the field stop.
    target_device_idx : int, optional
        Target device index (default: None, uses the global setting).
    precision : int, optional
        0 for double, 1 for single precision (default: None, uses the global setting).
    """
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
        # Field stop diameter as a fraction of the FFT frame, which spans
        # lambda / pixel_pitch (before the interpolation by fov_res)
        self.fp_masking = fov / (wavelengthInNm * 1e-9 / simul_params.pixel_pitch * RAD2ASEC)
        super().__init__(simul_params=simul_params,
                 wavelengthInNm=wavelengthInNm,
                 fov=fov,
                 fov_errinf=fov_errinf,
                 fov_errsup=fov_errsup,
                 fft_res=fft_res,
                 target_device_idx=target_device_idx,
                 precision=precision)


    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()

    def make_pupil_plane_mask(self):
        return 1.0

    def make_focal_plane_mask(self):
        fp_masking = self.fp_masking / self.fov_res
        fp_obsratio = 0
        if self.fp_obs:
            # make_mask() takes the obstruction as a fraction of the field stop
            # diameter: fp_masking times the frame, of fft_totsize / fft_res lambda/D
            stop_diameter_ld = fp_masking * self.fft_totsize / self.fft_res
            fp_obsratio = self.fp_obs / stop_diameter_ld
            if fp_obsratio >= 1:
                raise ValueError(f'fp_obs ({self.fp_obs} lambda/D) must be smaller than the '
                                 f'field stop ({stop_diameter_ld:.2f} lambda/D)')
        fp_mask = make_mask(self.fft_totsize, diaratio=fp_masking, obsratio=fp_obsratio, xp=self.xp)
        return fp_mask
