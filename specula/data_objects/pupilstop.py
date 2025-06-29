
from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.layer import Layer
from specula.lib.make_mask import make_mask
from specula.data_objects.simul_params import SimulParams
from specula import cpuArray


class Pupilstop(BaseProcessingObj):
    '''Pupil stop'''

    def __init__(self,
                 simul_params: SimulParams,
                 input_mask = None,
                 mask_diam: float=1.0,
                 obs_diam: float=None,
                 shiftXYinPixel: tuple=(0.0, 0.0),
                 rotInDeg: float=0.0,
                 magnification: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        self.layer = Layer(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, height=0,
                           shiftXYinPixel=shiftXYinPixel, rotInDeg=rotInDeg, magnification=magnification,
                           target_device_idx=target_device_idx, precision=precision)

        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if self._input_mask is not None:
            self._input_mask = self.to_xp(input_mask)
            mask_amp = self._input_mask
        else:
            mask_amp = make_mask(self.pixel_pupil, obs_diam, mask_diam, xp=self.xp)
        self.layer.A = mask_amp
        self.outputs['out_layer'] = self.layer

    def trigger_code(self):
        self.layer.generation_time = self.current_time

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1

        super().save(filename, hdr)

        fits.append(filename, cpuArray(self.A))
        fits.append(filename, cpuArray(self.A.shape))
        fits.append(filename, cpuArray([self.pixel_pitch]))

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        input_mask = fits.getdata(filename, ext=1)
        dim = fits.getdata(filename, ext=2)
        pixel_pitch = fits.getdata(filename, ext=3)[0]

        tempParams = SimulParams(dim[0], pixel_pitch)
        pupilstop = Pupilstop(tempParams, input_mask=input_mask, target_device_idx=target_device_idx)
        return pupilstop
