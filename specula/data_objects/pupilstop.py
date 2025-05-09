
from astropy.io import fits
import numpy as np

from specula.data_objects.layer import Layer
from specula.lib.make_mask import make_mask
from specula.data_objects.simul_params import SimulParams
from specula import cpuArray

class Pupilstop(Layer):
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

        self.simul_params = simul_params
        self.pixel_pupil = self.simul_params.pixel_pupil
        self.pixel_pitch = self.simul_params.pixel_pitch

        super().__init__(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, height=0,
                        shiftXYinPixel=shiftXYinPixel, rotInDeg=rotInDeg, magnification=magnification,
                        target_device_idx=target_device_idx, precision=precision)

        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if self._input_mask is not None:
            self.A[:] = self.xp.array(input_mask)
        else:
            self.A[:] = make_mask(self.pixel_pupil, obs_diam, mask_diam, xp=self.xp)

        # Initialise time for at least the first iteration
        self._generation_time = 0

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Pupilstop'
        return hdr

    def save(self, filename):
        super().save(filename)
        fits.append(filename, np.zeros(1), self.get_fits_header())

    def read(self, filename):
        # Nothing extra to read
        super().read(filename)
 
    @staticmethod
    def restore(filename):
        ef_hdr = fits.getheader(filename, ext=0)

        version = ef_hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")

        dimx = ef_hdr['DIMX']
        pitch = ef_hdr['PIXPITCH']
        layer = Layer.restore(filename)
        tempParams = SimulParams(dimx, pitch)
        pupilstop = Pupilstop(tempParams, input_mask=layer.A)
        return pupilstop
    