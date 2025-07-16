
from astropy.io import fits

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
            self._input_mask = self.to_xp(input_mask)
            mask_amp = self._input_mask
        else:
            mask_amp = make_mask(self.pixel_pupil, obs_diam, mask_diam, xp=self.xp)
        self.A = mask_amp

        # Initialise time for at least the first iteration
        self._generation_time = 0

    def get_fits_header(self):
        hdr = super().get_fits_header()
        hdr['OBJ_TYPE'] = 'Pupilstop'
        hdr['VERSION'] = 1
        return hdr

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = self.get_fits_header()
        super().save(filename, hdr)

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        # Takes the values from the header
        dimx = int(hdr['DIMX'])
        pixel_pitch = float(hdr['PIXPITCH'])

        tempParams = SimulParams(dimx, pixel_pitch)
        
        # Use electric field constructor to create the Pupilstop
        with fits.open(filename) as hdul:
            pupilstop = Pupilstop(tempParams, target_device_idx=target_device_idx)
            pupilstop.A[:] = hdul[0].data
            pupilstop.phaseInNm[:] = hdul[1].data
            
        return pupilstop
