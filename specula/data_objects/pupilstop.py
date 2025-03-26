
from astropy.io import fits

from specula.data_objects.layer import Layer
from specula.lib.make_mask import make_mask


class Pupilstop(Layer):
    '''Pupil stop'''

    def __init__(self,
                 pixel_pupil: int=160.0,
                 pixel_pitch: float=0.05,
                 input_mask = None,
                 mask_diam: float=1.0,
                 obs_diam: float=None,
                 shiftXYinPixel=(0.0, 0.0),
                 rotInDeg: float=0.0,
                 magnification: float=1.0,
                 target_device_idx: int=None,
                 precision: int=None):

        super().__init__(pixel_pupil, pixel_pupil, pixel_pitch, height=0, target_device_idx=target_device_idx, precision=precision,
                         shiftXYinPixel=shiftXYinPixel, rotInDeg=rotInDeg, magnification=magnification)

        self._input_mask = input_mask
        self._mask_diam = mask_diam
        self._obs_diam = obs_diam

        if self._input_mask is not None:
            self._input_mask = self.xp.array(input_mask)
            mask_amp = self._input_mask
        else:
            mask_amp = make_mask(pixel_pupil, obs_diam, mask_diam, xp=self.xp)
        self.A = mask_amp

        # Initialise time for at least the first iteration
        self._generation_time = 0

    def save(self, filename, hdr=None):
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1

        super().save(filename, hdr)

        fits.append(filename, self._A)
        fits.append(filename, self._A.shape)
        fits.append(filename, [self._pixel_pitch])

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        version = int(hdr['VERSION'])

        if version != 1:
            raise ValueError(f"Error: unknown version {version} in file {filename}")

        input_mask = fits.getdata(filename, ext=1)
        dim = fits.getdata(filename, ext=2)
        pixel_pitch = fits.getdata(filename, ext=3)[0]

        pupilstop = Pupilstop(dim[0], pixel_pitch, input_mask=input_mask, target_device_idx=target_device_idx)
        return pupilstop

    # TODO: this is a data object, not a processing object
    def finalize(self):
        pass
