
import warnings

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
        self.shiftXYinPixel = shiftXYinPixel
        self.rotInDeg = rotInDeg
        self.magnification = magnification

        super().__init__(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, height=0,
                        shiftXYinPixel=self.shiftXYinPixel,
                        rotInDeg=self.rotInDeg,
                        magnification=self.magnification,
                        target_device_idx=target_device_idx, precision=precision)

        self._input_mask = input_mask
        self.mask_diam = mask_diam
        self.obs_diam = obs_diam

        if self._input_mask is not None:
            self._input_mask = self.to_xp(input_mask,dtype=self.dtype)
            mask_amp = self._input_mask
        else:
            mask_amp = make_mask(self.pixel_pupil, obs_diam, mask_diam, xp=self.xp)

        # field dtype must be self.dtype
        if mask_amp.dtype != self.dtype:
            mask_amp = self.xp.asarray(mask_amp, dtype=self.dtype)

        self.A = mask_amp

        # Initialise time for at least the first iteration
        self.generation_time = 0

    def get_value(self):
        '''
        Get the amplitude mask as a numpy/cupy array
        '''
        return self.field[0]

    def set_value(self, v):
        '''
        Set a new amplitude mask.
        Arrays are not reallocated
        '''
        assert v.shape == self.field[0].shape, \
            f"Error: input array shape {v.shape} does not match pupilstop shape {self.field[0].shape}"
        self.field[0][:]= self.to_xp(v)

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'Pupilstop'
        hdr['PIXPUPIL'] = self.field[0].shape[0]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['SHIFTX'] = float(self.shiftXYinPixel[0])
        hdr['SHIFTY'] = float(self.shiftXYinPixel[1])
        hdr['ROTATION'] = self.rotInDeg
        hdr['MAGNIFIC'] = self.magnification
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.field[0]), name='AMPLITUDE'))
        # phaseInNm is not used in Pupilstop
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        pixel_pupil = int(hdr['PIXPUPIL'])
        pixel_pitch = float(hdr['PIXPITCH'])
        shiftX = float(hdr['SHIFTX'])
        shiftY = float(hdr['SHIFTY'])
        rotInDeg = float(hdr['ROTATION'])
        magnification = float(hdr['MAGNIFIC'])
        simul_params = SimulParams(pixel_pupil, pixel_pitch)
        pupilstop = Pupilstop(simul_params, shiftXYinPixel=(shiftX, shiftY), rotInDeg=rotInDeg, \
                              magnification=magnification, target_device_idx=target_device_idx)
        return pupilstop

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'Pupilstop':
            # Maybe it's a PASSATA object
            try:
                return Pupilstop.restore_from_passata(filename, target_device_idx)
            except ValueError as exc:
                raise ValueError(f"Error: file {filename} does not contain a SPECULA or PASSATA Pupilstop object") from exc

        pupilstop = Pupilstop.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            pupilstop.field[0] = pupilstop.to_xp(hdul[1].data.copy(), dtype=pupilstop.dtype)
            # phaseInNm is not used in Pupilstop
        return pupilstop

    @staticmethod
    def restore_from_passata(filename, target_device_idx=None):
        """Restore a :class:`~specula.data_objects.pupilstop.Pupilstop` object from a PASSATA format file."""
        with fits.open(filename) as hdul:
            if len(hdul) == 4:
                pixel_pitch = float(hdul[3].data[0])
                A = hdul[1].data.copy()
                pixel_pupil = A.shape[0]
                simul_params = SimulParams(pixel_pupil, pixel_pitch)
                pupilstop = Pupilstop(simul_params, input_mask=A, target_device_idx=target_device_idx)
                warnings.warn('Detected PASSATA pupilstop file', RuntimeWarning)
                return pupilstop
        raise ValueError(f"Error: file {filename} does not contain a PASSATA Pupilstop object")

    # array_for_display is inherited from Layer (ElectricField)
