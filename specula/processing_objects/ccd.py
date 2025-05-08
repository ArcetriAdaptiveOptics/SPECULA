import math

from scipy.stats import gamma
from scipy.ndimage import convolve

from specula import fuse
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.intensity import Intensity
from specula.lib.calc_detector_noise import calc_detector_noise
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.sh import SH
from specula.data_objects.simul_params import SimulParams


@fuse(kernel_name='clamp_generic')
def clamp_generic(x, c, y, xp):
    y[:] = xp.where(y < x, c, y)


class CCD(BaseProcessingObj):
    '''Simple CCD from intensity field'''
    def __init__(self,                 
                 size: int,           # TODO list=[80,80],
                 dt: float,           # TODO =0.001,
                 bandw: float,        # TODO =300.0,
                 name: str='',        # TODO ='OCAM2k',
                 binning: int=1,
                 photon_noise: bool=False,
                 readout_noise: bool=False,
                 excess_noise: bool=False,
                 darkcurrent_noise: bool=False,
                 background_noise: bool=False,
                 cic_noise: bool=False,
                 cte_noise: bool=False,
                 readout_level: str='', # check this is ok
                 darkcurrent_level: str='', # check this is ok
                 background_level: str='', # check this is ok
                 cic_level: float=0,
                 cte_mat=None, # ??
                 quantum_eff: float=1.0,
                 pixelGains=None,                 
                 photon_seed: int=1,
                 readout_seed: int=2,
                 excess_seed: int=3,
                 excess_delta: float=1.0,
                 start_time: int=0,
                 ADU_gain: int=8,
                 ADU_bias: int=400,
                 emccd_gain: int=1,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if readout_level and darkcurrent_level and background_level:
            # Compute RON and dark current
            if readout_level == 'auto' or darkcurrent_level == 'auto' or background_level == 'auto':
                noise = calc_detector_noise(1./dt, name, binning)
                if readout_level == 'auto':
                    readout_level = noise[0]
                if darkcurrent_level == 'auto':
                    darkcurrent_level = noise[1]

        # TODO: move this code inside the wfs
        # if wfs and background_level:
        #     # Compute sky background
        #     if background_level == 'auto':
        #         if background_noise:
        #             surf = (self.pixel_pupil * self.pixel_pitch) ** 2. / 4. * math.pi

        #             if sky_bg_norm:
        #                 if isinstance(wfs, ModulatedPyramid):
        #                     subaps = round(wfs.pup_diam ** 2. / 4. * math.pi)
        #                     tot_pix = subaps * 4.
        #                     fov = wfs.fov ** 2. / 4. * math.pi
        #                 elif isinstance(wfs, SH):
        #                     subaps = round(wfs.subap_on_diameter ** 2. / 4. * math.pi)
        #                     if subaps != 1 and subaps < 4.:
        #                         subaps = 4.
        #                     tot_pix = subaps * wfs.sensor_npx ** 2.
        #                     fov = wfs.sensor_fov ** 2
        #                 else:
        #                     raise ValueError(f'Unsupported WFS class: {type(wfs)}')
        #                 background_level = \
        #                     sky_bg_norm * dt * fov * surf / tot_pix * binning ** 2
        #             else:
        #                 raise ValueError('sky_bg_norm key must be set to update background_level key')
        #         else:
        #             background_level = 0

        # Adjust ADU / EM gain values
        
        # TODO old code
        if excess_noise:
            emccd_gain = 400         
            ADU_gain = 1 / 20

        # TODO new code to be tested
        # if emccd_gain is None:
        #     emccd_gain = 400 if excess_noise else 1

        # if ADU_gain is None:
        #     ADU_gain = 1 / 20 if excess_noise else 8

        if ADU_gain <= 1 and (not excess_noise or emccd_gain <= 1):
            print('ATTENTION: ADU gain is less than 1 and there is no electronic multiplication.')

        self._dt = self.seconds_to_t(dt)
        self._start_time = self.seconds_to_t(start_time)
        self._photon_noise = photon_noise
        self._readout_noise = readout_noise
        self._excess_noise = excess_noise
        self._darkcurrent_noise = darkcurrent_noise
        self._background_noise = background_noise
        self._cic_noise = cic_noise
        self._cte_noise = cte_noise

        self._binning = binning
        self._readout_level = readout_level
        self._darkcurrent_level = darkcurrent_level
        self._background_level = background_level
        self._cic_level = cic_level
        self._cte_mat = cte_mat if cte_mat is not None else self.xp.zeros((size[0], size[1], 2), dtype=self.dtype)
        self._qe = quantum_eff

        self._pixels = Pixels(size[0] // binning, size[1] // binning, target_device_idx=target_device_idx)
        s = self._pixels.size * self._binning
        self._integrated_i = Intensity(s[0], s[1], target_device_idx=target_device_idx, precision=precision)
        self._photon_seed = photon_seed
        self._readout_seed = readout_seed
        self._excess_seed = excess_seed

        self._excess_delta = excess_delta
        self._keep_ADU_bias = False
        self._doNotChangeI = False
        self._bg_remove_average = False
        self._do_not_remove_dark = False
        self._ADU_gain = ADU_gain
        self._ADU_bias = ADU_bias
        self._emccd_gain = emccd_gain
        self._bandw = bandw
        self._pixelGains = pixelGains
        self._notUniformQeMatrix = None
        self._one_over_notUniformQeMatrix = None
        self._notUniformQe = False
        self._normNotUniformQe = False
        self._poidev = None
        self._gaussian_noise = None
        self._photon_rng = self.xp.random.default_rng(self._photon_seed)
        self._readout_rng = self.xp.random.default_rng(self._readout_seed)

        self.inputs['in_i'] = InputValue(type=Intensity)
        self.outputs['out_pixels'] = self._pixels
        self.outputs['integrated_i'] = self._integrated_i


    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = self.seconds_to_t(value)

    @property
    def size(self):
        return self._pixels.size

    @property
    def bandw(self):
        return self._bandw
    
    @bandw.setter
    def bandw(self, bandw):
        self._bandw = bandw

    @property
    def binning(self):
        return self._binning

    @binning.setter
    def binning(self, value):
        s = self._pixels.size * self._binning
        if (s[0] % self._binning) != 0:
            raise ValueError('Warning: binning requested not applied')
        self._pixels.size = (s[0] // value, s[1] // value)
        self._binning = value

    def trigger_code(self):
        if self._start_time <= 0 or self.current_time >= self._start_time:
            in_i = self.local_inputs['in_i']
            if in_i.generation_time == self.current_time:
                if self._doNotChangeI:
                    self._integrated_i.sum(in_i, factor=self._loop_dt / self._dt)
                else:
                    self._integrated_i.sum(in_i, factor=self.t_to_seconds(self._loop_dt) * self._bandw)

            if (self.current_time + self._loop_dt - self._dt - self._start_time) % self._dt == 0:
                if self._doNotChangeI:
                    self._pixels.pixels = self._integrated_i.i.copy()
                else:
                    self.apply_binning()
                    self.apply_qe()
                    self.apply_noise()

                self._pixels.generation_time = self.current_time
                self._integrated_i.i *= 0.0

    def apply_noise(self):
        ccd_frame = self._pixels.pixels
        if self._background_noise or self._darkcurrent_noise:
            ccd_frame += (self._background_level + self._darkcurrent_level)

        if self._cte_noise:
            ccd_frame = self.xp.dot(self.xp.dot(self._cte_mat[:, :, 0], ccd_frame), self._cte_mat[:, :, 1])

        if self._cic_noise:
            ccd_frame += self.xp.random.binomial(1, self._cic_level, ccd_frame.shape)
        
        if self._photon_noise:
            ccd_frame = self._photon_rng.poisson(ccd_frame)

        if self._excess_noise:
            ccd_frame = 1.0 / self._excess_delta * gamma.rvs(self._excess_delta * ccd_frame, self._emccd_gain, random_state=self._excess_seed)

        if self._readout_noise:
            ron_vector = self._readout_rng.standard_normal(size=ccd_frame.size)
            ccd_frame += (ron_vector.reshape(ccd_frame.shape) * self._readout_level).astype(ccd_frame.dtype)

        if self._pixelGains is not None:
            ccd_frame *= self._pixelGains

        if self._notUniformQe and self._normNotUniformQe:
            if self._one_over_notUniformQeMatrix is None:
                self._one_over_notUniformQeMatrix = 1 / self._notUniformQeMatrix
            ccd_frame *= self._one_over_notUniformQeMatrix

        if self._photon_noise:
            ccd_frame = self.xp.round(ccd_frame * self._ADU_gain) + self._ADU_bias
            clamp_generic(0, 0, ccd_frame, xp=self.xp)

            if not self._keep_ADU_bias:
                ccd_frame -= self._ADU_bias

            ccd_frame = (ccd_frame / self._ADU_gain).astype(ccd_frame.dtype)
            if self._excess_noise:
                ccd_frame = (ccd_frame / self._emccd_gain).astype(ccd_frame.dtype)
            if self._darkcurrent_noise and not self._do_not_remove_dark:
                ccd_frame -= self._darkcurrent_level
            if self._bg_remove_average and not self._do_not_remove_dark:
                ccd_frame -= self._background_level

        self._pixels.pixels = ccd_frame

    def apply_binning(self):
        in_dim = self._integrated_i.i.shape
        out_dim = self._pixels.size

        if in_dim[0] != out_dim[0] * self._binning:
            ccd_frame = self.xp.zeros(out_dim * self._binning, dtype=self.dtype)
            ccd_frame[:in_dim[0], :in_dim[1]] = self._integrated_i.i
        else:
            ccd_frame = self._integrated_i.i.copy()

        if self._binning > 1:
            tot_ccd_frame = self.xp.sum(ccd_frame)
            ccd_frame = ccd_frame.reshape(out_dim[0], self._binning, out_dim[1], self._binning).sum(axis=(1, 3))
            ccd_frame = ccd_frame * self._binning ** 2 * (tot_ccd_frame / self.xp.sum(ccd_frame))
            self._pixels.pixels = ccd_frame
        else:
            self._pixels.pixels = ccd_frame

    def apply_qe(self):
        if self._qe != 1:
            self._pixels.multiply(self._qe)
        if self._notUniformQe:
            ccd_frame = self._pixels.pixels * self._notUniformQeMatrix
            self._pixels.pixels = ccd_frame

    def setQuadrantGains(self, quadrantsGains):
        dim2d = self._pixels.pixels.shape
        pixelGains = self.xp.zeros(dim2d, dtype=self.dtype)
        for i in range(2):
            for j in range(2):
                pixelGains[(dim2d[0] // self._binning // 2) * i:(dim2d[0] // self._binning // 2) * (i + 1),
                           (dim2d[1] // self._binning // 2) * j:(dim2d[1] // self._binning // 2) * (j + 1)] = quadrantsGains[j * 2 + i]
        self._pixelGains = pixelGains

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        in_i = self.inputs['in_i'].get(self.target_device_idx)
        if in_i is None:
            raise ValueError('Input intensity object has not been set')
        if self._dt <= 0:
            raise ValueError(f'dt (integration time) is {self._dt} and must be greater than zero')
        if self._dt % loop_dt != 0:
            raise ValueError(f'integration time dt={self._dt} must be a multiple of the basic simulation time_step={loop_dt}')
        if self._cte_noise and self._cte_mat is None:
            raise ValueError('CTE matrix must be set if CTE noise is activated')

