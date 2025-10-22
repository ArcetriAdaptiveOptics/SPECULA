
from specula.lib.calc_psf import calc_psf, calc_psf_geometry

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

import numpy as np


class PSF(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,    # TODO =500.0,
                 nd: float=None,
                 pixel_size_mas: float=None,
                 start_time: float=0.0,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if wavelengthInNm <= 0:
            raise ValueError('PSF wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm

        self.psf_pixel_size, self.nd = calc_psf_geometry(
                                            simul_params.pixel_pupil,
                                            simul_params.pixel_pitch,
                                            wavelengthInNm,
                                            nd,
                                            pixel_size_mas)

        self.start_time = start_time

        self.sr = BaseValue(target_device_idx=self.target_device_idx)
        self.int_sr = BaseValue(target_device_idx=self.target_device_idx)
        self.psf = BaseValue(target_device_idx=self.target_device_idx)
        self.int_psf = BaseValue(target_device_idx=self.target_device_idx)
        self.std_psf = BaseValue(target_device_idx=self.target_device_idx)
        self.ref = None
        self.count = 0
        self.first = True
        self._sum_psf_squared = None # For std dev calculation

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.outputs['out_sr'] = self.sr
        self.outputs['out_psf'] = self.psf
        self.outputs['out_int_sr'] = self.int_sr
        self.outputs['out_int_psf'] = self.int_psf
        self.outputs['out_std_psf'] = self.std_psf

    def setup(self):
        super().setup()
        in_ef = self.local_inputs['in_ef']
        s = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.int_psf.value = self.xp.zeros(s, dtype=self.dtype)
        self._sum_psf_squared = self.xp.zeros(s, dtype=self.dtype)
        self.std_psf.value = self.xp.zeros(s, dtype=self.dtype)
        self.int_sr.value = 0

        self.out_size = [int(np.around(dim * self.nd/2)*2) for dim in in_ef.size]
        self.ref = Intensity(self.out_size[0], self.out_size[1],
                             target_device_idx=self.target_device_idx)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        in_ef = self.local_inputs['in_ef']

        # First time, calculate reference PSF.
        if self.first:
            self.ref.i[:] = calc_psf(in_ef.A * 0.0,
                                     in_ef.A,
                                     imwidth=self.out_size[0],
                                     normalize=True,
                                     xp=self.xp,
                                     complex_dtype=self.complex_dtype)
            self.first = False

    def trigger_code(self):
        in_ef = self.local_inputs['in_ef']
        self.psf.value, self.total_psf = calc_psf(in_ef.phi_at_lambda(self.wavelengthInNm),
                                                  in_ef.A, imwidth=self.out_size[0], normalize=True,
                                                  xp=self.xp, complex_dtype=self.complex_dtype,
                                                  return_total=True)
        self.sr.value = self.psf.value[self.out_size[0] // 2, \
                                       self.out_size[1] // 2] / self.ref.i[self.out_size[0] // 2, \
                                       self.out_size[1] // 2]
        print('SR:', self.sr.value, flush=True)

    def post_trigger(self):
        super().post_trigger()
        if self.current_time_seconds >= self.start_time:
            self.count += 1
            self.int_sr.value += self.sr.value
            self.int_psf.value += self.psf.value
            self._sum_psf_squared += self.psf.value ** 2
        self.psf.generation_time = self.current_time
        self.sr.generation_time = self.current_time

    def finalize(self):
        if self.count > 0:
            self.int_psf.value /= self.count
            self.int_sr.value /= self.count
            self.std_psf.value = self.xp.sqrt(self._sum_psf_squared / self.count \
                                              - self.int_psf.value ** 2)

        self.int_psf.generation_time = self.current_time
        self.int_sr.generation_time = self.current_time
        self.std_psf.generation_time = self.current_time
