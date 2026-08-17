from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.calc_psf import calc_psf
from specula.processing_objects.atmo_propagation import AtmoPropagation, angular_spectrum_propagation, fraunhofer_far_field_propagation

import numpy as np


class PowerLoss(BaseProcessingObj):
    """
    Power Loss processing object. 
    Computes power loss in dB from the flux at the sensor and the receiver diameter.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 prop: AtmoPropagation,
                 target_device_idx: int = None,
                 precision: int = None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if prop.prop_sign == 1:
            raise ValueError('Power loss computation is only supported for upwards '
                             'propagation.')

        self.first = True
        self.total_power_ref = 0
        self.pixel_pupil = simul_params.pixel_pupil
        self.pixel_pitch = simul_params.pixel_pitch
        self.pad_size = int(self.pixel_pupil * prop.padding_factor)
        self.prop_obj = prop
        self.buffer = self.xp.zeros([self.pad_size, self.pad_size], dtype=self.complex_dtype)
        self.psf_ref = 0.0

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.power_loss = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_power_loss'] = self.power_loss
        self.sr = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_sr'] = self.sr
        self.psf = BaseValue(target_device_idx=self.target_device_idx)
        self.psf.value = self.xp.zeros([self.pad_size, self.pad_size])
        self.outputs['out_psf'] = self.psf

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Electric Field')}

    @classmethod
    def output_names(cls):
        return {'out_power_loss': OutputDesc(BaseValue, 'Output power loss in dB'),
                'out_sr': OutputDesc(BaseValue, 'Output Strehl ratio'),
                'out_psf': OutputDesc(BaseValue, 'PSF')}

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        if not self.prop_obj.common_layer_list:
            raise ValueError('At least one element in common_layer_list is required for'
                             'power loss calculation.')

        if self.first:
            self.first = False

            s = (self.pad_size - self.pixel_pupil) // 2
            ef_in = self.xp.zeros([self.pad_size, self.pad_size], dtype=self.complex_dtype)
            ef_in[s:s + self.pixel_pupil, s:s + self.pixel_pupil] = self.prop_obj.common_layer_list[0].A * self.xp.exp(
                1j * 0)

            # Fresnel propagation without turbulent layers
            if self.prop_obj.doFresnel:
                if self.prop_obj.propagators is not None:
                    for pi, propagator in enumerate(self.prop_obj.propagators):
                        if propagator is not None:
                            if not self.prop_obj.far_field_propagation[pi]:
                                angular_spectrum_propagation(ef_in, propagator, self.buffer, self.xp)
                            else:
                                fraunhofer_far_field_propagation(ef_in, propagator, self.buffer)

            tmp_ef = ef_in[s + self.prop_obj.beam_center[0]:s + self.prop_obj.beam_center[0] + self.pixel_pupil,
                     s + self.prop_obj.beam_center[1]:s + self.prop_obj.beam_center[1] + self.pixel_pupil]
            self.psf.value[:] = calc_psf(self.xp.angle(tmp_ef), abs(tmp_ef), xp=self.xp, imwidth=self.pad_size,
                                         complex_dtype=self.complex_dtype)
            self.psf_ref = self.psf.value[self.pad_size // 2, self.pad_size // 2].copy()

    def trigger_code(self):
        in_ef = self.local_inputs['in_ef']
        self.psf.value[:] = calc_psf(in_ef.phi_at_lambda(self.prop_obj.wavelengthInNm), in_ef.A, imwidth=self.pad_size,
                                     xp=self.xp, complex_dtype=self.complex_dtype)

        self.sr.value = self.psf.value[self.pad_size // 2, self.pad_size // 2] / self.psf_ref
        self.logger.info(f'SR at {int(self.prop_obj.wavelengthInNm)}nm : {self.sr.value}')
        self.power_loss.value = 10 * np.log10(self.psf.value[self.pad_size // 2, self.pad_size // 2] / self.psf_ref)
        self.logger.info(f'Power loss at {int(self.prop_obj.wavelengthInNm)}nm : {self.power_loss.value}')

    def post_trigger(self):
        self.power_loss.generation_time = self.current_time
        self.sr.generation_time = self.current_time
        self.psf.generation_time = self.current_time
