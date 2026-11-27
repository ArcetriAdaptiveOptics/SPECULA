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
                 receiver_diameter: float = None,
                 target_device_idx: int = None,
                 precision: int = None
                 ):
        """
        Parameters
        ----------
        simul_params : SimulParams
        prop : AtmoPropagation
            The upwards propagation that produces the input field.
        receiver_diameter : float [m], optional
            Diameter of the receiver aperture in the receiver plane, centred on the beam centre. Only
            used with physical propagation. None (default): point receiver (intensity at the beam
            centre).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if prop.prop_sign == 1:
            raise ValueError('Power loss computation is only supported for upwards '
                             'propagation.')
        if receiver_diameter is not None and receiver_diameter <= 0:
            raise ValueError(f'receiver_diameter must be > 0, got {receiver_diameter}')

        self.first = True
        self.pixel_pupil = simul_params.pixel_pupil
        self.pixel_pitch = simul_params.pixel_pitch
        self.pad_size = int(self.pixel_pupil * prop.padding_factor)
        self.prop_obj = prop
        self.receiver_diameter = receiver_diameter
        self.buffer = self.xp.zeros([self.pad_size, self.pad_size], dtype=self.complex_dtype)
        self.psf_ref = 0.0
        self.power_ref = 0.0
        self.receiver_mask = self._make_receiver_mask()

        self.inputs['in_ef'] = InputValue(type=ElectricField)
        self.power_loss = BaseValue(target_device_idx=self.target_device_idx, precision=self.precision)
        self.outputs['out_power_loss'] = self.power_loss
        self.sr = BaseValue(target_device_idx=self.target_device_idx, precision=self.precision)
        self.outputs['out_sr'] = self.sr
        self.psf = BaseValue(target_device_idx=self.target_device_idx, precision=self.precision)
        self.psf.value = self.xp.zeros([self.pad_size, self.pad_size], dtype=self.dtype)
        self.outputs['out_psf'] = self.psf

    @classmethod
    def input_names(cls):
        return {'in_ef': InputDesc(ElectricField, 'Electric Field')}

    @classmethod
    def output_names(cls):
        return {'out_power_loss': OutputDesc(BaseValue, 'Output power loss in dB'),
                'out_sr': OutputDesc(BaseValue, 'Output Strehl ratio'),
                'out_psf': OutputDesc(BaseValue, 'PSF (geometric) or intensity in the receiver plane (Fresnel)')}

    def _make_receiver_mask(self):
        n = self.pixel_pupil
        c = (np.arange(n) - n / 2 + 0.5) * self.pixel_pitch
        r = np.hypot(c[:, None], c[None, :])
        point = r <= r.min() + 1e-9 * self.pixel_pitch           # central pixel(s): point receiver
        if self.receiver_diameter is None:
            mask = point
        else:
            mask = r <= self.receiver_diameter / 2
            if not mask.any() or mask.sum() < point.sum():
                mask = point
        return self.to_xp(mask, dtype=self.dtype)

    def _received_power(self, amplitude):
        return self.xp.sum(amplitude ** 2 * self.receiver_mask)

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

            if self.prop_obj.doFresnel:
                # Fresnel propagation without turbulent layers, to the receiver plane
                if self.prop_obj.propagators is not None:
                    for pi, propagator in enumerate(self.prop_obj.propagators):
                        if propagator is not None:
                            if not self.prop_obj.far_field_propagation[pi]:
                                angular_spectrum_propagation(ef_in, propagator, self.buffer, self.xp)
                            else:
                                fraunhofer_far_field_propagation(ef_in, propagator, self.buffer)
                b = self.prop_obj.beam_center
                tmp_ef = ef_in[s + b[0]:s + b[0] + self.pixel_pupil, s + b[1]:s + b[1] + self.pixel_pupil]
                self.power_ref = self._received_power(abs(tmp_ef)).copy()
            else:
                tmp_ef = ef_in[s:s + self.pixel_pupil, s:s + self.pixel_pupil]
                calc_psf(self.xp.angle(tmp_ef), abs(tmp_ef), xp=self.xp, imwidth=self.pad_size,
                         complex_dtype=self.complex_dtype, out=self.psf.value)
                self.psf_ref = self.psf.value[self.pad_size // 2, self.pad_size // 2].copy()

    def trigger_code(self):
        in_ef = self.local_inputs['in_ef']
        if self.prop_obj.doFresnel:
            # the input is the field at the receiver: no FFT, received intensity directly
            s = (self.pad_size - self.pixel_pupil) // 2
            self.psf.value[:] = 0
            self.psf.value[s:s + self.pixel_pupil, s:s + self.pixel_pupil] = in_ef.A ** 2
            ratio = self._received_power(in_ef.A) / self.power_ref
        else:
            calc_psf(in_ef.phi_at_lambda(self.prop_obj.wavelengthInNm), in_ef.A, imwidth=self.pad_size,
                     xp=self.xp, complex_dtype=self.complex_dtype, out=self.psf.value)
            ratio = self.psf.value[self.pad_size // 2, self.pad_size // 2] / self.psf_ref

        # In place (set_value()) after the first trigger: a new array at each step would
        # fail the check of the arrays of an accelerated reader
        self.sr.set_value(ratio)
        self.logger.info(f'SR at {int(self.prop_obj.wavelengthInNm)}nm : {self.sr.value}')
        self.power_loss.set_value(10 * self.xp.log10(ratio))
        self.logger.info(f'Power loss at {int(self.prop_obj.wavelengthInNm)}nm : {self.power_loss.value}')

    def post_trigger(self):
        self.power_loss.generation_time = self.current_time
        self.sr.generation_time = self.current_time
        self.psf.generation_time = self.current_time
