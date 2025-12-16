
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.simul_params import SimulParams

import numpy as np

class PowerLoss(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 nd: int,
                 prop_distance: float,
                 receiver_diam: float,
                 target_device_idx: int = None,
                 precision: int = None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if wavelengthInNm <= 0:
            raise ValueError('Wavelength must be >0')
        self.wavelengthInNm = wavelengthInNm
        if nd <= 0:
            raise ValueError('PSF padding must be >0')
        if prop_distance <= 0:
            raise ValueError('Propagation distance must be >0')
        if receiver_diam <= 0:
            raise ValueError('Diameter of receiver aperture must be >0')

        self.receiver_diam = receiver_diam
        self.dx_sat_sq = self.wavelengthInNm*1e-9*prop_distance/(simul_params.pixel_pupil*nd*simul_params.pixel_pitch)
        self.inputs['se_sr'] = InputValue(type=BaseValue)
        self.power_loss = BaseValue(target_device_idx=self.target_device_idx)
        self.outputs['out_power_loss'] = self.power_loss

    def trigger_code(self):
        se_sr = self.local_inputs['se_sr']
        flux = se_sr.value/self.dx_sat_sq
        power = flux*self.receiver_diam
        self.power_loss.value = 10*np.log10(power)

    def post_trigger(self):
        self.power_loss.generation_time = self.current_time
