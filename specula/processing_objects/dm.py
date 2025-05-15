from specula.base_value import BaseValue
from specula.connections import InputValue

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.layer import Layer
from specula.data_objects.pupilstop import Pupilstop
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.simul_params import SimulParams

class DM(BaseProcessingObj):
    def __init__(self,
                 simul_params: SimulParams,
                 height: float,          # TODO =0.0,
                 ifunc: IFunc=None,
                 m2c: M2C=None,
                 type_str: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 input_offset: int=0,
                 idx_modes = None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 sign: int=-1,
                 target_device_idx: int=None, 
                 precision: int=None
                 ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.simul_params = simul_params
        self.pixel_pitch = self.simul_params.pixel_pitch
        self.pixel_pupil = self.simul_params.pixel_pupil
        mask = None
        if pupilstop:
            mask = pupilstop.A

        if not ifunc:
            ifunc = IFunc(type_str=type_str, mask=mask, npixels=self.pixel_pupil,
                           obsratio=obsratio, diaratio=diaratio, nzern=nzern,
                           nmodes=nmodes, start_mode=start_mode, idx_modes=idx_modes,
                           target_device_idx=target_device_idx, precision=precision)
        self._ifunc = ifunc
        
        s = self._ifunc.mask_inf_func.shape
        nmodes_if = self._ifunc.size[0]
        
        self.if_commands = self.xp.zeros(nmodes_if, dtype=self.dtype)
        self.layer = Layer(s[0], s[1], self.pixel_pitch, height, target_device_idx=target_device_idx, precision=precision)
        self.layer.A = self._ifunc.mask_inf_func
        
        if m2c is not None:
            nmodes_m2c = m2c.m2c.shape[1]
            self.m2c_commands = self.xp.zeros(nmodes_m2c, dtype=self.dtype)
            self.m2c = m2c.m2c
        else:
            self.m2c = None
            self.m2c_commands = None

        self.input_offset = input_offset
        self.nmodes = nmodes

        # Default sign is -1 to take into account the reflection in the propagation
        self.sign = sign
        self.inputs['in_command'] = InputValue(type=BaseValue)
        self.outputs['out_layer'] = self.layer
        
    def trigger_code(self):
        input_commands = self.local_inputs['in_command'].value
        
        if self.nmodes is not None:
            input_commands = input_commands[self.input_offset: self.input_offset + self.nmodes]

        if self.m2c is not None:
            self.m2c_commands[:len(input_commands)] = input_commands
            cmd = self.m2c @ self.m2c_commands
        else:
            cmd = input_commands
            
        self.if_commands[:len(cmd)] = self.sign * cmd
        self.layer.phaseInNm[self._ifunc.idx_inf_func] = self.if_commands @ self._ifunc.influence_function
        self.layer.generation_time = self.current_time
    
    # Getters and Setters for the attributes
    @property
    def ifunc(self):
        return self._ifunc.influence_function

    @ifunc.setter
    def ifunc(self, value):
        self._ifunc.influence_function = value
    
       
