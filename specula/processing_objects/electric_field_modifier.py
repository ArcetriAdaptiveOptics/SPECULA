from specula.base_value import BaseValue
from specula.connections import InputValue

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

class ElectricFieldModifier(DM):
    """
    Applies a layer's phase and amplitude to an input electric field.
    
    This class takes an electric field as input and modifies it by:
    - Adding the layer's phase to the input field's phase
    - Multiplying the layer's amplitude with the input field's amplitude
    """
    def __init__(self,
                 simul_params: SimulParams,
                 height: float=0.0,
                 ifunc: IFunc=None,
                 m2c: M2C=None,
                 type_str: str=None,
                 nmodes: int=None,
                 nzern: int=None,
                 start_mode: int=None,
                 input_offset: int=0,
                 idx_modes = None,
                 npixels: int=None,
                 obsratio: float=None,
                 diaratio: float=None,
                 pupilstop: Pupilstop=None,
                 sign: int=1,
                 filter: bool=False,
                 target_device_idx: int=None, 
                 precision: int=None
                 ):
        super().__init__(
                 simul_params,
                 height,
                 ifunc,
                 m2c,
                 type_str,
                 nmodes,
                 nzern,
                 start_mode,
                 input_offset,
                 idx_modes,
                 npixels,
                 obsratio,
                 diaratio,
                 pupilstop,
                 sign,
                 target_device_idx, 
                 precision)

        self._out_ef = ElectricField(
            dimx=self.layer.A.shape[0],
            dimy=self.layer.A.shape[1],
            pixel_pitch=self.pixel_pitch,
            S0=self.S0,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.inputs['in_ef'] = InputValue(type=ElectricField,optional=True)
        self.inputs['in_command'] = InputValue(type=BaseValue,optional=True)
        self.outputs['out_ef'] = self._out_ef

        self._filter = filter
        if self._filter:
            nmodes = self._ifunc.size[0]
            if m2c is not None:
                nmodes = self.m2c.m2c.shape[1]
            self.phase2modes = ifunc.inverse()
            self._filter_modes = BaseValue(value=self.xp.zeros(nmodes, dtype=self.dtype), target_device_idx=target_device_idx)
            self.inputs['in_command'].set(self._filter_modes)

    def prepare_trigger(self):
        super().prepare_trigger()
        
        in_ef = self.local_inputs['in_ef'].value
        if in_ef.A.shape != self.layer.A.shape:
            raise ValueError(f"Input electric field shape {in_ef.A.shape} does not match DM layer shape {self.layer.A.shape}")

    def trigger_code(self):
        
        # Get the input electric field
        in_ef = self.local_inputs['in_ef'].value

        if self._filter:
            ph = in_ef.phaseInNm[self.phase2modes.idx_inf_func]
            m = -1 * self.xp.dot(ph, self.phase2modes.ifunc_inv)
            self._filter_modes.value = m
            self._filter_modes.generation_time = self.current_time

        super().trigger_code()

        # Combine the electric fields
        # Add phases
        self._out_ef.phaseInNm = in_ef.phaseInNm + self.layer.phaseInNm

        # Multiply amplitudes
        self._out_ef.A = in_ef.A * self.layer.A

        # Preserve S0 value from input
        self._out_ef.S0 = in_ef.S0
        
        # Set the generation time to the current time
        self._out_ef.generation_time = self.current_time
