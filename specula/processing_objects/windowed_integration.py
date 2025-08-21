
from specula.data_objects.simul_params import SimulParams
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue


class WindowedIntegration(BaseProcessingObj):
    '''Simple windowed integration of a signal'''
    def __init__(self,
                 simul_params: SimulParams,
                 n_elem: int,
                 dt: float,
                 start_time: float=0,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.loop_dt = self.seconds_to_t(simul_params.time_step)

        self.dt = self.seconds_to_t(dt)
        self.start_time = self.seconds_to_t(start_time)

        if self.dt <= 0:
            raise ValueError(f'dt (integration time) is {dt} and must be greater than zero')
        if self.dt % self.loop_dt != 0:
            raise ValueError(f'integration time dt={dt} must be a multiple of the basic simulation time_step={simul_params.time_step}')

        self.inputs['input'] = InputValue(type=BaseValue)

        self.output = BaseValue(value=self.xp.zeros(n_elem, dtype=self.dtype), target_device_idx=target_device_idx)
        self.outputs['output'] = self.output
        self.integrated_value = self.xp.zeros(n_elem, dtype=self.dtype)

    def trigger_code(self):
        if self.start_time <= 0 or self.current_time >= self.start_time:
            input = self.local_inputs['input']
            self.output.value *= 0.0
            self.integrated_value += input.value * self.loop_dt / self.dt

            if (self.current_time + self.loop_dt - self.dt - self.start_time) % self.dt == 0:
                self.output.value[:] = self.integrated_value
                self.integrated_value *= 0.0
                
        self.output.generation_time = self.current_time
