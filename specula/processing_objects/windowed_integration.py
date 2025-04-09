
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue


class WindowedIntegration(BaseProcessingObj):
    '''Simple windowed integration of a signal'''
    def __init__(self, 
                 n_elem: int, 
                 dt: float,
                 target_device_idx: int=None, 
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self._dt = self.seconds_to_t(dt)
        self._start_time = self.seconds_to_t(0)

        self.inputs['input'] = InputValue(type=BaseValue)

        self.n_elem = n_elem
        self.output = BaseValue(target_device_idx=target_device_idx, value=self.xp.zeros(self.n_elem, dtype=self.dtype))
        self.outputs['output'] = self.output
        self.output_value = self.xp.zeros(self.n_elem, dtype=self.dtype)
        
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = self.seconds_to_t(value)

    def trigger(self):
        if self._start_time <= 0 or self.current_time >= self._start_time:
            input = self.local_inputs['input']
            if input.generation_time == self.current_time:
                self.output.value *= 0.0
                self.output.generation_time = self.current_time
                self.output_value += input.value * self._loop_dt / self._dt

            if (self.current_time + self._loop_dt - self._dt - self._start_time) % self._dt == 0:
                self.output.value = self.output_value.copy()
                self.output.generation_time = self.current_time
                self.output_value *= 0.0

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        input = self.inputs['input'].get(self.target_device_idx)
        if input is None:
            raise ValueError('Input object has not been set')
        if self._dt <= 0:
            raise ValueError(f'dt (integration time) is {self._dt} and must be greater than zero')
        if self._dt % loop_dt != 0:
            raise ValueError(f'integration time dt={self._dt} must be a multiple of the basic simulation time_step={loop_dt}')

