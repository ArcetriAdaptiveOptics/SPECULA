
from specula.processing_objects.integrator import Integrator
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray


class DynamicIntegrator(Integrator):
    def __init__(self,
                 simul_params: SimulParams,
                 int_gain: float,
                 ff: list=None,
                 n_modes: int=None,
                 delay: float=0,
                 integration: bool=True,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        """
        Dynamic integrator processing object. Specialized IIR filter with integration and dynamic gain.
        """
        super().__init__(simul_params=simul_params,
                         int_gain=int_gain,
                         ff=ff,
                         n_modes=n_modes,
                         delay=delay,
                         integration=integration,
                         target_device_idx=target_device_idx,
                         precision=precision)

        self.inputs['reset'] = InputValue(type=BaseValue, optional=True)
        self.inputs['int_gain'] = InputValue(type=BaseValue, optional=True)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Reset internal state
        reset_input = self.local_inputs['reset']
        if reset_input is not None and reset_input.generation_time == self.current_time:
            self.reset_states()

        # Update internal IIR filter data if gain input changes
        gain_input = self.local_inputs['int_gain']
        if gain_input is not None and gain_input.generation_time == self.current_time:
            int_gain = cpuArray(gain_input.get_value())
            self.iir_filter_data.set_gain(int_gain)
