from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula import cpuArray


class DynamicIirFilter(IirFilter):
    """ 
    Dynamic Infinite Impulse Response filter processing object.
    Same as standard IIR filter, with dynamic parameters.
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters containing time step information
    iir_filter_data : IirFilterData
        Filter coefficients (numerator and denominator)
    delay : float, optional
        Delay in frames to apply to the output (default: 0)
    integration : bool, optional
        If False, disables feedback terms (converts IIR to FIR).
        This is done by masking the denominator coefficients while
        preserving the normalizing factor. (default: True)
    target_device_idx : int, optional
        Target device for computation (-1 for CPU, >=0 for GPU)
    precision : int, optional
        Numerical precision (0 for double, 1 for single)
    
    Notes
    -----
    When integration=False, the filter becomes purely feedforward (FIR),
    removing all feedback/memory from previous outputs while maintaining
    the gain characteristics defined by the numerator coefficients.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 delay: float = 0,
                 integration: bool = True,
                 target_device_idx=None,
                 precision=None):

        super().__init__(
            simul_params=simul_params,
            iir_filter_data=iir_filter_data,
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

