import numpy as np

from specula.data_objects.iir_filter_data import IirFilterData
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams

class IirFilter(BaseProcessingObj):
    '''Infinite Impulse Response filter based Time Control
    
    This class implements IIR filtering with optional integration control.
    
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
    '''
    def __init__(self,
                 simul_params: SimulParams,
                 iir_filter_data: IirFilterData,
                 delay: float=0,
                 integration: bool=True,
                 target_device_idx=None,
                 precision=None
                 ):

        self.time_step = simul_params.time_step

        self.verbose = True
        self.iir_filter_data = iir_filter_data

        super().__init__(target_device_idx=target_device_idx, precision=precision)        

        self.delay = delay if delay is not None else 0
        self._n = iir_filter_data.nfilter
        self._type = iir_filter_data.num.dtype
        self.set_state_buffer_length(int(np.ceil(self.delay)) + 1)

        self._gain_mod = None

        # Initialize state vectors
        self._ist = self.xp.zeros_like(iir_filter_data.num)
        self._ost = self.xp.zeros_like(iir_filter_data.den)

        # Create integration mask: if integration=False, zero out feedback terms
        self._den_mask = self.xp.ones_like(self.iir_filter_data.den)
        if integration is False:
            # Zero out all denominator coefficients except the last one (normalizer)
            self._den_mask[:, :-1] = 0

        self.out_comm = BaseValue(value=self.xp.zeros(self._n, dtype=self.dtype),
                                  target_device_idx=target_device_idx,
                                  precision=precision)
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['gain_mod'] = InputValue(type=BaseValue,optional=True)
        self.outputs['out_comm'] = self.out_comm

    def set_state_buffer_length(self, total_length):
        self._total_length = total_length
        if self._n is not None and self._type is not None:
            self.state = self.xp.zeros((self._n, self._total_length), dtype=self.dtype)

    # TODO not used
    @property
    def last_state(self):
        return self.state[:, 0]

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_comm = self.local_inputs['delta_comm'].value

        # Update the state
        if self.delay > 0:
            self.state[:, 1:self._total_length] = self.state[:, 0:self._total_length-1]

        # check if gain_mod is provided
        if self.local_inputs['gain_mod'] is not None:
            self._gain_mod = self.local_inputs['gain_mod'].value
        else:
            # Default gain_mod is an array of ones
            self._gain_mod = self.xp.ones_like(self.delta_comm, dtype=self.dtype)

        return

    def trigger_code(self):
        sden = self.iir_filter_data.den.shape
        snum = self.iir_filter_data.num.shape
        no = sden[1]
        ni = snum[1]

        # Delay the vectors
        self._ost[:, :-1] = self._ost[:, 1:]
        self._ost[:, -1] = 0  # Reset the last column

        self._ist[:, :-1] = self._ist[:, 1:]
        self._ist[:, -1] = 0  # Reset the last column

        # New input
        self._ist[:, ni - 1] = self.delta_comm

        # Precompute the reciprocal of the denominator
        factor = 1 / self.iir_filter_data.den[:, no - 1]

        # Compute new output
        num_contrib = self.xp.sum(self.iir_filter_data.num \
                                  * self._gain_mod[:, None] * self._ist, axis=1)
        den_contrib = self.xp.sum(self.iir_filter_data.den[:, :no - 1] \
                                  * self._den_mask[:, :no - 1] * self._ost[:, :no - 1], axis=1)
        self._ost[:, no - 1] = factor * (num_contrib - den_contrib)
        output = self._ost[:, no - 1]

        # Update the state
        self.state[:, 0] = output

    def post_trigger(self):
        super().post_trigger()

        # Calculate output from the state considering the delay
        remainder_delay = self.delay % 1
        if remainder_delay == 0:
            output = self.state[:, int(self.delay)]
        else:
            output = (remainder_delay * self.state[:, int(np.ceil(self.delay))] + \
                     (1 - remainder_delay) * self.state[:, int(np.ceil(self.delay))-1])

        self.out_comm.value = output
        self.out_comm.generation_time = self.current_time

    def reset_states(self):
        """Reset all internal states to zero."""
        self._ist[:] = 0
        self._ost[:] = 0
        self.state[:] = 0
