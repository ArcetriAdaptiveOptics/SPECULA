import numpy as np

from specula.data_objects.ssr_filter_data import SsrFilterData
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams

class SsrFilter(BaseProcessingObj):
    '''State Space Representation filter based Time Control
    
    Implements discrete-time state-space filtering:
    x[k+1] = A*x[k] + B*u[k]
    y[k]   = C*x[k] + D*u[k]
    '''
    def __init__(self,
                 simul_params: SimulParams,
                 ssr_filter_data: SsrFilterData,
                 delay: float=0,
                 offset: float=None,
                 target_device_idx=None,
                 precision=None
                 ):

        self.time_step = simul_params.time_step
        self.verbose = True
        self.ssr_filter_data = ssr_filter_data

        if offset is not None:
            raise NotImplementedError('Offset not implemented yet')

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.delay = delay if delay is not None else 0
        self._nfilter = ssr_filter_data.nfilter

        # Get dimensions for each filter
        self._state_sizes = ssr_filter_data.get_state_size()
        self._output_sizes = ssr_filter_data.get_output_size()
        self._max_output_size = max(self._output_sizes)

        # Set up delay buffer
        self.set_state_buffer_length(int(np.ceil(self.delay)) + 1)

        # Initialize state vectors for each filter
        self._x = [self.xp.zeros(n, dtype=self.dtype) for n in self._state_sizes]

        # Output
        self.out_comm = BaseValue(value=self.xp.zeros(self._nfilter, dtype=self.dtype),
                                  target_device_idx=target_device_idx,
                                  precision=precision)

        # Inputs
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_comm'] = self.out_comm

        self._offset = None
        self._start_time = 0

    def set_state_buffer_length(self, total_length):
        """Set up output buffer for delay implementation."""
        self._total_length = total_length
        self.output_buffer = self.xp.zeros((self._nfilter, self._total_length), dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_comm = self.local_inputs['delta_comm'].value

        # Update the delay buffer
        if self.delay > 0:
            self.output_buffer[:, 1:self._total_length] = \
                self.output_buffer[:, 0:self._total_length-1]

        # Check if gain_mod is provided
        if self.local_inputs['gain_mod'] is not None:
            self._gain_mod = self.local_inputs['gain_mod'].value
        else:
            # Default gain_mod is an array of ones
            self._gain_mod = self.xp.ones_like(self.delta_comm, dtype=self.dtype)

    def trigger_code(self):
        """Apply state-space update equations for each filter."""

        for i in range(self._nfilter):
            # Get matrices for this filter
            A = self.ssr_filter_data.A[i]
            B = self.ssr_filter_data.B[i]
            C = self.ssr_filter_data.C[i]
            D = self.ssr_filter_data.D[i]

            # Current state and input
            x = self._x[i]
            u = self.delta_comm[i] * self._gain_mod[i]
            u = self.xp.atleast_1d(u)

            # State update: x[k+1] = A*x[k] + B*u[k]
            x_new = A @ x + B @ u

            # Update state
            self._x[i] = x_new

            # Output: y[k] = C*x[k] + D*u[k]
            y = C @ x_new + D @ u

            # Store output (extract scalar if single output)
            self.output_buffer[i, 0] = y.item() if y.size == 1 else y[0]

    def post_trigger(self):
        super().post_trigger()

        # Calculate output from the buffer considering the delay
        remainder_delay = self.delay % 1
        if remainder_delay == 0:
            output = self.output_buffer[:, int(self.delay)]
        else:
            output = (remainder_delay * self.output_buffer[:, int(np.ceil(self.delay))] + \
                     (1 - remainder_delay) * self.output_buffer[:, int(np.ceil(self.delay))-1])

        if self._offset is not None and self.xp.all(output == 0):
            output[:self._offset.shape[0]] += self._offset

        self.out_comm.value = output
        self.out_comm.generation_time = self.current_time

    def reset_states(self):
        """Reset all internal states to zero."""
        for i in range(self._nfilter):
            self._x[i][:] = 0
        self.output_buffer[:] = 0
