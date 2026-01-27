import numpy as np

from specula.data_objects.ssr_filter_data import SsrFilterData
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams

class SsrFilter(BaseProcessingObj):
    '''State Space Representation filter based Time Control
    
    Implements discrete-time state-space filtering:
    x[k+1] = A*x[k]  + B*u[k]
    y[k]   = C*x[k'] + D*u[k]
    
    where x[k'] is either x[k] or x[k+1] depending on output_uses_new_state parameter.
    
    All filters are handled simultaneously with single matrix operations.
    
    Parameters
    ----------
    simul_params : SimulParams
        Simulation parameters
    ssr_filter_data : SsrFilterData
        State-space matrices (A, B, C, D) in block-diagonal form
    delay : float, optional
        Output delay in frames (default: 0)
    output_uses_new_state : bool, optional
        If True, output equation uses updated state: y[k] = C*x[k+1] + D*u[k]
        If False, output equation uses current state: y[k] = C*x[k] + D*u[k]
        (default: True, which is standard for discrete integrators)
    target_device_idx : int, optional
        Target device index
    precision : int, optional
        Numerical precision
    '''
    def __init__(self,
                 simul_params: SimulParams,
                 ssr_filter_data: SsrFilterData,
                 delay: float=0,
                 output_uses_new_state: bool=True,
                 target_device_idx=None,
                 precision=None
                 ):

        self.time_step = simul_params.time_step
        self.verbose = True
        self.ssr_filter_data = ssr_filter_data
        self.output_uses_new_state = output_uses_new_state

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.delay = delay if delay is not None else 0
        self._nfilter = ssr_filter_data.nfilter

        # Set up delay buffer
        self.set_state_buffer_length(int(np.ceil(self.delay)) + 1)

        self.delta_comm = None
        self._gain_mod = None

        # Initialize single state vector for all filters (concatenated)
        self._x = self.xp.zeros(ssr_filter_data.total_states, dtype=self.dtype)

        # Output
        self.out_comm = BaseValue(value=self.xp.zeros(self._nfilter, dtype=self.dtype),
                                  target_device_idx=target_device_idx,
                                  precision=precision)

        # Inputs
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_comm'] = self.out_comm

    def set_state_buffer_length(self, total_length):
        """Set up output buffer for delay implementation."""
        self._total_length = total_length
        self.output_buffer = self.xp.zeros((self._nfilter, self._total_length), dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        delta_comm = self.local_inputs['delta_comm'].value

        # Convert to array and flatten to 1D
        delta_comm_array = self.xp.asarray(delta_comm, dtype=self.dtype)
        self.delta_comm = self.xp.atleast_1d(delta_comm_array).ravel()

        # Validate size matches number of filters
        if self.delta_comm.size != self._nfilter:
            if self.delta_comm.size == 1:
                # Broadcast scalar to all filters
                self.delta_comm = self.xp.full(self._nfilter, self.delta_comm[0], 
                                              dtype=self.dtype)
            else:
                raise ValueError(f"Input delta_comm has size {self.delta_comm.size} "
                               f"but filter expects {self._nfilter} inputs")

        # Update the delay buffer
        if self.delay > 0:
            self.output_buffer[:, 1:self._total_length] = \
                self.output_buffer[:, 0:self._total_length-1]

        # Check if gain_mod is provided
        if self.local_inputs['gain_mod'] is not None:
            gain_mod = self.local_inputs['gain_mod'].value
            gain_mod_array = self.xp.asarray(gain_mod, dtype=self.dtype)
            self._gain_mod = self.xp.atleast_1d(gain_mod_array).ravel()

            # Validate and broadcast if needed
            if self._gain_mod.size != self._nfilter:
                if self._gain_mod.size == 1:
                    self._gain_mod = self.xp.full(self._nfilter, self._gain_mod[0],
                                                 dtype=self.dtype)
                else:
                    raise ValueError(f"gain_mod size {self._gain_mod.size} doesn't match "
                                   f"nfilter {self._nfilter}")
        else:
            # Default gain_mod is an array of ones
            self._gain_mod = self.xp.ones(self._nfilter, dtype=self.dtype)

    def trigger_code(self):
        """Apply state-space update equations (vectorized for all filters)."""

        # Get block-diagonal matrices
        A = self.ssr_filter_data.A
        B = self.ssr_filter_data.B
        C = self.ssr_filter_data.C
        D = self.ssr_filter_data.D

        # Input vector (modulated) - shape: (nfilter,)
        # delta_comm and gain_mod are already guaranteed to be 1D from prepare_trigger
        u = self.delta_comm * self._gain_mod

        # State update: x[k+1] = A @ x[k] + B @ u
        # A: (total_states, total_states), x: (total_states,),
        # B: (total_states, nfilter), u: (nfilter,)
        x_new = A @ self._x + B @ u

        # Output: y[k] = C @ x[k'] + D @ u
        # C: (nfilter, total_states), x: (total_states,)
        # D: (nfilter, nfilter), u: (nfilter,)
        # Result: (nfilter,)
        x_for_output = x_new if self.output_uses_new_state else self._x
        y = C @ x_for_output + D @ u

        # Update state
        self._x = x_new

        # Store output - shape: (nfilter,)
        self.output_buffer[:, 0] = y

    def post_trigger(self):
        super().post_trigger()

        # Calculate output from the buffer considering the delay
        if self.delay == 0:
            output = self.output_buffer[:, 0]
        else:
            remainder_delay = self.delay % 1
            if remainder_delay == 0:
                output = self.output_buffer[:, int(self.delay)]
            else:
                output = (remainder_delay * self.output_buffer[:, int(np.ceil(self.delay))] +
                         (1 - remainder_delay) * self.output_buffer[:, int(np.ceil(self.delay))-1])

        self.out_comm.value = output
        self.out_comm.generation_time = self.current_time

    def reset_states(self):
        """Reset all internal states to zero."""
        self._x[:] = 0
        self.output_buffer[:] = 0
