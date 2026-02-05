from abc import abstractmethod

from specula import np
from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams


class BaseFilter(BaseProcessingObj):
    '''Base class for time-domain filters with delay support.
    
    Provides common functionality for:
    - Delay buffer management
    - Interpolation for fractional delays
    - Gain modulation
    - Synchronous (no-delay) outputs for POLC
    '''

    def __init__(self,
                 simul_params: SimulParams,
                 nfilter: int,
                 delay: float = 0,
                 target_device_idx=None,
                 precision=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.time_step = simul_params.time_step
        self.delay = delay if delay is not None else 0
        self._nfilter = nfilter

        # Set up delay buffer
        buffer_length = int(np.ceil(self.delay)) + 1
        self.output_buffer = self.xp.zeros((self._nfilter, buffer_length),
                                           dtype=self.dtype)

        self.delta_comm = None
        self._gain_mod = None

        # Outputs
        self.out_comm = BaseValue(
            value=self.xp.zeros(self._nfilter, dtype=self.dtype),
            target_device_idx=target_device_idx,
            precision=precision)

        self.out_comm_no_delay = BaseValue(
            value=self.xp.zeros(self._nfilter, dtype=self.dtype),
            target_device_idx=target_device_idx,
            precision=precision)

        # Inputs
        self.inputs['delta_comm'] = InputValue(type=BaseValue)
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)

        # Outputs
        self.outputs['out_comm'] = self.out_comm
        # This output provides the non-delayed command for POLC use
        # Please don't use it for drive DM / actuator directly
        self.outputs['out_comm_no_delay'] = self.out_comm_no_delay

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Process delta_comm input
        delta_comm = self.local_inputs['delta_comm'].value
        delta_comm_array = self.xp.asarray(delta_comm, dtype=self.dtype)
        self.delta_comm = self.xp.atleast_1d(delta_comm_array).ravel()

        # Validate/broadcast delta_comm size
        if self.delta_comm.size != self._nfilter:
            if self.delta_comm.size == 1:
                self.delta_comm = self.xp.full(self._nfilter, self.delta_comm[0],
                                              dtype=self.dtype)
            else:
                raise ValueError(f"Input delta_comm has size {self.delta_comm.size} "
                               f"but filter expects {self._nfilter} inputs")

        # Update delay buffer
        if self.delay > 0:
            self.output_buffer[:, 1:] = self.output_buffer[:, :-1]

        # Process gain_mod input
        if self.local_inputs['gain_mod'] is not None:
            gain_mod = self.local_inputs['gain_mod'].value
            gain_mod_array = self.xp.asarray(gain_mod, dtype=self.dtype)
            self._gain_mod = self.xp.atleast_1d(gain_mod_array).ravel()

            # Validate/broadcast gain_mod size
            if self._gain_mod.size != self._nfilter:
                if self._gain_mod.size == 1:
                    self._gain_mod = self.xp.full(self._nfilter, self._gain_mod[0],
                                                 dtype=self.dtype)
                else:
                    raise ValueError(f"gain_mod size {self._gain_mod.size} doesn't match "
                                   f"nfilter {self._nfilter}")
        else:
            self._gain_mod = self.xp.ones(self._nfilter, dtype=self.dtype)

    @abstractmethod
    def trigger_code(self):
        """Implement filter-specific computation.
        
        Must populate self.output_buffer[:, 0] with current output.
        """
        pass

    def post_trigger(self):
        super().post_trigger()

        # Calculate delayed output with interpolation
        if self.delay == 0:
            output = self.output_buffer[:, 0]
        else:
            remainder_delay = self.delay % 1
            delay_idx = int(np.ceil(self.delay))

            if remainder_delay == 0:
                output = self.output_buffer[:, delay_idx]
            else:
                output = (remainder_delay * self.output_buffer[:, delay_idx] +
                         (1 - remainder_delay) * self.output_buffer[:, delay_idx - 1])

        self.out_comm.value = output
        self.out_comm.generation_time = self.current_time

        # No-delay output (for POLC)
        self.out_comm_no_delay.value = self.output_buffer[:, 0]
        self.out_comm_no_delay.generation_time = self.current_time

    @abstractmethod
    def reset_states(self):
        """Reset filter internal states."""
        self.output_buffer[:] = 0
