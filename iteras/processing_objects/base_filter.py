from abc import abstractmethod

from iteras import np
from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.connections import InputValue
from iteras.base_value import BaseValue


class BaseFilter(BaseProcessingObj):
    """
    Base filter processing object.
    Base class for time-domain filters with delay support.
    """
    def __init__(self,
                 nfilter: int,
                 delay: float = 0,
                 target_device_idx=None,
                 precision=None):
        """
        Note
        ----
        Provides common functionality for:
            - Delay buffer management
            - Interpolation for fractional delays
            - Gain modulation
            - Synchronous (no-delay) outputs for POLC
        """

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.delay = delay if delay is not None else 0
        self._nfilter = nfilter

        # Set up delay buffer
        buffer_length = int(np.ceil(self.delay)) + 1
        self.output_buffer = self.xp.zeros((self._nfilter, buffer_length),
                                           dtype=self.dtype)

        self.delta_comm = None
        self._gain_mod = None
        self._gain_mod_from_input = False

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
        # This output provides the non-delayed command.
        # POLC requires the non-delayed command.
        # Actuators are usually driven with the delayed output.
        self.outputs['out_comm_no_delay'] = self.out_comm_no_delay

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Process delta_comm input. The inputs are copied into the same arrays at
        # each step: the CUDA graph of a accelerated object keeps using them
        delta_comm = self.local_inputs['delta_comm'].value
        delta_comm_array = self.xp.atleast_1d(self.xp.asarray(delta_comm, dtype=self.dtype)).ravel()

        # Validate/broadcast delta_comm size
        if delta_comm_array.size != self._nfilter and delta_comm_array.size != 1:
            raise ValueError(f"Input delta_comm has size {delta_comm_array.size} "
                             f"but filter expects {self._nfilter} inputs")
        if self.delta_comm is None:
            self.delta_comm = self.xp.empty(self._nfilter, dtype=self.dtype)
        self.delta_comm[:] = delta_comm_array

        # Update delay buffer
        if self.delay > 0:
            self.output_buffer[:, 1:] = self.output_buffer[:, :-1]

        # Process gain_mod input
        if self._gain_mod is None:
            self._gain_mod = self.xp.ones(self._nfilter, dtype=self.dtype)
        if self.local_inputs['gain_mod'] is not None:
            gain_mod = self.local_inputs['gain_mod'].value
            gain_mod_array = self.xp.atleast_1d(self.xp.asarray(gain_mod, dtype=self.dtype)).ravel()

            # Validate/broadcast gain_mod size
            if gain_mod_array.size != self._nfilter and gain_mod_array.size != 1:
                raise ValueError(f"gain_mod size {gain_mod_array.size} doesn't match "
                                 f"nfilter {self._nfilter}")
            self._gain_mod[:] = gain_mod_array
            self._gain_mod_from_input = True
        elif self._gain_mod_from_input:
            self._gain_mod[:] = 1
            self._gain_mod_from_input = False

    @classmethod
    def input_names(cls):
        return {'delta_comm': InputDesc(BaseValue, 'Input delta command vector'),
                'gain_mod': InputDesc(BaseValue, 'Optional gain modulation vector (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_comm': OutputDesc(BaseValue, 'Output command vector with delay applied'),
                'out_comm_no_delay': OutputDesc(BaseValue, 'Output command vector without delay (for POLC)')}

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
