from collections import deque

from iteras.data_objects.sim_params import SimParams
from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.connections import InputValue
from iteras.base_value import BaseValue


class DelayLine(BaseProcessingObj):
    """
    Pure time delay of a value.

    Each value refreshed at the input at time t is copied to the output at
    time t + delay_s, and held until the next one. It is triggered at every
    simulation step, so the delay is exact at the resolution of the time step,
    also when the input is refreshed at a slower rate (e.g. once per sensor
    frame). The output generation time changes only when a new value is
    released, so downstream objects are triggered at the input rate.

    Typical use: between a filter (with ``delay: 0``) and a DM, to set the
    latency of a loop in seconds. The ``:-1`` connection of the DM output
    adds one time step: from the end of a sensor frame to the DM update the
    lag is delay_s + one time step.
    """
    def __init__(self,
                 simul_params: SimParams,
                 delay_s: float,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        Parameters
        ----------
        simul_params : SimParams
            Simulation parameters (time step).
        delay_s : float
            Delay [s]. It must be zero or a multiple of the simulation time step.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        loop_dt = self.seconds_to_t(simul_params.time_step)
        self.delay_t = self.seconds_to_t(delay_s)   # [time ticks]
        if self.delay_t < 0:
            raise ValueError(f'delay_s is {delay_s} and must not be negative')
        if self.delay_t % loop_dt != 0:
            raise ValueError(f'delay_s={delay_s} must be a multiple '
                             f'of the basic simulation time_step={simul_params.time_step}')

        # At most delay/time_step + 1 values are waiting at the same time
        # (one per step in (t - delay, t])
        self.n_slots = self.delay_t // loop_dt + 1
        self.buffer = None
        self.pending = deque()   # (input generation time, buffer slot), oldest first
        self.next_slot = 0

        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.output = BaseValue(target_device_idx=target_device_idx, precision=precision)
        self.outputs['out_value'] = self.output

    @classmethod
    def input_names(cls):
        return {'in_value': InputDesc(BaseValue, 'Value to delay')}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'Input value delayed by delay_s seconds')}

    def setup(self):
        super().setup()
        shape = self.xp.asarray(self.local_inputs['in_value'].value).shape
        self.buffer = self.xp.zeros((self.n_slots,) + shape, dtype=self.dtype)
        self.output.value = self.xp.zeros(shape, dtype=self.dtype)

    def checkInputTimes(self):
        # Triggered at every step: a value must be released also at steps
        # when the input is not refreshed
        self.get_all_inputs()
        return True

    def trigger_code(self):
        t = self.current_time
        in_value = self.local_inputs['in_value']
        if in_value.generation_time >= t:
            self.buffer[self.next_slot] = in_value.value
            self.pending.append((in_value.generation_time, self.next_slot))
            self.next_slot = (self.next_slot + 1) % self.n_slots

        # Release the newest value that is old enough
        slot = None
        while self.pending and self.pending[0][0] + self.delay_t <= t:
            slot = self.pending.popleft()[1]
        if slot is not None:
            self.output.value[...] = self.buffer[slot]
            self.output.generation_time = t
