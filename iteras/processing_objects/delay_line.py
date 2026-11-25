from collections import deque

from iteras.data_objects.sim_params import SimParams
from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.connections import InputValue
from iteras.base_value import BaseValue


class DelayLine(BaseProcessingObj):
    """
    Pure time delay of a value.

    Each value refreshed at the input at time t becomes active at time
    t + delay_s, and stays active until the next one. The object is triggered
    at every simulation step, so the delay does not depend on the rate of the
    input (e.g. one value per sensor frame).

    When t + delay_s falls inside a time step, the output of that step is the
    mean over the step of the active value: r * previous + (1 - r) * new, with
    r the fraction of the step in which the previous value is still active;
    the new value alone is output from the next step. With an input refreshed
    at every step this is the same as the fractional *delay* of the filters
    (linear interpolation of two consecutive commands).

    The output generation time changes only when the output changes, so
    downstream objects are triggered at the input rate (twice per input value
    when delay_s is not a multiple of the time step).

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
            Delay [s], zero or positive.
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.loop_dt = self.seconds_to_t(simul_params.time_step)
        self.delay_t = self.seconds_to_t(delay_s)   # [time ticks]
        if self.delay_t < 0:
            raise ValueError(f'delay_s is {delay_s} and must not be negative')

        # Values still waiting after a step were produced in [t + time_step - delay, t]:
        # at most ceil(delay / time_step) of them, plus the one received at that step
        self.n_slots = -(-self.delay_t // self.loop_dt) + 1
        self.buffer = None
        self.active = None        # value active at the end of the last step
        self.pending = deque()    # (activation time, buffer slot), oldest first
        self.next_slot = 0
        self.mixed_output = False  # the last output was a mean of two values

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
        self.active = self.xp.zeros(shape, dtype=self.dtype)
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
            self.pending.append((in_value.generation_time + self.delay_t, self.next_slot))
            self.next_slot = (self.next_slot + 1) % self.n_slots

        # Values that become active during this step, i.e. in [t, t + time_step)
        changed = self.mixed_output
        self.mixed_output = False
        while self.pending and self.pending[0][0] < t + self.loop_dt:
            t_active, slot = self.pending.popleft()
            r = (t_active - t) / self.loop_dt   # fraction of the step with the previous value
            if r > 0:
                self.output.value[...] = r * self.active + (1 - r) * self.buffer[slot]
                self.mixed_output = True
            else:
                self.mixed_output = False
            self.active[...] = self.buffer[slot]
            changed = True
        if changed:
            if not self.mixed_output:
                self.output.value[...] = self.active
            self.output.generation_time = t
