from specula.processing_objects.base_generator import BaseGenerator
import numpy as np


class ScheduleGenerator(BaseGenerator):
    """
    Generates scheduled values that change at specified times.
    """
    def __init__(self,
                 scheduled_values: list,
                 scheduled_times: list,
                 modes_per_group: list,
                 target_device_idx: int = None,
                 precision: int = None):

        if len(scheduled_values) != len(scheduled_times) + 1:
            raise ValueError('Length of scheduled_values must be length of scheduled_times + 1')

        # Expand scheduled_values according to modes_per_group
        if isinstance(modes_per_group, int):
            modes_per_group = [modes_per_group]

        expanded_values = []
        for value_set in scheduled_values:
            if len(modes_per_group) != len(value_set):
                raise ValueError(f"Length of modes_per_group {len(modes_per_group)} must match length of each value set {len(value_set)}")
            expanded_value = np.repeat(value_set, modes_per_group)
            expanded_values.append(expanded_value)

        output_size = len(expanded_values[0])

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.scheduled_values = self.to_xp(expanded_values, dtype=self.dtype)
        self.scheduled_times = self.to_xp(scheduled_times, dtype=self.dtype)

    def trigger_code(self):
        # Find the index of the current time in the time schedule
        time_idx = self.xp.searchsorted(
            self.scheduled_times,
            self.current_time_gpu,
            side='right'
        )

        # Clamp to valid bounds
        time_idx = self.xp.clip(time_idx, 0, self.scheduled_values.shape[0] - 1)

        self.output.value[:] = self.scheduled_values[time_idx, :]