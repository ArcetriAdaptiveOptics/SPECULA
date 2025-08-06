import numpy as np
from specula.processing_objects.base_generator import BaseGenerator


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
            expanded_value = [val for i, val in enumerate(value_set) for _ in range(modes_per_group[i])]
            expanded_values.append(expanded_value)

        output_size = len(expanded_values[0])

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.value_schedule = {
            'values': self.to_xp(expanded_values, dtype=self.dtype),
            'times': self.to_xp(scheduled_times, dtype=self.dtype)
        }

    def trigger_code(self):
        # Find the index of the current time in the time schedule
        time_idx = self.xp.searchsorted(
            self.value_schedule['times'],
            self.current_time_gpu,
            side='right'
        )

        # Clamp to valid bounds
        time_idx = self.xp.clip(time_idx, 0, self.value_schedule['values'].shape[0] - 1)

        self.output.value[:] = self.value_schedule['values'][time_idx, :]