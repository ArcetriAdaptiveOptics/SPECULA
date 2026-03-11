

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams


class Integrator(IirFilter):
    def __init__(self,
                 simul_params: SimulParams,
                 int_gain: list,
                 ff: list=None,
                 n_modes: int=None,
                 delay: float=0,
                 integration: bool=True,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        """
        Integrator processing object. Specialized IIR filter with integration.

        This class is a specialized version of the IirFilter class, designed to handle
        integration operations with specific gain and forgetting factor settings.
        """
        # Handle gain (int_gain) and forgetting factor (ff) setup based on n_modes:
        # - If n_modes is provided, it specifies how many modes (channels) to use.
        # - If n_modes is an integer, convert it to a list for uniform processing.
        # - If n_modes is a list, its length must match int_gain.
        # - Each int_gain[i] is expanded into a block of size n_modes[i].
        #   Example: n_modes=[2,3], int_gain=[0.5, 1.0] -> int_gain = [0.5, 0.5, 1.0, 1.0, 1.0]
        # - If ff is provided, it is expanded in the same way as int_gain.
        # - Raises ValueError if the lengths do not match.
        # Note: this behaviour (repeat each element of int_gain and ff by the corresponding
        #       number in n_modes) is the same as numpy.repeat
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        int_gain = self._repeat_for_nmodes(n_modes, int_gain, 'int_gain')
        ff = self._repeat_for_nmodes(n_modes, ff, 'ff')

        self.ff = ff
        self.n_modes = n_modes
        iir_filter_data = IirFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IirFilter object
        super().__init__(simul_params, iir_filter_data, delay=delay, integration=integration,
                         target_device_idx=target_device_idx, precision=precision)

        self.inputs['int_gain'] = InputValue(type=BaseValue, optional=True)

    def _repeat_for_nmodes(self, n_modes, array_to_repeat, array_name):
        if n_modes is None or array_to_repeat is None:
            return array_to_repeat
        if type(n_modes) is list and (
                type(array_to_repeat) is not list or
                len(n_modes) != len(array_to_repeat)):
            raise ValueError(f"When n_modes is a list, length of n_modes {len(n_modes)} must"
                             f" match length of {array_name} {len(array_to_repeat)}")
        return np.repeat(array_to_repeat, n_modes)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

        # Update internal IIR filter data if gain input changes
        gain_input = self.local_inputs['int_gain']
        if gain_input is not None and gain_input.generation_time == self.current_time:

            int_gain = cpuArray(gain_input.get_value())
            int_gain = self._repeat_for_nmodes(self.n_modes, [int_gain], 'int_gain')

            new_data = IirFilterData.from_gain_and_ff(int_gain, ff=self.ff,
                                                      target_device_idx=self.target_device_idx)
            self.iir_filter_data = new_data

