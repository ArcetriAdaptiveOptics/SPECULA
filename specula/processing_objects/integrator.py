
from specula.processing_objects.iir_filter import IirFilter
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams


class Integrator(IirFilter):
    def __init__(self,
                 simul_params: SimulParams,
                 int_gain: float,
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
        if n_modes is not None:
            if isinstance(n_modes, int):
                n_modes = [n_modes]
            if len(n_modes) != len(int_gain):
                raise ValueError(f"When n_modes is a list, length of n_modes {len(n_modes)} must"
                                 f" match length of int_gain {len(int_gain)}")
            int_gain = [val for i, val in enumerate(int_gain) for _ in range(n_modes[i])]
            if ff is not None:
                if isinstance(ff, int):
                    ff = [ff]
                if len(n_modes) != len(ff):
                    raise ValueError(f"When n_modes is a list, length of n_modes {len(n_modes)}"
                                     f" must match length of ff {len(ff)}")
                ff = [val for i, val in enumerate(ff) for _ in range(n_modes[i])]

        self.ff = ff
        self.n_modes = n_modes
        iir_filter_data = IirFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        self.inputs['int_gain'] = InputValue(type=BaseValue, optional=True)

        # Initialize IirFilter object
        super().__init__(simul_params, iir_filter_data, delay=delay, integration=integration,
                         target_device_idx=target_device_idx, precision=precision)

        def prepare_trigger(self, t):

            # Updated internal IIR filter data if gain input changes
            if self.inputs['int_gain'].generation_time == self.current_time:

                int_gain = self.inputs['int_gain'].get()
                if self.n_modes is not None:
                    if len(self.n_modes) != len(int_gain):
                        raise ValueError(f"Length of int_gain {len(int_gain)} does not match"
                                         f"length of n_modes {len(self.n_modes)}")

                    int_gain = [val for i, val in enumerate(int_gain) for _ in range(self.n_modes[i])]

                new_data = IirFilterData.from_gain_and_ff(int_gain, ff=self.ff,
                                                          target_device_idx=target_device_idx)
                self.iir_filter_data = iir_filter_data

            super().prepare_trigger(t)

