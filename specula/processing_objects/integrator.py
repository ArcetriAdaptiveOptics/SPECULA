
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
                 offset: float=None,
                 og_shaper=None,         
                 target_device_idx: int=None,
                 precision: int=None
                ):
        """
        Integrator class for processing signals using an IIR filter.
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
        # Note: this behaviour (repeat each element of int_gain and ff by the corresponding number in n_modes)
        #       is the same as numpy.repeat
        if n_modes is not None:
            if isinstance(n_modes, int):
                n_modes = [n_modes]
            if len(n_modes) != len(int_gain):
                raise ValueError("When n_modes is a list, length of n_modes {len(n_modes)} must match length of int_gain {len(int_gain)}")
            int_gain = [val for i, val in enumerate(int_gain) for _ in range(n_modes[i])]
            if ff is not None:
                if isinstance(ff, int):
                    ff = [ff]
                if len(n_modes) != len(ff):
                    raise ValueError("When n_modes is a list, length of n_modes {len(n_modes)} must match length of ff {len(ff)}")
                ff = [val for i, val in enumerate(ff) for _ in range(n_modes[i])]

        iir_filter_data = IirFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IirFilter object
        super().__init__(simul_params, iir_filter_data, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
