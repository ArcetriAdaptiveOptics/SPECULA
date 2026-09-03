
import numpy as np

from specula.processing_objects.iir_filter import IirFilter
from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.iir_filter_data import IirFilterData


class Integrator(IirFilter):
    def __init__(self,
                 int_gain: list,
                 ff: list=None,
                 n_modes: list=None, # list[int]
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
        # - If ff is provided:
        #     * scalar / length-1 -> broadcast to all modes
        #     * length == len(n_modes) -> expand like int_gain (numpy.repeat)
        #     * length == sum(n_modes) (or longer) -> per-mode vector (truncate if longer)
        # - Raises ValueError if the lengths do not match.
        if n_modes is not None:
            if isinstance(n_modes, int):
                n_modes = [n_modes]
            if len(n_modes) != len(int_gain):
                raise ValueError(f"When n_modes is a list, length of n_modes {len(n_modes)} must"
                                 f" match length of int_gain {len(int_gain)}")
            n_total = int(sum(n_modes))
            int_gain = [val for i, val in enumerate(int_gain) for _ in range(n_modes[i])]
            if ff is not None:
                ff_arr = np.asarray(ff, dtype=float).ravel()
                if ff_arr.size == 1:
                    ff = np.repeat(ff_arr, n_total)
                elif ff_arr.size == len(n_modes):
                    ff = np.repeat(ff_arr, n_modes)
                elif ff_arr.size >= n_total:
                    ff = ff_arr[:n_total]
                else:
                    raise ValueError(
                        f"ff length {ff_arr.size} incompatible with n_modes "
                        f"(blocks={len(n_modes)}, total={n_total})"
                    )

        iir_filter_data = IirFilterData.from_gain_and_ff(int_gain, ff=ff,
                                               target_device_idx=target_device_idx)

        # Initialize IirFilter object
        super().__init__(iir_filter_data, delay=delay, integration=integration,
                         target_device_idx=target_device_idx, precision=precision)

    @classmethod
    def input_names(cls):
        return super().input_names()

    @classmethod
    def output_names(cls):
        return super().output_names()
