
import numpy as np

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.time_history import TimeHistory
from specula.lib.modal_pushpull_signal import modal_pushpull_signal


# TODO
class Vibrations():
    pass

def is_scalar(x):
    return np.isscalar(x) or (hasattr(x, 'shape') and x.shape == ())

class FuncGenerator(BaseProcessingObj):
    """
    Function generator for creating various signal types.
    
    Parameters
    ----------
    func_type : str, default='SIN'
        Type of function to generate. Options include:
        'SIN', 'SQUARE_WAVE', 'LINEAR', 'RANDOM', 'RANDOM_UNIFORM',
        'PUSH', 'PUSHPULL', 'TIME_HIST', 'VALUE_SCHEDULE'
    
    nmodes : int, optional
        Number of modes for PUSHPULL, PUSHPULLREPEAT, VIB_HIST or VIB_PSD types.
    time_hist : TimeHistory, optional
        Time history data for VIB_HIST or TIME_HIST types.
    psd : array, optional
        Power spectral density for VIB_PSD type.
    fr_psd : array, optional
        Frequency array corresponding to psd for VIB_PSD type.
    continuous_psd : array, optional
        Continuous PSD for VIB_PSD type.
    constant : list, optional
        Constant value to add to the output signal. Default is 0.0.
    amp : list, optional
        Amplitude of the signal. For SIN, SQUARE_WAVE, LINEAR, RANDOM types.
        If not set, defaults to 0.0.
    freq : list, optional
        Frequency of the signal. For SIN, SQUARE_WAVE, LINEAR types.
        If not set, defaults to 0.0.
    offset : list, optional
        Offset to add to the signal. For SIN, SQUARE_WAVE, LINEAR types.
        If not set, defaults to 0.0.
    vect_amplitude : list, optional
        Vector amplitude for PUSHPULL type. If not set, defaults to 0.0.
    nsamples : int, default=1
        Number of samples to generate for PUSHPULL type. Must be 1 for other types.
    seed : int, optional
        Random seed for generating random signals. If 'auto', a random seed is generated.
    ncycles : int, default=1
        Number of cycles for PUSHPULL type. If PUSHPULLREPEAT, cycles are repeated.
    vsize : int, default=1
        Size of the output vector. If nmodes is set, this is multiplied by nmodes.
    scheduled_values : list, optional
        For VALUE_SCHEDULE type only. List of value arrays for each time interval.
        Each element should be a list/array with length matching modes_per_group.
        Example: [[0.1, 0.0], [0.5, 1.0], [0.2, 0.3]]
    time_intervals : list, optional
        For VALUE_SCHEDULE type only. List of time limits (in seconds) for each interval.
        Length must match scheduled_values.
        Example: [0.1, 0.2, 0.5] means:
        - First values used for t < 0.1s
        - Second values used for 0.1s ≤ t < 0.2s  
        - Third values used for t ≥ 0.2s
    modes_per_group : list or int, optional
        For VALUE_SCHEDULE type only. Number of modes for each group of values.
        If int, converted to single-element list.
        Each scheduled_values[i][j] is replicated modes_per_group[j] times.
        Example: modes_per_group=[2, 3] with scheduled_values=[[0.1, 0.5]]
        produces output [0.1, 0.1, 0.5, 0.5, 0.5]
    target_device_idx : int, optional
        Index of the target device for this processing object.
    precision : int, optional
        Precision of the output values. If not set, defaults to float32.
    
    Examples
    --------
    Basic VALUE_SCHEDULE usage:

    >>> # Create a gain schedule that changes over time
    >>> func_gen = FuncGenerator(
    ...     func_type='VALUE_SCHEDULE',
    ...     scheduled_values=[
    ...         [0.1, 0.0],    # Low gain initially
    ...         [0.5, 0.2],    # Medium gain 
    ...         [1.0, 0.8]     # High gain
    ...     ],
    ...     time_intervals=[0.1, 0.3],  # Only 2 values for 3 sets of scheduled_values
    ...     modes_per_group=[2, 3]    # 2 modes for first value, 3 for second
    ... )
    >>> # Output will be:
    >>> # [0.1, 0.1, 0.0, 0.0, 0.0] for t < 0.1s
    >>> # [0.5, 0.5, 0.2, 0.2, 0.2] for 0.1s ≤ t < 0.3s  
    >>> # [1.0, 1.0, 0.8, 0.8, 0.8] for t ≥ 0.3s
    
    Notes
    -----
    - For VALUE_SCHEDULE, all three parameters (scheduled_values, time_intervals, modes_per_group) are mandatory
    - The expansion logic follows numpy.repeat behavior: each value is repeated modes_per_group[i] times
    - Time intervals define upper bounds: values change when current_time >= time_intervals[i]
    - After the last time interval, the last set of values is maintained indefinitely
    """
    def __init__(self,
                 func_type='SIN',
                 nmodes: int=None,
                 time_hist: TimeHistory=None,
                 psd=None,
                 fr_psd=None,
                 continuous_psd=None,
                 constant: list=None,
                 amp: list=None,
                 freq: list=None,
                 offset: list=None,
                 vect_amplitude: list=None,
                 nsamples: int=1,
                 seed: int=None,
                 ncycles: int=1,
                 vsize: int=1,
                 scheduled_values: list=None,
                 time_intervals: list=None,
                 modes_per_group: list=None,
                 target_device_idx: int=None,
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if nmodes is not None and vsize>1:
            raise ValueError('NMODES and VSIZE cannot be used together. Use NMODES only for PUSHPULL, PUSHPULLREPEAT, VIB_HIST or VIB_PSD types')

        self.type = func_type.upper()
        if self.type == 'PUSHPULLREPEAT':
            repeat_ncycles = True
            self.type = 'PUSHPULL'
        else:
            repeat_ncycles = False

        if nsamples != 1 and self.type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL or PUSHPULLREPEAT types')

        if str(seed).strip() == 'auto':
            self.seed = self.xp.around(self.xp.random.random() * 1e4)
        elif seed is not None:
            self.seed = self.to_xp(seed, dtype=self.dtype)
        else:
            self.seed = 0

        self.constant = self.to_xp(constant, dtype=self.dtype) if constant is not None else 0.0
        self.amp = self.to_xp(amp, dtype=self.dtype) if amp is not None else 0.0
        self.freq = self.to_xp(freq, dtype=self.dtype) if freq is not None else 0.0
        self.offset = self.to_xp(offset, dtype=self.dtype) if offset is not None else 0.0
        self.vect_amplitude = self.to_xp(vect_amplitude, dtype=self.dtype) if vect_amplitude is not None else 0.0

        if self.type in ['SIN', 'SQUARE_WAVE', 'LINEAR', 'RANDOM', 'RANDOM_UNIFORM']:
            # Check if the parameters are scalars or arrays and have coherent sizes
            params = [self.amp, self.freq, self.offset, self.constant]
            param_names = ['amp', 'freq', 'offset', 'constant']
            vector_lengths = [p.shape[0] for p in params if not is_scalar(p)]

            if len(vector_lengths) > 0:
                unique_lengths = set(vector_lengths)
                if len(unique_lengths) > 1:
                    # Find the names of the parameters with different lengths
                    details = [f"{name}={p.shape[0]}" for p, name in zip(params, param_names) if not is_scalar(p)]
                    raise ValueError(
                        f"Shape mismatch: parameter lengths are {details} (must all be equal if not scalar)"
                    )
                output_size = unique_lengths.pop()
            else:
                output_size = vsize if nmodes is None else vsize * nmodes
        elif self.type in ['PUSH', 'PUSHPULL', 'TIME_HIST']:
            if time_hist is not None:
                output_size = self.to_xp(time_hist.time_history).shape[1]
            elif nmodes is not None:
                output_size = nmodes
        elif self.type in ['VALUE_SCHEDULE']:
            if scheduled_values is None or time_intervals is None or modes_per_group is None:
                raise ValueError('SCHEDULED_VALUES, TIME_INTERVALS and MODES_PER_GROUP keywords are mandatory for type VALUE_SCHEDULE')
            output_size = np.sum(modes_per_group)
        else:
            output_size = vsize if nmodes is None else vsize * nmodes

        self.output = BaseValue(target_device_idx=target_device_idx, value=self.xp.zeros(output_size, dtype=self.dtype))
        self.vib = None

        if seed is not None:
            self.seed = seed

        # Initialize attributes based on the type
        if self.type == 'SIN':
            pass

        elif self.type == 'SQUARE_WAVE':
            pass

        elif self.type == 'LINEAR':
            self.slope = 0.0

        elif self.type == 'RANDOM' or self.type == 'RANDOM_UNIFORM':
            pass

        elif self.type == 'VIB_HIST':
            raise NotImplementedError('VIB_HIST type is not implemented')
        
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_HIST')
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type VIB_HIST')
            self.vib = Vibrations(nmodes, time_hist=time_hist)

        elif self.type == 'VIB_PSD':
            raise NotImplementedError('VIB_PSD type is not implemented')

            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_PSD')
            if psd is None and continuous_psd is None:
                raise ValueError('PSD or CONTINUOUS_PSD keyword is mandatory for type VIB_PSD')
            if fr_psd is None:
                raise ValueError('FR_PSD keyword is mandatory for type VIB_PSD')
            self.vib = Vibrations(nmodes, psd=psd, freq=fr_psd, continuous_psd=continuous_psd, seed=seed)

        elif self.type == 'PUSH':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSH')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, only_push=True, ncycles=ncycles)

        elif self.type == 'PUSHPULL':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSHPULL')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSHPULL')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles, repeat_ncycles=repeat_ncycles, nsamples=nsamples)

        elif self.type == 'TIME_HIST':
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type TIME_HIST')
            self.time_hist = self.to_xp(time_hist.time_history)

        elif self.type == 'VALUE_SCHEDULE':
            if scheduled_values is None or time_intervals is None or modes_per_group is None:
                raise ValueError('SCHEDULED_VALUES, TIME_INTERVALS and MODES_PER_GROUP keywords are mandatory for type VALUE_SCHEDULE')

            if len(scheduled_values) != len(time_intervals) + 1:
                raise ValueError('LENGTH of SCHEDULED_VALUES must be LENGTH of TIME_INTERVALS + 1')

            # Expand scheduled_values according to modes_per_group
            if isinstance(modes_per_group, int):
                modes_per_group = [modes_per_group]

            expanded_values = []
            for value_set in scheduled_values:
                if len(modes_per_group) != len(value_set):
                    raise ValueError(f"Length of modes_per_group {len(modes_per_group)} must match length of each value set {len(value_set)}")
                expanded_value = [val for i, val in enumerate(value_set) for _ in range(modes_per_group[i])]
                expanded_values.append(expanded_value)

            self.value_schedule = {
                'values': self.to_xp(expanded_values, dtype=self.dtype),
                'times': self.to_xp(time_intervals, dtype=self.dtype)
            }

        else:
            raise ValueError(f'Unknown function type: {self.type}')

        self.nmodes = nmodes
        self.outputs['output'] = self.output
        self.iter_counter = 0
        self.current_time_gpu = self.xp.zeros(1, dtype=self.dtype)
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.current_time_gpu[:] = self.current_time_seconds

    def trigger_code(self):

        if self.type == 'SIN':
            phase = self.freq*2 * self.xp.pi * self.current_time_gpu + self.offset
            self.output.value[:] = (self.amp * self.xp.sin(phase, dtype=self.dtype) + self.constant) * self.vsize_array

        elif self.type == 'SQUARE_WAVE':
            phase = self.freq*2 * self.xp.pi*self.current_time_gpu + self.offset
            self.output.value[:] = (self.amp * self.xp.sign(self.xp.sin(phase, dtype=self.dtype)) + self.constant) * self.vsize_array

        elif self.type == 'LINEAR':
            self.output.value[:] = (self.slope * self.current_time_gpu + self.constant) * self.vsize_array

        elif self.type == 'RANDOM':
            self.output.value[:] = (self.xp.random.normal(size=len(self.amp)) * self.amp + self.constant) * self.vsize_array

        elif self.type == 'RANDOM_UNIFORM':
            lowv = self.constant - self.amp/2
            highv = self.constant + self.amp/2
            self.output.value[:] = (self.xp.random.uniform(low=lowv, high=highv)) * self.vsize_array

        elif self.type in ['VIB_HIST', 'VIB_PSD', 'PUSH', 'PUSHPULL', 'TIME_HIST']:
            self.output.value[:] = self.get_time_hist_at_current_time() * self.vsize_array

        elif self.type == 'VALUE_SCHEDULE':
            # Find the index of the current time in the time schedule using searchsorted
            time_idx = self.xp.searchsorted(self.value_schedule['times'], self.current_time_gpu, side='right')

            # Clamp to valid bounds
            time_idx = self.xp.clip(time_idx, 0, self.value_schedule['values'].shape[0] - 1)

            self.output.value[:] = self.value_schedule['values'][time_idx, :]

        else:
            raise ValueError(f'Unknown function generator type: {self.type}')

    def post_trigger(self):

        self.output.generation_time = self.current_time
        self.iter_counter += 1

    def get_time_hist_at_current_time(self):
        return self.to_xp(self.time_hist[self.iter_counter])

    def setup(self):
        super().setup()

#       TODO
#       if self.vib:
#           self.vib.set_niters(self.loop_niters + 1)
#           self.vib.set_samp_freq(1.0 / self.t_to_seconds(self.loop_dt))
#           self.vib.compute()
#           self.time_hist = self.vib.get_time_hist()

        if self.type in ['SIN', 'LINEAR', 'RANDOM']:
            self.build_stream()

