from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams
from specula.lib.demodulate_signal import demodulate_signal


class Demodulator(BaseProcessingObj):
    """
    Demodulator for modal amplitude estimation.
    Demodulates input signals using carrier frequencies and outputs scalar values
    representing modal amplitudes.
    """

    def __init__(self,
                 simul_params: SimulParams,
                 mode_numbers: list,
                 carrier_frequencies: list,
                 demod_dt: float,  # Demodulation time interval
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.mode_numbers = self.xp.array(mode_numbers, dtype=int)
        self.carrier_frequencies = self.xp.array(carrier_frequencies, dtype=self.dtype)
        self.demod_dt = self.seconds_to_t(demod_dt)

        # Data history storage
        self.data_history = []
        self.time_history = []

        self.loop_dt = self.seconds_to_t(simul_params.time_step)

        # Outputs
        self.output = BaseValue(target_device_idx=target_device_idx, precision=precision)
        if len(self.mode_numbers) == 1:
            self.output.value = self.dtype(0.0)
        else:
            self.output.value = self.xp.zeros(len(self.mode_numbers), dtype=self.dtype)

        # Inputs
        self.inputs['in_data'] = InputValue(type=BaseValue)

        # Outputs
        self.outputs['output'] = self.output

        self.verbose = False

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.input = self.local_inputs['in_data']

    def trigger_code(self):
        t = self.current_time

        # Extract data for the specified modes
        if self.input.value.ndim > 1:
            # Multi-dimensional data - extract modes
            mode_data = self.input.value[self.mode_numbers]
        else:
            # 1D data
            mode_data = self.input.value

        self.data_history.append(mode_data.copy())
        self.time_history.append(t)

        # Check if it's time to demodulate
        if (t + self.loop_dt - self.demod_dt) % self.demod_dt == 0:
            self._perform_demodulation(t)

    def _perform_demodulation(self, t):
        """
        Perform demodulation on accumulated data.
        """
        if len(self.data_history) == 0:
            return

        # Convert history to array
        data_array = self.xp.array(self.data_history)

        values = self.xp.zeros(len(self.mode_numbers), dtype=self.dtype)

        dt = self.t_to_seconds(self.loop_dt)
        sampling_freq = 1.0 / dt

        for i, mode in enumerate(self.mode_numbers):
            value, phase = demodulate_signal(
                signal_data=data_array[:, i],
                carrier_freq=float(self.carrier_frequencies[i]),
                sampling_freq=sampling_freq,
                cumulated=True,
                verbose=self.verbose,
                xp=self.xp,
                dtype=self.dtype
            )
            values[i] = value

        # Clear history
        self.data_history = []
        self.time_history = []

        # Set output
        self.output.value = values
        self.output.generation_time = t

        if self.verbose:
            print(f"Demodulated value at t={self.t_to_seconds(t):.3f}s: {values}")
