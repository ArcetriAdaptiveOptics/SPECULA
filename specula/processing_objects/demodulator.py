from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue
from specula.data_objects.simul_params import SimulParams


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
        self.output = BaseValue(target_device_idx=target_device_idx)
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

        for i, mode in enumerate(self.mode_numbers):
            values[i] = self._demodulate_signal(
                data_array[:, mode],
                self.carrier_frequencies[i]
            )

        # Clear history
        self.data_history = []
        self.time_history = []

        # Set output
        self.output.value = values
        self.output.generation_time = t

        if self.verbose:
            print(f"Demodulated value at t={self.t_to_seconds(t):.3f}s: {values}")

    def _demodulate_signal(self, signal_data, carrier_freq, cumulated=True):
        """
        Demodulate a single signal using the given carrier frequency.
        This implements the demodulation algorithm from demodulate_passata.pro
        """
        # Convert to numpy/cupy for processing
        data = self.to_xp(signal_data)

        # Parameters
        dt = self.t_to_seconds(self.loop_dt)
        sampling_freq = 1.0 / dt
        nt = len(data)
        t = self.xp.arange(nt, dtype=self.dtype) * dt

        # Handle single frequency input
        sinFreq = carrier_freq

        w = 2 * self.xp.pi * sinFreq

        # Calculate N4mean (number of samples for averaging at the end)
        periods = int(self.xp.floor(self.xp.max(t) * sinFreq))
        if periods > 0:
            testVect = (self.xp.arange(periods) + 1) * sampling_freq / sinFreq
            errors = self.xp.abs(testVect - self.xp.round(testVect))
            idx = self.xp.where(errors <= 1e-3)[0]
            if len(idx) > 0:
                N4mean = int(testVect[self.xp.max(idx)])
            else:
                min_error_idx = self.xp.argmin(errors)
                N4mean = int(testVect[min_error_idx])
        else:
            N4mean = max(1, nt // 4)  # Fallback

        # Linear detrend
        cur_data = data - self.xp.mean(data)
        tilt = (cur_data[-1] - cur_data[0]) / nt
        cur_data = cur_data - tilt * self.xp.arange(nt, dtype=self.dtype) - cur_data[0]

        # 1) Demodulate with reference carrier to find phase
        Qa_ref = self.xp.mean(cur_data * self.xp.sin(w * t))
        Pa_ref = self.xp.mean(cur_data * self.xp.cos(w * t))
        pphi0 = self.xp.arctan2(Qa_ref, Pa_ref)

        # Generate phased carriers
        dem_sin = self.xp.sin(w * t - pphi0)
        dem_cos = self.xp.cos(w * t - pphi0)

        if cumulated:
            # 2) Cumulated demodulation
            Qa = self.xp.zeros(nt, dtype=self.dtype)
            Pa = self.xp.zeros(nt, dtype=self.dtype)

            for j in range(2, nt):
                window_data = data[:j+1] - self.xp.mean(data[:j+1])
                window_tilt = (window_data[j] - window_data[0]) / j
                window_data = window_data - window_tilt * self.xp.arange(j+1, dtype=self.dtype) - window_data[0]

                Qa[j] = self.xp.sum(window_data * dem_sin[:j+1]) / (j + 1)
                Pa[j] = self.xp.sum(window_data * dem_cos[:j+1]) / (j + 1)

            data_dem_temp = 2.0 * self.xp.sqrt(Qa[2:]**2 + Pa[2:]**2)
            pphi_temp = self.xp.arctan2(Qa[2:], Pa[2:])

            start_idx = max(0, nt - 2 - N4mean)
            end_idx = nt - 2

            if end_idx > start_idx:
                value = self.xp.mean(data_dem_temp[start_idx:end_idx])
                pphi = self.xp.mean(pphi_temp[start_idx:end_idx])
            else:
                value = data_dem_temp[-1] if len(data_dem_temp) > 0 else 0.0
                pphi = pphi_temp[-1] if len(pphi_temp) > 0 else 0.0
        else:
            Qa = self.xp.mean(cur_data * dem_sin)
            Pa = self.xp.mean(cur_data * dem_cos)
            pphi = self.xp.arctan2(Qa, Pa)
            value = 2.0 * self.xp.sqrt(Qa**2 + Pa**2)

        pphi += pphi0

        # Convert back to target device if needed
        value = self.to_xp(value, dtype=self.dtype)

        if self.verbose:
            print(f"Demodulated value: {value}, Phase: {pphi}")
            print(f"Carrier frequency: {sinFreq}, Sampling frequency: {sampling_freq}, N4mean: {N4mean}")
            print(f"Data length: {len(data)}, Time steps: {nt}, dt: {dt:.3f}s")

        return value
