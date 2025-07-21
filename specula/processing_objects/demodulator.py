import numpy as np
from scipy import signal

from specula.base_processing_obj import BaseProcessingObj
from specula.connections import InputValue
from specula.base_value import BaseValue


class Demodulator(BaseProcessingObj):
    """
    Demodulator for modal amplitude estimation.
    Demodulates input signals using carrier frequencies and outputs scalar values
    representing modal amplitudes.
    """
    
    def __init__(self,
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
        
        # Outputs
        self.output = BaseValue(target_device_idx=target_device_idx)
        
        # Inputs
        self.inputs['in_data'] = InputValue(type=BaseValue)
        
        # Outputs
        self.outputs['output'] = self.output
        
        self.verbose = False

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.current_data = self.local_inputs['in_data']

    def trigger_code(self):
        t = self.current_time
        
        # Store data if input is ready
        if self.current_data.generation_time == t:
            # Extract data for the specified modes
            if self.current_data.value.ndim > 1:
                # Multi-dimensional data - extract modes
                mode_data = self.current_data.value[self.mode_numbers]
            else:
                # 1D data
                mode_data = self.current_data.value
            
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
        
        if data_array.ndim == 2:
            # Multiple modes
            n_time, n_modes = data_array.shape
            values = self.xp.zeros(n_modes, dtype=self.dtype)
            
            for i in range(n_modes):
                values[i] = self._demodulate_signal(
                    data_array[:, i], 
                    self.carrier_frequencies[i]
                )
        else:
            # Single mode
            values = self._demodulate_signal(
                data_array, 
                self.carrier_frequencies[0]
            )
        
        # Clear history
        self.data_history = []
        self.time_history = []
        
        # Set output
        self.output.value = values
        self.output.generation_time = t
        
        if self.verbose:
            print(f"Demodulated value at t={self.t_to_seconds(t):.3f}s: {values}")

    def _demodulate_signal(self, signal_data, carrier_freq):
        """
        Demodulate a single signal using the given carrier frequency.
        This implements the cumulated demodulation as in the IDL code.
        """
        n_samples = len(signal_data)
        fs = 1.0 / self.t_to_seconds(self.loop_dt)  # Sampling frequency
        
        # Generate time vector
        t_vec = self.xp.arange(n_samples, dtype=self.dtype) / fs
        
        # Generate carrier signals (in-phase and quadrature)
        carrier_i = self.xp.cos(2 * np.pi * carrier_freq * t_vec)
        carrier_q = self.xp.sin(2 * np.pi * carrier_freq * t_vec)
        
        # Multiply signal with carriers
        i_component = signal_data * carrier_i
        q_component = signal_data * carrier_q
        
        # Low-pass filter (simple moving average for now)
        # In a more sophisticated implementation, you could use a proper LPF
        i_filtered = self.xp.mean(i_component)
        q_filtered = self.xp.mean(q_component)
        
        # Calculate amplitude (magnitude of complex signal)
        amplitude = self.xp.sqrt(i_filtered**2 + q_filtered**2)
        
        # Apply gain factor of 2 to compensate for double-sideband
        amplitude *= 2.0
        
        return amplitude

    def setup(self):
        """
        Setup the demodulator.
        """
        super().setup()
        
        # Initialize output
        if len(self.mode_numbers) == 1:
            self.output.value = self.dtype(0.0)
        else:
            self.output.value = self.xp.zeros(len(self.mode_numbers), dtype=self.dtype)

    def post_trigger(self):
        super().post_trigger()
        
        # Ensure output generation time is set
        if hasattr(self.output, 'generation_time'):
            self.output.generation_time = self.current_time