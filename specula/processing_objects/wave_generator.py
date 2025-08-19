import numpy as np
from specula.processing_objects.base_generator import BaseGenerator


class WaveGenerator(BaseGenerator):
    """
    Generates periodic waveforms (SIN, SQUARE, TRIANGLE).
    """
    def __init__(self,
                 wave_type='SIN',  # 'SIN', 'SQUARE', 'TRIANGLE'
                 amp: float = 0.0,
                 freq: float = 0.0,
                 offset: float = 0.0,
                 constant: float = 0.0,
                 slope: float = 0.0,
                 vsize: int = 1,
                 output_size: int = 1,
                 target_device_idx: int = None,
                 precision: int = None):

        # Determine output size from arrays
        arrays = [np.atleast_1d(x) if not np.isscalar(x) else np.array([x])
                 for x in [amp, freq, offset, slope, constant]]
        if output_size == 1:
            output_size = max(len(arr) for arr in arrays)

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.wave_type = wave_type.upper()

        self.amp = self.to_xp(amp, dtype=self.dtype)
        self.freq = self.to_xp(freq, dtype=self.dtype)
        self.offset = self.to_xp(offset, dtype=self.dtype)
        self.slope = self.to_xp(slope, dtype=self.dtype)
        self.constant = self.to_xp(constant, dtype=self.dtype)

        # Create vsize_array like in original
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)

        # Validate array sizes
        self._validate_array_sizes(
            self.amp, self.freq, self.offset, self.slope, self.constant,
            names=['amp', 'freq', 'offset', 'slope', 'constant']
        )

    def trigger_code(self):
        phase = self.freq * 2 * self.xp.pi * self.current_time_gpu + self.offset
        if self.wave_type == 'SIN':
            wave = self.xp.sin(phase, dtype=self.dtype)
            self.output.value[:] = (self.slope * self.current_time_gpu +self.amp * wave + self.constant) * self.vsize_array

        elif self.wave_type == 'SQUARE':
            wave = self.xp.sign(self.xp.sin(phase, dtype=self.dtype))
            self.output.value[:] = (self.slope * self.current_time_gpu + self.amp * wave + self.constant) * self.vsize_array

        elif self.wave_type == 'TRIANGLE':
            # Triangle wave using arcsin
            wave = 2 * self.xp.arcsin(self.xp.sin(phase)) / self.xp.pi
            self.output.value[:] = (self.slope * self.current_time_gpu + self.amp * wave + self.constant) * self.vsize_array

        else:
            raise ValueError(f"Unknown wave type: {self.wave_type}")