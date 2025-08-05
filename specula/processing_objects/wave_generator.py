import numpy as np
from specula.processing_objects.func_generator import FuncGenerator


class WaveGenerator(FuncGenerator):
    """
    Generates periodic waveforms (SIN, SQUARE, TRIANGLE).
    """
    def __init__(self,
                 wave_type='SIN',  # 'SIN', 'SQUARE', 'TRIANGLE', 'LINEAR'
                 amp: float = 1.0,
                 freq: float = 1.0,  # for LINEAR, this is the slope
                 offset: float = 0.0,
                 constant: float = 0.0,
                 vsize: int = 1,
                 output_size: int = 1,
                 target_device_idx: int = None,
                 precision: int = None):

        # Determine output size from arrays
        arrays = [np.atleast_1d(x) if not np.isscalar(x) else np.array([x]) 
                 for x in [amp, freq, offset, constant]]
        if output_size == 1:
            output_size = max(len(arr) for arr in arrays)

        super().__init__(
            output_size=output_size,
            constant=constant,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.wave_type = wave_type.upper()
        self.amp = self.to_xp(amp, dtype=self.dtype)
        self.freq = self.to_xp(freq, dtype=self.dtype)
        if wave_type.upper() == 'LINEAR':
            self.slope = self.freq
            self.amp = 1.0
        else:
            self.slope = 0.0
        self.offset = self.to_xp(offset, dtype=self.dtype)

        # Create vsize_array like in original
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)

        # Validate array sizes
        self._validate_array_sizes(
            self.amp, self.freq, self.offset, self.constant,
            names=['amp', 'freq', 'offset', 'constant']
        )

    def trigger_code(self):
        phase = self.freq * 2 * self.xp.pi * self.current_time_gpu + self.offset

        if self.wave_type == 'SIN':
            wave = self.xp.sin(phase, dtype=self.dtype)
        elif self.wave_type == 'SQUARE':
            wave = self.xp.sign(self.xp.sin(phase, dtype=self.dtype))
        elif self.wave_type == 'TRIANGLE':
            # Triangle wave using arcsin
            wave = 2 * self.xp.arcsin(self.xp.sin(phase)) / self.xp.pi
        elif self.wave_type == 'LINEAR':
            wave = self.slope * self.current_time_gpu
        else:
            raise ValueError(f"Unknown wave type: {self.wave_type}")

        self.output.value[:] = (self.amp * wave + self.constant) * self.vsize_array