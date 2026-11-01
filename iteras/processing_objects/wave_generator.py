import numpy as np
from iteras.processing_objects.base_generator import BaseGenerator
from typing import List

class WaveGenerator(BaseGenerator):
    """
    Wave Generator processing object.
    Generates periodic waveforms (SIN, SQUARE, TRIANGLE).
    """
    def __init__(self,
                 wave_type='SIN',  # 'SIN', 'SQUARE', 'TRIANGLE'
                 amp: float | List[float] = [0.0],
                 freq: float | List[float] = [0.0],
                 offset: float | List[float] = [0.0],
                 constant: float | List[float] = [0.0],
                 slope: float | List[float] = [0.0],
                 duty_cycle: float | List[float] = [0.5],
                 output_size: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        """
        duty_cycle : float or list of float, optional
            Fraction of each period (in [0, 1]) spent at +amp before switching
            to -amp. Used only for wave_type='SQUARE'. Default 0.5 (symmetric).
        """

        self.wave_type = wave_type.upper()
        if self.wave_type not in ['SIN', 'SQUARE', 'TRIANGLE']:
            raise ValueError(f"Unknown wave type: {wave_type}")

        if np.any(np.atleast_1d(duty_cycle) < 0) or np.any(np.atleast_1d(duty_cycle) > 1):
            raise ValueError(f"duty_cycle must be between 0 and 1, got {duty_cycle}")

        # Determine output size from arrays
        arrays = [np.atleast_1d(x)
                 for x in [amp, freq, offset, slope, constant, duty_cycle]]
        
        # code to check if the input arrays have consistent sizes, and if output_size is <1, set it to the max size of the input arrays
        if output_size is None or output_size < 1:
            output_size = max(len(arr) for arr in arrays)


        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.amp = self.to_xp(np.atleast_1d(amp), dtype=self.dtype)
        self.freq = self.to_xp(np.atleast_1d(freq), dtype=self.dtype)
        self.offset = self.to_xp(np.atleast_1d(offset), dtype=self.dtype)
        self.slope = self.to_xp(np.atleast_1d(slope), dtype=self.dtype)
        self.constant = self.to_xp(np.atleast_1d(constant), dtype=self.dtype)
        self.duty_cycle = self.to_xp(np.atleast_1d(duty_cycle), dtype=self.dtype)
        self.output_size_array = self.xp.ones(output_size, dtype=self.dtype)

        # Validate array sizes
        self._validate_array_sizes(
            self.amp, self.freq, self.offset, self.slope, self.constant, self.duty_cycle,
            names=['amp', 'freq', 'offset', 'slope', 'constant', 'duty_cycle']
        )

    def can_accelerate(self):
        # The time reaches trigger_code() through current_time_gpu, set in prepare_trigger()
        return True

    def trigger_code(self):
        phase = self.freq * 2 * self.xp.pi * self.current_time_gpu + self.offset

        if self.wave_type == 'SIN':
            wave = self.xp.sin(phase, dtype=self.dtype)
            self.output.set_value(
                (self.slope * self.current_time_gpu + self.amp * wave + self.constant) \
                    * self.output_size_array
            )

        elif self.wave_type == 'SQUARE':
            # +amp for the first duty_cycle fraction of each period, -amp for the rest.
            # The symmetric case keeps sign(sin()), which is 0 at the switching points
            cycle_frac = self.xp.mod(phase / (2 * self.xp.pi), 1.0)
            wave = self.xp.where(self.duty_cycle == 0.5,
                                 self.xp.sign(self.xp.sin(phase, dtype=self.dtype)),
                                 self.xp.where(cycle_frac < self.duty_cycle, 1.0, -1.0)
                                 ).astype(self.dtype)
            self.output.set_value(
                (self.slope * self.current_time_gpu + self.amp * wave + self.constant) \
                    * self.output_size_array
            )

        elif self.wave_type == 'TRIANGLE':
            # Triangle wave using arcsin
            wave = 2 * self.xp.arcsin(self.xp.sin(phase, dtype=self.dtype)) / self.xp.pi
            self.output.set_value(
                (self.slope * self.current_time_gpu + self.amp * wave + self.constant) \
                    * self.output_size_array
            )
