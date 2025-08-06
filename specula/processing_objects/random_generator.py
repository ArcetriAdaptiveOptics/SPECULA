import numpy as np
from specula.processing_objects.base_generator import BaseGenerator


class RandomGenerator(BaseGenerator):
    """
    Generates random signals (normal or uniform distribution).
    """
    def __init__(self,
                 distribution='NORMAL',  # 'NORMAL' or 'UNIFORM'
                 amp: float = 1.0,
                 constant: float = 0.0,
                 seed: int = None,
                 vsize: int = 1,
                 output_size: int = 1,
                 target_device_idx: int = None,
                 precision: int = None):

        # Validate arrays and determine output size
        temp_amp = np.atleast_1d(amp) if not np.isscalar(amp) else np.array([amp])
        temp_const = np.atleast_1d(constant) if not np.isscalar(constant) else np.array([constant])

        if output_size == 1:
            output_size = max(len(temp_amp), len(temp_const), output_size)

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.distribution = distribution.upper()
        self.amp = self.to_xp(amp, dtype=self.dtype)
        self.constant = self.to_xp(constant, dtype=self.dtype)

        # Validate array sizes
        self._validate_array_sizes(self.amp, self.constant, names=['amp', 'constant'])

        # Setup random number generator
        if seed is not None:
            self.seed = int(seed)
        else:
            self.seed = int(self.xp.around(self.xp.random.random() * 1e4))

        if hasattr(self.xp.random, "default_rng"):
            self.rng = self.xp.random.default_rng(self.seed)
        else:
            self.rng = self.xp.random

        # Create vsize_array like in original
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)

    def trigger_code(self):
        if self.distribution == 'NORMAL':
            self.output.value[:] = (
                (self.rng.standard_normal(size=self.output_size) * self.amp + self.constant) * self.vsize_array
            )
        elif self.distribution == 'UNIFORM':
            lowv = self.constant - self.amp / 2
            highv = self.constant + self.amp / 2
            self.output.value[:] = (
                self.rng.uniform(low=lowv, high=highv, size=self.output_size) * self.vsize_array
            )
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")