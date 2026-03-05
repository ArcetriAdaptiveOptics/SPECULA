import numpy as np
from specula.processing_objects.base_generator import BaseGenerator


class RandomGenerator(BaseGenerator):
    """
    Random Generator processing object.
    Generates random signals (normal or uniform distribution).
    
    Parameters:
    - distribution: 'NORMAL' or 'UNIFORM' (default: 'NORMAL')
    - amp: Amplitude of the random signal. For 'NORMAL', this is the standard deviation;
           for 'UNIFORM', this is the width of the distribution. (default: None) 
    - constant: A constant offset added to the random signal (default: 0.0)
    - seed: Seed for the random number generator (default: None, which means random seed)
    - vsize: Size of the vector to be generated (default: 1)
    - output_size: Number of random values to generate (default: 1)
    - modal_rms: Desired RMS value for the modes (mutually exclusive with 'amp') (default: None)
    - forced_zero_modes: Number of initial modes to force to 0.0
                         (default: 0, must be <= output_size)
    - scaling_law: The relationship between amplitude and radial order 'n' (options: 'CONSTANT',
                   'INVERSE', 'LINEAR') (default: 'INVERSE')
    - target_device_idx: Index of the target device for computation (e.g., GPU) (default: None)
    - precision: Numerical precision for the output (e.g., 32 or 64) (default: None)
    """
    def __init__(self,
                 distribution='NORMAL',  # 'NORMAL' or 'UNIFORM'
                 amp=None,
                 constant: float = 0.0,
                 seed: int = None,
                 vsize: int = 1,
                 output_size: int = 1,
                 modal_rms: float = None, # Modal amplitude scaling arguments
                 forced_zero_modes: int = 0,
                 scaling_law: str = 'INVERSE', # Options: 'CONSTANT', 'INVERSE', 'LINEAR'
                 target_device_idx: int = None,
                 precision: int = None):

        # 1. Mutual exclusion check for amplitudes
        if amp is not None and modal_rms is not None:
            raise ValueError("Cannot set both 'amp' and 'modal_rms'"
                             " simultaneously. Choose only one.")

        # 2. Dynamic generation of the amplitude vector
        if modal_rms is not None:
            if forced_zero_modes > output_size:
                raise ValueError(f"Number of forced zero modes ({forced_zero_modes})"
                                 f" cannot exceed output_size ({output_size}).")
            # Generate the vector and force output_size for consistency
            amp = self._generate_scaled_amps(modal_rms, output_size,
                                             forced_zero_modes, scaling_law)

        elif amp is None:
            # Fallback to the original default behavior if no amplitude parameter is passed
            amp = 1.0

        # Validate arrays and determine output size
        temp_amp = np.atleast_1d(amp) if not np.isscalar(amp) else np.array([amp])
        temp_const = np.atleast_1d(constant) if not np.isscalar(constant) \
                     else np.array([constant])

        if output_size == 1:
            output_size = max(len(temp_amp), len(temp_const), output_size)
        if output_size == 1:
            output_size = vsize

        super().__init__(
            output_size=output_size,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.distribution = distribution.upper()
        if self.distribution not in ['NORMAL', 'UNIFORM']:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Target device conversion (e.g., cupy/numpy)
        self.amp = self.to_xp(amp, dtype=self.dtype)
        self.constant = self.to_xp(constant, dtype=self.dtype)

        # Validate array sizes
        self._validate_array_sizes(self.amp, self.constant, names=['amp', 'constant'])

        # Setup random number generator
        if seed is not None:
            seed = int(seed)
        else:
            seed = int(self.xp.around(self.xp.random.random() * 1e4))

        self.rng = self.xp.random.default_rng(seed)
        self.vsize_array = self.xp.ones(vsize, dtype=self.dtype)
        self.output_size = output_size

    @staticmethod
    def _generate_scaled_amps(modal_rms: float,
                              n_modes: int,
                              forced_zero: int,
                              scaling_law: str):
        """
        Generates an amplitude vector scaled by the radial order.
        
        Args:
            modal_rms: Desired RMS value for the modes (RMS is computed as the square root
                       of the TOTAL of the squared amplitudes).
            n_modes: Total number of modes to generate.
            forced_zero: Number of initial modes forced to 0.0.
            scaling_law: The relationship between amplitude and radial order 'n'.
                         Supported laws: 'CONSTANT', 'INVERSE', 'LINEAR'.
        """
        amps = np.zeros(n_modes)
        scaling_law = scaling_law.upper()

        for i in range(n_modes):
            # Skip calculation for modes forced to zero
            if i < forced_zero:
                amps[i] = 0.0
                continue

            j = i + 1
            # Calculate radial order 'n' from Noll index 'j'
            n = int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))

            # Apply the requested scaling law
            if scaling_law == 'CONSTANT':
                amps[i] = 1
            elif scaling_law == 'INVERSE':
                amps[i] = 1 if n == 0 else 1 / n
            elif scaling_law == 'LINEAR':
                amps[i] = 1 if n == 0 else 1 * n
            else:
                raise ValueError(f"Unknown scaling law: '{scaling_law}'."
                                 f"Supported laws are: 'CONSTANT', 'INVERSE',"
                                 " 'LINEAR'.")

        # Normalize to achieve the desired modal RMS value
        current_rms = np.sqrt(np.sum(amps**2))
        if current_rms > 0:
            amps *= (modal_rms / current_rms)

        return amps

    def trigger_code(self):
        if self.distribution == 'NORMAL':
            self.output.value[:] = (
                (self.rng.standard_normal(size=self.output_size) \
                    * self.amp + self.constant) * self.vsize_array
            )
        elif self.distribution == 'UNIFORM':
            lowv = self.constant - self.amp / 2
            highv = self.constant + self.amp / 2
            self.output.value[:] = (
                self.rng.uniform(low=lowv, high=highv, size=self.output_size) * self.vsize_array
            )
