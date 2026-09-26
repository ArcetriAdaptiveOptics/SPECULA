import numpy as np

from specula.processing_objects.base_generator import BaseGenerator
from specula.lib.modal_pushpull_signal import modal_pushpull_amplitudes


class PushPullGenerator(BaseGenerator):
    """
    Push-Pull Generator processing object.
    Generates push-pull signals for modal calibration.

    The output sequence is the same as `modal_pushpull_signal`, but each step
    is computed on the fly instead of storing the full (nsteps, nmodes) time
    history, which is mostly zeros and grows as nmodes**2 * ncycles * nsamples.
    """
    def __init__(self,
                 nmodes: int,
                 first_mode: int=0,
                 push_pull_type: str = 'PUSHPULL',  # 'PUSH' or 'PUSHPULL'
                 amp: float = None,
                 constant_amp: bool=False,
                 pattern: list = [1, -1],
                 vect_amplitude: list = None,
                 ncycles: int = 1,
                 nsamples: int = 1,
                 repeat_ncycles: bool = False,
                 repeat_full_sequence: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        push_pull_type = push_pull_type.upper()

        if amp is None and vect_amplitude is None:
            raise ValueError('Either "amp" or "vect_amplitude" parameters is mandatory for type PUSH/PUSHPULL')

        if nsamples != 1 and push_pull_type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL type')

        if push_pull_type == 'PUSH':
            pattern = [1]
        elif push_pull_type != 'PUSHPULL':
            raise ValueError(f'Unknown push_pull_type: {push_pull_type}')

        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Kept on host: trigger_code() only needs scalars, no device sync
        self.vect_amplitude = np.asarray(modal_pushpull_amplitudes(
            nmodes,
            first_mode=first_mode,
            amplitude=amp,
            constant=constant_amp,
            vect_amplitude=vect_amplitude,
        ), dtype=float)
        self.pattern = np.asarray(pattern, dtype=float)
        self.first_mode = first_mode
        self.ncycles = ncycles
        self.nsamples = nsamples
        self.repeat_ncycles = repeat_ncycles
        self.repeat_full_sequence = repeat_full_sequence

        n_pokes = len(self.pattern)
        self.nsteps = n_pokes * (nmodes - first_mode) * ncycles * nsamples

    def step_to_mode_and_poke(self, step: int):
        """Return (mode index, pattern index) actuated at a given step."""
        n_pokes = len(self.pattern)
        n_modes = len(self.vect_amplitude) - self.first_mode
        row = step // self.nsamples
        if self.repeat_full_sequence:
            row = row % (n_pokes * n_modes)
            mode, poke = divmod(row, n_pokes)
        elif self.repeat_ncycles:
            mode, within = divmod(row, n_pokes * self.ncycles)
            poke = within // self.ncycles
        else:
            mode, within = divmod(row, n_pokes * self.ncycles)
            poke = within % n_pokes
        return mode + self.first_mode, poke

    def trigger_code(self):
        if self.iter_counter >= self.nsteps:
            raise IndexError(f'PushPullGenerator: step {self.iter_counter} is beyond '
                             f'the end of the push-pull sequence ({self.nsteps} steps)')
        mode, poke = self.step_to_mode_and_poke(self.iter_counter)
        self.output.value[:] = 0
        self.output.value[mode] = self.vect_amplitude[mode] * self.pattern[poke]
