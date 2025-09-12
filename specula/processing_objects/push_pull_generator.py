from specula.processing_objects.base_generator import BaseGenerator
from specula.lib.modal_pushpull_signal import modal_pushpull_signal


class PushPullGenerator(BaseGenerator):
    """
    Generates push-pull signals for modal calibration.
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
                 repeat_cycles: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        push_pull_type = push_pull_type.upper()

        if amp is None and vect_amplitude is None:
            raise ValueError('Either "amp" or "vect_amplitude" parameters is mandatory for type PUSH/PUSHPULL')

        if nsamples != 1 and push_pull_type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL type')

        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        # Generate the time history using modal_pushpull_signal (from original)
        if push_pull_type == 'PUSH':
            time_hist = modal_pushpull_signal(
                nmodes,
                first_mode=first_mode,
                amplitude=amp,
                constant=constant_amp,
                vect_amplitude=vect_amplitude,
                only_push=True,
                ncycles=ncycles
            )
        elif push_pull_type == 'PUSHPULL':
            time_hist = modal_pushpull_signal(
                nmodes,
                first_mode=first_mode,
                amplitude=amp,
                constant=constant_amp,
                vect_amplitude=vect_amplitude,
                pattern=pattern,
                ncycles=ncycles,
                repeat_ncycles=repeat_cycles,
                nsamples=nsamples
            )
        else:
            raise ValueError(f'Unknown push_pull_type: {push_pull_type}')
        
        self.time_hist = self.to_xp(time_hist)

    def trigger_code(self):
        self.output.value[:] = self.time_hist[self.iter_counter]

