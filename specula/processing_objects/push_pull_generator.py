import numpy as np
from specula.processing_objects.base_generator import BaseGenerator
from specula.lib.modal_pushpull_signal import modal_pushpull_signal


class PushPullGenerator(BaseGenerator):
    """
    Generates push-pull signals for modal calibration.
    
    Extracted from BaseGenerator's PUSH and PUSHPULL types.
    """
    def __init__(self,
                 nmodes: int,
                 push_pull_type: str = 'PUSHPULL',  # 'PUSH' or 'PUSHPULL'
                 amp: float = None,
                 vect_amplitude: list = None,
                 ncycles: int = 1,
                 nsamples: int = 1,
                 repeat_cycles: bool = False,
                 target_device_idx: int = None,
                 precision: int = None):

        if amp is None and vect_amplitude is None:
            raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH/PUSHPULL')

        if nsamples != 1 and push_pull_type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL type')

        super().__init__(
            output_size=nmodes,
            target_device_idx=target_device_idx,
            precision=precision
        )

        self.push_pull_type = push_pull_type.upper()

        # Generate the time history using modal_pushpull_signal (from original)
        if self.push_pull_type == 'PUSH':
            self.time_hist = modal_pushpull_signal(
                nmodes,
                amplitude=amp,
                vect_amplitude=vect_amplitude,
                only_push=True,
                ncycles=ncycles
            )
        elif self.push_pull_type == 'PUSHPULL':
            self.time_hist = modal_pushpull_signal(
                nmodes,
                amplitude=amp,
                vect_amplitude=vect_amplitude,
                ncycles=ncycles,
                repeat_ncycles=repeat_cycles,
                nsamples=nsamples
            )
        else:
            raise ValueError(f'Unknown push_pull_type: {self.push_pull_type}')

    def trigger_code(self):
        """From original: VIB_HIST, VIB_PSD, PUSH, PUSHPULL, TIME_HIST case"""
        self.output.value[:] = self.get_time_hist_at_current_time()

    def get_time_hist_at_current_time(self):
        """From original BaseGenerator"""
        return self.to_xp(self.time_hist[self.iter_counter])