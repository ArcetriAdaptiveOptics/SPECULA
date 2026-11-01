from typing import List

from iteras.processing_objects.random_generator import RandomGenerator


class PeriodicRandomGenerator(RandomGenerator):
    """
    Periodic Random Generator processing object.

    Behaves exactly like RandomGenerator (same distributions and
    parameters), except that a new random value is only drawn every
    `update_interval` seconds of simulated time, instead of at every
    trigger; the previously drawn value is held constant in between.

    This is useful to drive a slowly-varying parameter (e.g. atmospheric
    seeing or wind) with genuinely random values that still force
    downstream objects to periodically react to a change, rather than
    either a fixed constant or noise that changes every single timestep.

    Parameters
    ----------
    update_interval : float [s]
        How often (in seconds of simulated time) a new random value is
        drawn. Must be > 0. A new value is always drawn on the very first
        trigger.
    (all other parameters: see RandomGenerator)
    """
    def __init__(self,
                 update_interval: float,
                 distribution='NORMAL',
                 amp: List[float] = None,
                 constant: List[float] = None,
                 seed: int = None,
                 output_size: int = 0,
                 modal_rms: float = None,
                 forced_zero_modes: int = 0,
                 scaling_law: str = 'INVERSE',
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(
            distribution=distribution,
            amp=amp,
            constant=constant,
            seed=seed,
            output_size=output_size,
            modal_rms=modal_rms,
            forced_zero_modes=forced_zero_modes,
            scaling_law=scaling_law,
            target_device_idx=target_device_idx,
            precision=precision,
        )

        if update_interval <= 0:
            raise ValueError(f'update_interval must be > 0, got {update_interval}')

        # Kept in the same internal integer time representation as
        # self.current_time, to avoid any floating-point drift from
        # repeatedly adding a float interval over a long simulation.
        self.update_interval_t = self.seconds_to_t(update_interval)
        self._next_update_t = 0

    def trigger_code(self):
        if self.current_time >= self._next_update_t:
            super().trigger_code()
            self._next_update_t = self.current_time + self.update_interval_t
