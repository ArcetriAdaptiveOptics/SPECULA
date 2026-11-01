import numpy as np

from iteras.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from iteras.base_value import BaseValue
from iteras.connections import InputValue


class ModalGainModulation(BaseProcessingObj):
    """
    Random per-mode loop gain factors, redrawn periodically, for the
    ``gain_mod`` input of a filter such as Integrator.

    On the first trigger and then every ``update_interval`` seconds, each mode
    of a tier is de-gained with that tier's probability: its factor is drawn
    uniformly in [min_factor, 1), otherwise it is 1. Modes outside every tier
    always stay at 1. The factors are held constant between redraws.

    If the optional ``gain_mod`` input is connected, it multiplies every
    factor, so a scalar wave can still open or close the whole loop.

    Parameters
    ----------
    n_modes : int
        Number of factors: one per filter of the controller they feed.
    tiers : list of [first, end, fraction]
        Modes first..end-1 are each de-gained with probability fraction at
        every redraw. Tiers must not overlap.
    update_interval : float [s]
        Simulated time between redraws. Must be > 0.
    min_factor : float, optional
        Lowest factor a de-gained mode can get, in [0, 1] (default 0.3).
    seed : int, optional
        Seed of the random draws. None draws a different sequence every run.
    target_device_idx : int, optional
        Target device index (CPU/GPU). If None, a global setting is used.
    precision : int, optional
        Precision for computation (0 = double, 1 = single). If None, a global
        setting is used.
    """
    def __init__(self,
                 n_modes: int,
                 tiers: list,
                 update_interval: float,
                 min_factor: float = 0.3,
                 seed: int = None,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        if update_interval <= 0:
            raise ValueError(f'update_interval must be > 0, got {update_interval}')
        if not 0 <= min_factor <= 1:
            raise ValueError(f'min_factor must be in [0, 1], got {min_factor}')

        # Per-mode probability of being de-gained at a redraw
        self.probability = np.zeros(n_modes)
        in_tier = np.zeros(n_modes, dtype=bool)
        for first, end, fraction in tiers:
            first, end = int(first), int(end)
            if not 0 <= first < end <= n_modes:
                raise ValueError(f'tier [{first}, {end}) is not within the {n_modes} modes')
            if not 0 <= fraction <= 1:
                raise ValueError(f'tier [{first}, {end}): fraction must be in [0, 1], got {fraction}')
            if in_tier[first:end].any():
                raise ValueError(f'tier [{first}, {end}) overlaps another tier')
            in_tier[first:end] = True
            self.probability[first:end] = fraction

        self.min_factor = min_factor
        self.rng = np.random.default_rng(seed)
        self.update_interval_t = self.seconds_to_t(update_interval)
        self._next_update_t = 0
        self.factors = self.xp.ones(n_modes, dtype=self.dtype)

        self.out_gain_mod = BaseValue(target_device_idx=target_device_idx, precision=precision,
                                      value=self.xp.ones(n_modes, dtype=self.dtype))
        self.inputs['gain_mod'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_gain_mod'] = self.out_gain_mod

    @classmethod
    def input_names(cls):
        return {'gain_mod': InputDesc(BaseValue, 'Gain modulation applied to every mode (optional)')}

    @classmethod
    def output_names(cls):
        return {'out_gain_mod': OutputDesc(BaseValue, 'Per-mode gain modulation')}

    def _draw(self):
        n = len(self.probability)
        degained = self.rng.random(n) < self.probability
        factors = np.where(degained, self.rng.uniform(self.min_factor, 1.0, n), 1.0)
        self.factors[:] = self.to_xp(factors, dtype=self.dtype)

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        if self.current_time >= self._next_update_t:
            self._draw()
            self._next_update_t = self.current_time + self.update_interval_t

    def trigger_code(self):
        out = self.out_gain_mod.value
        out[:] = self.factors
        gain_mod = self.local_inputs['gain_mod']
        if gain_mod is not None:
            out *= gain_mod.value

    def post_trigger(self):
        super().post_trigger()
        self.out_gain_mod.generation_time = self.current_time
