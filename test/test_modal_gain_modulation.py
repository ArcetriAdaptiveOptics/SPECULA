import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray
from specula.base_value import BaseValue

from iteras.processing_objects.modal_gain_modulation import ModalGainModulation

from test.specula_testlib import cpu_and_gpu


def drive(obj, t_seconds_list, gain_mod=None):
    """Run obj through check_ready/trigger/post_trigger at each given time
    (in seconds), returning the output at each (copies). gain_mod, if given,
    is a list of values fed to the gain_mod input, one per time."""
    values = []
    for k, t_sec in enumerate(t_seconds_list):
        t = obj.seconds_to_t(t_sec)
        if gain_mod is not None:
            g = BaseValue(value=np.atleast_1d(np.asarray(gain_mod[k], dtype=np.float32)))
            g.generation_time = t
            obj.inputs['gain_mod'].set(g)
        obj.check_ready(t)
        obj.trigger()
        obj.post_trigger()
        values.append(cpuArray(obj.outputs['out_gain_mod'].value).copy())
    return values


class TestModalGainModulation(unittest.TestCase):

    @cpu_and_gpu
    def test_invalid_parameters_are_rejected(self, target_device_idx, xp):
        bad = [dict(update_interval=0.0),
               dict(min_factor=1.5),
               dict(tiers=[[5, 20, 0.5]]),                 # past n_modes
               dict(tiers=[[5, 5, 0.5]]),                  # empty
               dict(tiers=[[0, 5, 1.5]]),                  # fraction > 1
               dict(tiers=[[0, 6, 0.5], [5, 10, 0.5]])]    # overlap
        for kwargs in bad:
            opts = dict(n_modes=10, tiers=[[2, 10, 0.5]], update_interval=0.5,
                        target_device_idx=target_device_idx)
            opts.update(kwargs)
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ModalGainModulation(**opts)

    @cpu_and_gpu
    def test_only_tier_modes_are_degained_within_the_bounds(self, target_device_idx, xp):
        obj = ModalGainModulation(n_modes=30, tiers=[[10, 20, 1.0]], update_interval=0.1,
                                  min_factor=0.3, seed=1, target_device_idx=target_device_idx)
        obj.setup()
        for v in drive(obj, [0.0, 0.1, 0.2]):
            np.testing.assert_array_equal(v[:10], 1.0)
            np.testing.assert_array_equal(v[20:], 1.0)
            self.assertTrue(np.all((v[10:20] >= 0.3) & (v[10:20] < 1.0)))

    @cpu_and_gpu
    def test_factors_are_held_between_redraws(self, target_device_idx, xp):
        obj = ModalGainModulation(n_modes=50, tiers=[[0, 50, 0.5]], update_interval=0.5,
                                  seed=2, target_device_idx=target_device_idx)
        obj.setup()
        v = drive(obj, [0.0, 0.2, 0.4, 0.5, 0.9])
        np.testing.assert_array_equal(v[1], v[0])
        np.testing.assert_array_equal(v[2], v[0])
        self.assertFalse(np.array_equal(v[3], v[0]))
        np.testing.assert_array_equal(v[4], v[3])

    @cpu_and_gpu
    def test_each_tier_is_degained_at_its_own_rate(self, target_device_idx, xp):
        obj = ModalGainModulation(n_modes=600, tiers=[[100, 300, 0.15], [300, 600, 0.20]],
                                  update_interval=0.1, min_factor=0.3, seed=3,
                                  target_device_idx=target_device_idx)
        obj.setup()
        v = np.array(drive(obj, np.arange(200) * 0.1))
        self.assertAlmostEqual(np.mean(v[:, 100:300] < 1), 0.15, delta=0.01)
        self.assertAlmostEqual(np.mean(v[:, 300:] < 1), 0.20, delta=0.01)
        degained = v[v < 1]
        self.assertAlmostEqual(degained.mean(), 0.65, delta=0.01)   # uniform in [0.3, 1)

    @cpu_and_gpu
    def test_gain_mod_input_multiplies_every_factor(self, target_device_idx, xp):
        obj = ModalGainModulation(n_modes=20, tiers=[[5, 20, 0.5]], update_interval=1.0,
                                  seed=4, target_device_idx=target_device_idx)
        obj.setup()
        closed, opened, half = drive(obj, [0.0, 0.1, 0.2], gain_mod=[1.0, 0.0, 0.5])
        np.testing.assert_array_equal(opened, 0.0)
        np.testing.assert_allclose(half, 0.5 * closed)

    @cpu_and_gpu
    def test_reproducible_with_the_same_seed(self, target_device_idx, xp):
        runs = []
        for _ in range(2):
            obj = ModalGainModulation(n_modes=40, tiers=[[0, 40, 0.3]], update_interval=0.1,
                                      seed=5, target_device_idx=target_device_idx)
            obj.setup()
            runs.append(drive(obj, [0.0, 0.1, 0.2]))
        for v1, v2 in zip(*runs):
            np.testing.assert_array_equal(v1, v2)


if __name__ == '__main__':
    unittest.main()
