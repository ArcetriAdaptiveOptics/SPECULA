import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from iteras.processing_objects.periodic_random_generator import PeriodicRandomGenerator

from test.specula_testlib import cpu_and_gpu


def drive(gen, t_seconds_list):
    """Run gen through check_ready/trigger/post_trigger at each given
    time (in seconds), returning the list of output values (copies)."""
    values = []
    for t_sec in t_seconds_list:
        t = gen.seconds_to_t(t_sec)
        gen.check_ready(t)
        gen.trigger()
        gen.post_trigger()
        values.append(cpuArray(gen.outputs['output'].value).copy())
    return values


class TestPeriodicRandomGenerator(unittest.TestCase):

    @cpu_and_gpu
    def test_rejects_non_positive_update_interval(self, target_device_idx, xp):
        with self.assertRaises(ValueError):
            PeriodicRandomGenerator(update_interval=0.0, target_device_idx=target_device_idx)
        with self.assertRaises(ValueError):
            PeriodicRandomGenerator(update_interval=-1.0, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_first_trigger_produces_a_value_immediately(self, target_device_idx, xp):
        gen = PeriodicRandomGenerator(update_interval=1.0, amp=[10.0], seed=1,
                                      target_device_idx=target_device_idx)
        gen.setup()
        values = drive(gen, [0.0])
        # A real random draw, not the all-zero default BaseValue.
        self.assertFalse(np.allclose(values[0], 0.0))

    @cpu_and_gpu
    def test_output_is_constant_within_the_update_interval(self, target_device_idx, xp):
        gen = PeriodicRandomGenerator(update_interval=0.5, amp=[10.0], seed=2,
                                      target_device_idx=target_device_idx)
        gen.setup()
        values = drive(gen, [0.0, 0.1, 0.2, 0.3, 0.4])
        for v in values[1:]:
            np.testing.assert_array_equal(v, values[0])

    @cpu_and_gpu
    def test_output_changes_after_the_update_interval_elapses(self, target_device_idx, xp):
        gen = PeriodicRandomGenerator(update_interval=0.5, amp=[10.0], seed=3,
                                      target_device_idx=target_device_idx)
        gen.setup()
        values = drive(gen, [0.0, 0.4, 0.5, 0.9])

        # still the first value just before the interval elapses
        np.testing.assert_array_equal(values[1], values[0])
        # a new value once the interval has elapsed, held until the next one
        self.assertFalse(np.allclose(values[2], values[0]))
        np.testing.assert_array_equal(values[3], values[2])

    @cpu_and_gpu
    def test_multiple_periods_each_draw_a_new_value(self, target_device_idx, xp):
        gen = PeriodicRandomGenerator(update_interval=0.1, amp=[10.0], seed=4,
                                      target_device_idx=target_device_idx)
        gen.setup()
        values = drive(gen, [0.0, 0.1, 0.2, 0.3, 0.4])
        # consecutive periods should (with overwhelming probability) differ
        for v1, v2 in zip(values[:-1], values[1:]):
            self.assertFalse(np.allclose(v1, v2))

    @cpu_and_gpu
    def test_reproducible_with_same_seed(self, target_device_idx, xp):
        gen1 = PeriodicRandomGenerator(update_interval=0.2, amp=[5.0], seed=42,
                                       target_device_idx=target_device_idx)
        gen2 = PeriodicRandomGenerator(update_interval=0.2, amp=[5.0], seed=42,
                                       target_device_idx=target_device_idx)
        gen1.setup()
        gen2.setup()
        values1 = drive(gen1, [0.0, 0.1, 0.2, 0.3])
        values2 = drive(gen2, [0.0, 0.1, 0.2, 0.3])
        for v1, v2 in zip(values1, values2):
            np.testing.assert_array_equal(v1, v2)

    @cpu_and_gpu
    def test_uniform_distribution_supported(self, target_device_idx, xp):
        gen = PeriodicRandomGenerator(update_interval=1.0, distribution='UNIFORM',
                                      amp=[2.0], constant=[10.0], seed=5,
                                      target_device_idx=target_device_idx)
        gen.setup()
        values = drive(gen, [0.0])
        self.assertTrue(9.0 <= values[0][0] <= 11.0)


if __name__ == '__main__':
    unittest.main()
