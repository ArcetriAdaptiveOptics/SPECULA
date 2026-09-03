import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.round_to_multiple import RoundToMultiple
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu


class TestRoundToMultiple(unittest.TestCase):

    @cpu_and_gpu
    def test_round_basic(self, target_device_idx, xp):
        """Values snap to the nearest multiple."""
        rtm = RoundToMultiple(multiple=800.0, target_device_idx=target_device_idx)

        # -50 -> 0, 450 -> 800, 1650 -> 1600, -1250 -> -1600
        input_vector = xp.array([-50.0, 450.0, 1650.0, -1250.0], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                 target_device_idx=target_device_idx)

        rtm.inputs['in_value'].set(input_value)
        rtm.setup()
        rtm.prepare_trigger(0)
        rtm.trigger_code()

        expected = xp.array([0.0, 800.0, 1600.0, -1600.0], dtype=xp.float64)
        xp.testing.assert_allclose(rtm.out_value.value, expected, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_round_exact_multiple_unchanged(self, target_device_idx, xp):
        """Values already at an exact multiple are unchanged."""
        rtm = RoundToMultiple(multiple=800.0, target_device_idx=target_device_idx)

        input_vector = xp.array([0.0, 800.0, -1600.0, 2400.0], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                 target_device_idx=target_device_idx)

        rtm.inputs['in_value'].set(input_value)
        rtm.setup()
        rtm.prepare_trigger(0)
        rtm.trigger_code()

        xp.testing.assert_allclose(rtm.out_value.value, input_vector, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_round_with_gain(self, target_device_idx, xp):
        """Gain scales the rounded (snapped) output, applied after rounding."""
        rtm = RoundToMultiple(multiple=800.0, gain=0.0, target_device_idx=target_device_idx)

        input_vector = xp.array([450.0, 1650.0], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                 target_device_idx=target_device_idx)

        rtm.inputs['in_value'].set(input_value)
        rtm.setup()
        rtm.prepare_trigger(0)
        rtm.trigger_code()

        # gain=0 gates the correction off entirely, regardless of the snap
        expected = xp.array([0.0, 0.0], dtype=xp.float64)
        xp.testing.assert_allclose(rtm.out_value.value, expected, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_round_zero_multiple_raises(self, target_device_idx, xp):
        """multiple=0 is rejected at construction time."""
        with self.assertRaises(ValueError) as cm:
            RoundToMultiple(multiple=0.0, target_device_idx=target_device_idx)
        self.assertIn("multiple must be non-zero", str(cm.exception))

    @cpu_and_gpu
    def test_round_generation_time(self, target_device_idx, xp):
        """generation_time follows current_time."""
        rtm = RoundToMultiple(multiple=800.0, target_device_idx=target_device_idx)

        input_vector = xp.array([100.0, 200.0], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                 target_device_idx=target_device_idx)

        rtm.inputs['in_value'].set(input_value)
        rtm.setup()

        for t in [5, 10, 15]:
            rtm.prepare_trigger(t)
            rtm.trigger_code()
            self.assertEqual(rtm.out_value.generation_time, rtm.current_time)

    @cpu_and_gpu
    def test_round_output_dimensions(self, target_device_idx, xp):
        """Output shape matches input shape."""
        rtm = RoundToMultiple(multiple=800.0, target_device_idx=target_device_idx)

        input_vector = xp.zeros(5, dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                 target_device_idx=target_device_idx)

        rtm.inputs['in_value'].set(input_value)
        rtm.setup()
        rtm.prepare_trigger(0)
        rtm.trigger_code()

        self.assertEqual(rtm.out_value.value.shape, (5,))


if __name__ == '__main__':
    unittest.main()
