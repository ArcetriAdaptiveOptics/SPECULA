import unittest

import specula
specula.init(0)

from specula import np, cpuArray
from specula.data_objects.slopes import Slopes
from specula.processing_objects.slopes2d import Slopes2D

from test.specula_testlib import cpu_and_gpu


class TestSlopes2D(unittest.TestCase):

    @cpu_and_gpu
    def test_trigger_outputs_get2d_result(self, target_device_idx, xp):
        conv = Slopes2D(target_device_idx=target_device_idx)

        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        slopes.slopes[:] = xp.arange(4)
        slopes.single_mask = xp.zeros((2, 2), dtype=bool)
        slopes.display_map = xp.array([0, 1])
        slopes.generation_time = conv.seconds_to_t(1)

        conv.inputs['in_slopes'].set(slopes)
        conv.setup()
        t = conv.seconds_to_t(1)
        conv.check_ready(t)
        conv.trigger()
        conv.post_trigger()

        expected = cpuArray(slopes.get2d())
        np.testing.assert_array_equal(cpuArray(conv.outputs['out_value'].value), expected)
        self.assertEqual(conv.outputs['out_value'].generation_time, t)


if __name__ == '__main__':
    unittest.main()
