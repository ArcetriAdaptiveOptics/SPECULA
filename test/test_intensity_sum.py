import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.intensity import Intensity
from specula.processing_objects.intensity_sum import IntensitySum
from test.specula_testlib import cpu_and_gpu


DIMX, DIMY = 8, 6  # non-square on purpose, to catch dimx/dimy swaps


def _make_intensity(arr, t, target_device_idx, xp):
    i = Intensity(DIMX, DIMY, target_device_idx=target_device_idx)
    i.i[:] = xp.asarray(arr)
    i.generation_time = t
    return i


def _run(obj, inputs, t):
    obj.inputs['in_i_list'].set(inputs)
    obj.setup()
    obj.check_ready(t)
    obj.trigger()
    obj.post_trigger()


class TestIntensitySum(unittest.TestCase):

    @cpu_and_gpu
    def test_sum_of_three_over_two_steps_is_not_accumulated(self, target_device_idx, xp):
        rng = np.random.default_rng(0)
        obj = IntensitySum(target_device_idx=target_device_idx)
        ins = [Intensity(DIMX, DIMY, target_device_idx=target_device_idx) for _ in range(3)]
        obj.inputs['in_i_list'].set(ins)
        obj.setup()  # once, as in a simulation: step 2 must not see step-1 values
        for t in (1, 2):
            arrs = [rng.random((DIMY, DIMX)) for _ in range(3)]
            for i, a in zip(ins, arrs):
                i.i[:] = xp.asarray(a)
                i.generation_time = t
            obj.check_ready(t)
            obj.trigger()
            obj.post_trigger()
            out = obj.outputs['out_i']
            self.assertEqual(cpuArray(out.i).shape, (DIMY, DIMX))
            np.testing.assert_allclose(cpuArray(out.i), arrs[0] + arrs[1] + arrs[2], rtol=1e-6)
            self.assertEqual(out.generation_time, t)

    @cpu_and_gpu
    def test_single_input_is_copied(self, target_device_idx, xp):
        ref = np.arange(DIMX * DIMY, dtype=float).reshape(DIMY, DIMX) + 1.0
        obj = IntensitySum(target_device_idx=target_device_idx)
        i1 = _make_intensity(ref, 1, target_device_idx, xp)
        _run(obj, [i1], 1)
        out = obj.outputs['out_i']
        np.testing.assert_allclose(cpuArray(out.i), ref, rtol=1e-6)
        out.i[:] = 0
        np.testing.assert_allclose(cpuArray(i1.i), ref, rtol=1e-6)

    @cpu_and_gpu
    def test_total_flux_is_conserved(self, target_device_idx, xp):
        rng = np.random.default_rng(1)
        arrs = [rng.random((DIMY, DIMX)) * s for s in (1.0, 10.0, 100.0)]
        obj = IntensitySum(target_device_idx=target_device_idx)
        ins = [_make_intensity(a, 1, target_device_idx, xp) for a in arrs]
        _run(obj, ins, 1)
        total_in = sum(float(cpuArray(i.i).sum()) for i in ins)
        np.testing.assert_allclose(float(cpuArray(obj.outputs['out_i'].i).sum()),
                                   total_in, rtol=1e-6)

    @cpu_and_gpu
    def test_shape_mismatch_raises(self, target_device_idx, xp):
        obj = IntensitySum(target_device_idx=target_device_idx)
        i1 = _make_intensity(np.ones((DIMY, DIMX)), 1, target_device_idx, xp)
        i2 = Intensity(DIMY, DIMX, target_device_idx=target_device_idx)  # swapped dims
        i2.generation_time = 1
        obj.inputs['in_i_list'].set([i1, i2])
        with self.assertRaises(ValueError):
            obj.setup()

    def test_names(self):
        self.assertIn('in_i_list', IntensitySum.input_names())
        self.assertIn('out_i', IntensitySum.output_names())


if __name__ == '__main__':
    unittest.main()
