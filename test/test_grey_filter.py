import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.intensity import Intensity
from specula.processing_objects.grey_filter import GreyFilter
from test.specula_testlib import cpu_and_gpu


DIMX, DIMY = 8, 6  # non-square on purpose, to catch dimx/dimy swaps


def _make_intensity(arr, t, target_device_idx, xp):
    i = Intensity(DIMX, DIMY, target_device_idx=target_device_idx)
    i.i[:] = xp.asarray(arr)
    i.generation_time = t
    return i


def _step(obj, t):
    obj.check_ready(t)
    obj.trigger()
    obj.post_trigger()


def _run(obj, in_i, t):
    obj.inputs['in_i'].set(in_i)
    obj.setup()
    _step(obj, t)


class TestGreyFilter(unittest.TestCase):

    @cpu_and_gpu
    def test_constant_transmission(self, target_device_idx, xp):
        ref = np.random.default_rng(0).random((DIMY, DIMX)) + 0.5
        in_i = _make_intensity(ref, 3, target_device_idx, xp)
        obj = GreyFilter(transmission=0.3, target_device_idx=target_device_idx)
        _run(obj, in_i, 3)
        out = obj.outputs['out_i']
        self.assertEqual(cpuArray(out.i).shape, (DIMY, DIMX))
        np.testing.assert_allclose(cpuArray(out.i), 0.3 * cpuArray(in_i.i), rtol=1e-6)
        self.assertEqual(out.generation_time, 3)

    @cpu_and_gpu
    def test_transmission_one_and_zero(self, target_device_idx, xp):
        ref = np.random.default_rng(1).random((DIMY, DIMX)) + 0.5
        in_i = _make_intensity(ref, 1, target_device_idx, xp)
        obj1 = GreyFilter(transmission=1.0, target_device_idx=target_device_idx)
        _run(obj1, in_i, 1)
        np.testing.assert_allclose(cpuArray(obj1.outputs['out_i'].i), cpuArray(in_i.i), rtol=1e-6)
        obj0 = GreyFilter(transmission=0.0, target_device_idx=target_device_idx)
        _run(obj0, in_i, 1)
        np.testing.assert_array_equal(cpuArray(obj0.outputs['out_i'].i), np.zeros((DIMY, DIMX)))

    @cpu_and_gpu
    def test_out_of_range_transmission_raises(self, target_device_idx, xp):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                GreyFilter(transmission=bad, target_device_idx=target_device_idx)
        for ok in (0.0, 1.0):
            GreyFilter(transmission=ok, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_in_transmission_overrides_parameter(self, target_device_idx, xp):
        ref = np.random.default_rng(2).random((DIMY, DIMX)) + 0.5
        in_i = _make_intensity(ref, 1, target_device_idx, xp)
        tr = BaseValue(value=xp.array([0.25]), target_device_idx=target_device_idx)
        tr.generation_time = 1
        obj = GreyFilter(transmission=1.0, target_device_idx=target_device_idx)
        obj.inputs['in_transmission'].set(tr)
        _run(obj, in_i, 1)
        np.testing.assert_allclose(cpuArray(obj.outputs['out_i'].i),
                                   0.25 * cpuArray(in_i.i), rtol=1e-6)

    @cpu_and_gpu
    def test_in_transmission_changes_every_step(self, target_device_idx, xp):
        rng = np.random.default_rng(3)
        in_i = _make_intensity(rng.random((DIMY, DIMX)) + 0.5, 1, target_device_idx, xp)
        tr = BaseValue(value=xp.array([0.25]), target_device_idx=target_device_idx)
        obj = GreyFilter(transmission=1.0, target_device_idx=target_device_idx)
        obj.inputs['in_i'].set(in_i)
        obj.inputs['in_transmission'].set(tr)
        obj.setup()  # once, as in a simulation: no stale value, no accumulation
        for t, val in ((1, 0.25), (2, 0.5)):
            in_i.i[:] = xp.asarray(rng.random((DIMY, DIMX)) + 0.5)
            in_i.generation_time = t
            tr.value[:] = val
            tr.generation_time = t
            _step(obj, t)
            out = obj.outputs['out_i']
            np.testing.assert_allclose(cpuArray(out.i), val * cpuArray(in_i.i), rtol=1e-6)
            self.assertEqual(out.generation_time, t)

    @cpu_and_gpu
    def test_magnitude_emulation(self, target_device_idx, xp):
        ref = np.random.default_rng(4).random((DIMY, DIMX)) + 0.5
        in_i = _make_intensity(ref, 1, target_device_idx, xp)
        dm = 2.5  # 2.5 mag fainter -> flux x 0.1
        obj = GreyFilter(transmission=10 ** (-0.4 * dm), target_device_idx=target_device_idx)
        _run(obj, in_i, 1)
        ratio = float(cpuArray(obj.outputs['out_i'].i).sum()) / float(cpuArray(in_i.i).sum())
        np.testing.assert_allclose(ratio, 10 ** (-0.4 * dm), rtol=1e-6)

    def test_names(self):
        self.assertIn('in_i', GreyFilter.input_names())
        self.assertIn('in_transmission', GreyFilter.input_names())
        self.assertIn('out_i', GreyFilter.output_names())


if __name__ == '__main__':
    unittest.main()
