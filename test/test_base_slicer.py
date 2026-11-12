import specula
specula.init(0)

from specula.loop_control import LoopControl

import unittest
from specula import cp, cpuArray, np
from iteras.processing_objects.wave_generator import WaveGenerator
from specula.processing_objects.base_operation import BaseOperation
from test.specula_testlib import cpu_and_gpu, accelerate, graph_captured
from specula.processing_objects.base_slicer import BaseSlicer
from specula.base_value import BaseValue

class TestBaseSlicer(unittest.TestCase):

    def test_indices(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(indices=[1, 3, 5])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [1, 3, 5])

    def test_slice_args(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[2, 7, 2])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [2, 4, 6])

    def test_no_args(self):
        arr = np.arange(5)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer()
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, arr)

    def test_slice_single_element(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[5, 6, 1])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [5])

    def test_slice_empty(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[3, 3, 1])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [])

    def test_slice_step_larger_than_range(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(slice_args=[2, 5, 10])
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [2])

    @cpu_and_gpu
    def test_int_index_gives_one_element_output(self, target_device_idx, xp):
        value = BaseValue(value=xp.arange(10, dtype=xp.float32) * 1.5,
                          target_device_idx=target_device_idx)
        value.generation_time = value.seconds_to_t(1)
        slicer = BaseSlicer(indices=4, target_device_idx=target_device_idx)
        slicer.inputs['in_value'].set(value)

        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output = cpuArray(slicer.outputs['out_value'].value)
        self.assertEqual(output.shape, (1,))
        np.testing.assert_array_equal(output, [6.0])

    @cpu_and_gpu
    def test_output_follows_input_dtype(self, target_device_idx, xp):
        for dtype in (xp.float32, xp.int32):
            value = BaseValue(target_device_idx=target_device_idx)
            value.value = xp.arange(10).astype(dtype)  # the constructor would cast to the default dtype
            value.generation_time = value.seconds_to_t(1)
            slicer = BaseSlicer(indices=[1, 3], target_device_idx=target_device_idx)
            slicer.inputs['in_value'].set(value)
            loop = LoopControl()
            loop.add(slicer, idx=0)
            loop.run(run_time=2, dt=1, t0=1)
            self.assertEqual(slicer.outputs['out_value'].value.dtype, dtype)

    @cpu_and_gpu
    def test_output_array_is_written_in_place(self, target_device_idx, xp):
        value = BaseValue(value=xp.arange(10, dtype=xp.float32), target_device_idx=target_device_idx)
        slicer = BaseSlicer(indices=[0, 2, 5], target_device_idx=target_device_idx)
        slicer.inputs['in_value'].set(value)
        loop = LoopControl()
        loop.add(slicer, idx=0)
        loop.start(run_time=3, dt=1)
        first = None
        for step in range(3):
            value.value[:] = xp.arange(10, dtype=xp.float32) * (step + 1)
            value.generation_time = slicer.seconds_to_t(step)
            loop.iter()
            out = slicer.outputs['out_value'].value
            np.testing.assert_array_equal(cpuArray(out), np.array([0, 2, 5]) * (step + 1))
            if first is None:
                first = out
            self.assertIs(out, first)

    def _chain(self, target_device_idx, consumer=False):
        """WaveGenerator -> BaseSlicer([0, 2, 5]) [-> BaseOperation(constant_mul=2)]"""
        gen = WaveGenerator(amp=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], freq=[37.0],
                            constant=[0.5], target_device_idx=target_device_idx)
        slicer = BaseSlicer(indices=[0, 2, 5], target_device_idx=target_device_idx)
        slicer.inputs['in_value'].set(gen.outputs['output'])
        objs = [gen, slicer]
        if consumer:
            op = BaseOperation(constant_mul=2.0, target_device_idx=target_device_idx)
            op.inputs['in_value1'].set(slicer.outputs['out_value'])
            objs.append(op)
        loop = LoopControl()
        for level, obj in enumerate(objs):
            loop.add(obj, level)
        return loop, objs

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_accelerated_slicer_follows_time_varying_input(self):
        loop, (gen, slicer) = self._chain(0)
        accelerate(slicer)
        loop.start(run_time=6e-3, dt=1e-3)
        ptrs, outs = [], []
        for _ in range(6):
            loop.iter()
            expected = cpuArray(gen.outputs['output'].value)[[0, 2, 5]]
            out = slicer.outputs['out_value'].value
            # Stale values from a replayed graph would differ from the current input
            np.testing.assert_allclose(cpuArray(out), expected, rtol=1e-6)
            ptrs.append(out.data.ptr)
            outs.append(expected)
        loop.finish()
        self.assertEqual(slicer._accel.status, 'accelerated')
        self.assertTrue(graph_captured(slicer))
        self.assertEqual(len(set(ptrs)), 1)
        # The input does change from step to step
        self.assertGreater(np.abs(outs[1] - outs[3]).max(), 1e-3)

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_accelerated_consumer_of_non_accelerated_slicer(self):
        loop, (gen, slicer, op) = self._chain(0, consumer=True)
        accelerate(op)  # the slicer is not accelerated
        loop.start(run_time=6e-3, dt=1e-3)
        for _ in range(6):
            loop.iter()  # no AcceleratorError: the slicer output is not reallocated
            expected = 2.0 * cpuArray(gen.outputs['output'].value)[[0, 2, 5]]
            np.testing.assert_allclose(cpuArray(op.outputs['out_value'].value), expected, rtol=1e-6)
        loop.finish()
        self.assertIsNone(getattr(slicer, '_accel', None))
        self.assertEqual(op._accel.status, 'accelerated')
        self.assertTrue(graph_captured(op))

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_accelerated_consumer_of_accelerated_slicer(self):
        loop, (gen, slicer, op) = self._chain(0, consumer=True)
        accelerate(slicer, op)
        loop.start(run_time=6e-3, dt=1e-3)
        for _ in range(6):
            loop.iter()
            expected = 2.0 * cpuArray(gen.outputs['output'].value)[[0, 2, 5]]
            np.testing.assert_allclose(cpuArray(op.outputs['out_value'].value), expected, rtol=1e-6)
        loop.finish()
        self.assertEqual(slicer._accel.status, 'accelerated')
        self.assertEqual(op._accel.status, 'accelerated')
