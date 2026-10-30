import unittest

import specula
specula.init(0)  # Default target device

from specula import cp, np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.loop_control import LoopControl
from iteras.processing_objects.wave_generator import WaveGenerator
from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
from iteras.accelerator import Accelerator


class Doubler(BaseProcessingObj):
    '''out = 2 * in, with options to break the rules of accelerated objects'''

    def __init__(self, target_device_idx=None, realloc=False, cublas=False, capturable=True,
                 time_in_trigger=False):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.out = BaseValue(value=self.xp.zeros(4, dtype=self.dtype), target_device_idx=target_device_idx)
        self.outputs['out_value'] = self.out
        self.buf = self.xp.zeros(4, dtype=self.dtype)
        self.mat = self.xp.eye(4, dtype=self.dtype)
        self.realloc = realloc
        self.cublas = cublas
        self.capturable = capturable
        self.time_in_trigger = time_in_trigger

    def can_accelerate(self):
        return self.capturable

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        v = self.local_inputs['in_value'].value
        if self.realloc:
            self.buf = v * 1.0      # a new array at each step
        else:
            self.buf[:] = v

    def trigger_code(self):
        if self.cublas:
            self.out.value[:] = 2 * (self.mat @ self.buf)
        else:
            self.out.value[:] = 2 * self.buf
        if self.time_in_trigger:
            self.out.generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
        if not self.time_in_trigger:
            self.out.generation_time = self.current_time


class TripleDoubler(Doubler):
    '''Another trigger_code(), not covered by the can_accelerate() of Doubler'''

    def trigger_code(self):
        self.out.value[:] = 3 * self.buf


class Counter(BaseProcessingObj):
    '''Counts the steps in which its input was refreshed'''

    def __init__(self, target_device_idx=None):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.count = 0

    def trigger_code(self):
        self.count += 1


class Slow(BaseProcessingObj):
    '''Long GPU work on its own stream (CUDA graph, see build_stream()), from its input'''

    def __init__(self, target_device_idx=None, n=1 << 21):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.out = BaseValue(value=self.xp.zeros(n, dtype=self.dtype), target_device_idx=target_device_idx)
        self.outputs['out_value'] = self.out
        self.ramp = self.xp.linspace(0, 1, n, dtype=self.dtype)

    def setup(self):
        super().setup()
        self.build_stream()

    def trigger_code(self):
        x = self.out.value
        x[:] = self.ramp * self.local_inputs['in_value'].value[0]
        for _ in range(30):
            x[:] = self.xp.sin(x) + 0.5

    def post_trigger(self):
        super().post_trigger()
        self.out.generation_time = self.current_time


class Total(BaseProcessingObj):
    '''Sum of its inputs, times a factor, on its own stream if *own_stream*'''

    def __init__(self, target_device_idx=None, factor=1.0, own_stream=True):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_values'] = InputList(type=BaseValue)
        self.out = BaseValue(value=self.xp.zeros(1, dtype=self.dtype), target_device_idx=target_device_idx)
        self.outputs['out_value'] = self.out
        self.factor = factor
        self.own_stream = own_stream

    def setup(self):
        super().setup()
        if self.own_stream:
            self.build_stream()

    def can_accelerate(self):
        return True

    def trigger_code(self):
        self.out.value[:] = 0
        for v in self.local_inputs['in_values']:
            self.out.value += self.xp.sum(v.value) * self.factor

    def post_trigger(self):
        super().post_trigger()
        self.out.generation_time = self.current_time


@unittest.skipIf(cp is None, 'Cupy not found')
class TestOwnStreams(unittest.TestCase):

    def run_graph(self, accelerate, steps=8):
        '''generator -> slow -> (a, b on their own streams, same level) -> total (default stream)'''
        gen = WaveGenerator(amp=[1.0], freq=[37.0], constant=[0.5], output_size=4, target_device_idx=0)
        slow = Slow(target_device_idx=0)
        a = Total(target_device_idx=0, factor=1.0)
        b = Total(target_device_idx=0, factor=-2.0)
        total = Total(target_device_idx=0, own_stream=False)
        slow.inputs['in_value'].set(gen.outputs['output'])
        a.inputs['in_values'].set([slow.outputs['out_value']])
        b.inputs['in_values'].set([slow.outputs['out_value']])
        total.inputs['in_values'].set([a.outputs['out_value'], b.outputs['out_value']])
        objs = [gen, slow, a, b, total]
        loop = LoopControl()
        for obj, level in zip(objs, [0, 1, 2, 2, 3]):
            loop.add(obj, level)
        if accelerate:
            Accelerator(AcceleratorOptions()).apply(objs)
        loop.start(run_time=steps * 1e-3, dt=1e-3)
        out = []
        for _ in range(steps):
            loop.iter()
            out.append(float(total.out.value[0]))
        loop.finish()
        return np.array(out), objs

    def test_same_results_with_own_streams(self):
        ref, _ = self.run_graph(False)
        out, (gen, slow, a, b, total) = self.run_graph(True)
        np.testing.assert_allclose(out, ref, rtol=1e-5)
        for obj in (slow, a, b):
            self.assertIsNotNone(obj._accel.graph)
            self.assertIs(obj._accel.launch_stream(), obj.stream)
        self.assertIsNot(total._accel.launch_stream(), slow.stream)


@unittest.skipIf(cp is None, 'Cupy not found')
class TestAccelerator(unittest.TestCase):

    def run_chain(self, accelerate, steps=12, options=None, on_step=None, **kwargs):
        '''Generator -> Doubler -> Counter. Return the Doubler outputs, the objects and the accelerator'''
        gen = WaveGenerator(amp=[1.0], freq=[37.0], constant=[0.5], output_size=4, target_device_idx=0)
        dbl = Doubler(target_device_idx=0, **kwargs)
        cnt = Counter(target_device_idx=0)
        dbl.inputs['in_value'].set(gen.outputs['output'])
        cnt.inputs['in_value'].set(dbl.outputs['out_value'])
        loop = LoopControl()
        for i, obj in enumerate([gen, dbl, cnt]):
            loop.add(obj, i)
        accelerator = None
        if accelerate:
            accelerator = Accelerator(options or AcceleratorOptions())
            accelerator.apply([gen, dbl, cnt])
        loop.start(run_time=steps * 1e-3, dt=1e-3)
        out = []
        for i in range(steps):
            loop.iter()
            if on_step is not None:
                on_step(i, gen, dbl)
            out.append(cpuArray(dbl.out.value).copy())
        loop.finish()
        return np.array(out), (gen, dbl, cnt), accelerator

    def check_same(self, accel_kwargs=None, **kwargs):
        ref, _, _ = self.run_chain(False, **kwargs)
        out, objs, accelerator = self.run_chain(True, **(accel_kwargs or {}), **kwargs)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
        # The generator depends on the time: the outputs change at each step
        self.assertTrue(np.all(np.abs(np.diff(ref[:, 0])) > 1e-4))
        return objs, accelerator

    def status(self, accelerator, name):
        return {obj.name: runner.status for obj, runner, _ in accelerator.rows if runner is not None}[name]

    def test_accelerated_chain_matches_normal_run(self):
        (gen, dbl, cnt), accelerator = self.check_same()
        self.assertEqual(self.status(accelerator, 'WaveGenerator'), 'accelerated')
        self.assertEqual(self.status(accelerator, 'Doubler'), 'accelerated')
        self.assertIsNotNone(dbl._accel.graph)
        # The counter has no graph-safe trigger_code: not accelerated, but triggered at each step
        self.assertTrue(self.status(accelerator, 'Counter').startswith('can_accelerate()'))
        self.assertEqual(cnt.count, 12)

    def test_reallocated_array_stops_acceleration(self):
        with self.assertLogs('iteras.accelerator', level='WARNING') as logs:
            (_, dbl, _), accelerator = self.check_same(realloc=True)
        self.assertIsNone(dbl._accel)
        self.assertIn('array buf reallocated', self.status(accelerator, 'Doubler'))
        self.assertIn('not accelerated anymore', logs.output[0])

    def test_failed_capture_stops_acceleration(self):
        with self.assertLogs('iteras.accelerator', level='WARNING'):
            (_, dbl, _), accelerator = self.check_same(cublas=True)
        self.assertIsNone(dbl._accel)
        self.assertIn('capture failed', self.status(accelerator, 'Doubler'))

    def test_can_accelerate_false(self):
        (_, dbl, _), accelerator = self.check_same(capturable=False)
        self.assertIsNone(dbl._accel.graph)
        self.assertEqual(self.status(accelerator, 'Doubler'), 'can_accelerate(): False')

    def test_output_time_set_in_trigger_code(self):
        (_, _, cnt), accelerator = self.check_same(time_in_trigger=True)
        self.assertEqual(self.status(accelerator, 'Doubler'), 'accelerated')
        # The consumer sees the output refreshed at each step, as without accelerator
        self.assertEqual(cnt.count, 12)

    def test_invalidate_captures_again(self):
        graphs = []

        def on_step(i, gen, dbl):
            if dbl._accel is not None:
                graphs.append(dbl._accel.graph)
            if i == 5:
                dbl.invalidate_graph()

        (_, dbl, _), accelerator = self.check_same(accel_kwargs={'on_step': on_step})
        self.assertEqual(self.status(accelerator, 'Doubler'), 'accelerated')
        # Graph recorded at each step, before the invalidation of step 5
        self.assertIs(graphs[4], graphs[5])
        self.assertIsNot(graphs[5], graphs[6])
        self.assertIs(graphs[6], graphs[7])

    def test_capture_after(self):
        def on_step(i, gen, dbl):
            # Captured at the trigger after capture_after triggers
            self.assertEqual(dbl._accel.graph is not None, i >= 3)

        self.check_same(accel_kwargs={'on_step': on_step,
                                      'options': AcceleratorOptions(capture_after=3)})

    def test_derived_trigger_code_needs_its_own_declaration(self):
        dbl = TripleDoubler(target_device_idx=0)
        accelerator = Accelerator(AcceleratorOptions())
        accelerator.apply([dbl])
        self.assertIsNone(dbl._accel)
        reasons = {obj.name: reason for obj, _, reason in accelerator.rows}
        self.assertIn('TripleDoubler.trigger_code() is not covered by Doubler.can_accelerate()',
                      reasons['TripleDoubler'])

    def test_classes_redefining_trigger_code_declare_can_accelerate(self):
        # PsfCoronagraph redefines the trigger_code() of PSF, and declares can_accelerate() itself
        from iteras.accelerator import _can_accelerate_covers_trigger_code
        from specula.processing_objects.psf import PSF
        from specula.processing_objects.psf_coronagraph import PsfCoronagraph
        self.assertTrue(_can_accelerate_covers_trigger_code(PSF))
        self.assertTrue(_can_accelerate_covers_trigger_code(PsfCoronagraph))
        self.assertIn('can_accelerate', vars(PsfCoronagraph))

    def test_excluded_object(self):
        (_, dbl, _), accelerator = self.check_same(accel_kwargs={'options': AcceleratorOptions(exclude=['Doubler'])})
        self.assertIsNone(dbl._accel)
        reasons = {obj.name: reason for obj, _, reason in accelerator.rows}
        self.assertEqual(reasons['Doubler'], 'excluded')
        self.assertIn('Doubler', accelerator.report())
        self.assertIn('1 of 3 objects accelerated', accelerator.report())

    def test_accelerated_object_does_not_wait_for_the_gpu(self):
        (_, dbl, _), _ = self.check_same()
        self.assertIsNotNone(dbl._accel)
        # post_trigger() of a accelerated object with its own graph does not wait for its stream
        dbl.stream = cp.cuda.Stream()
        dbl.cuda_graph = object()
        dbl.inputs_changed = True
        dbl.post_trigger()


class TestAcceleratorOptions(unittest.TestCase):

    def test_capture_after_must_be_positive(self):
        with self.assertRaises(ValueError):
            AcceleratorOptions(capture_after=0)

    def test_simul_sections(self):
        from specula.simul import Simul
        sim = Simul('unused.yml')
        params = {'main': {'class': 'SimulParams'}, 'accelerator': {'class': 'Accelerator', 'exclude': ['a']}}
        self.assertEqual(sim.accelerator_keys(params), ['accelerator'])
        options = sim.take_accelerator_section(params)
        self.assertEqual(options.exclude, ['a'])
        self.assertNotIn('accelerator', params)
        self.assertIsNone(sim.take_accelerator_section(params))
        sim.add_accelerator_section(params)
        self.assertEqual(params['accelerator'], {'class': 'Accelerator'})
        # Not added twice
        sim.add_accelerator_section(params)
        self.assertEqual(sim.accelerator_keys(params), ['accelerator'])

    def test_allocation_of_other_acceleration_mode_is_stale(self):
        from iteras.allocator import check_meta, StaleAllocation
        meta = {'signature': {}, 'pins': {}, 'architecture': {}, 'precision': 1, 'accelerate': False}
        check_meta(dict(meta), dict(meta), 'f')
        # Allocations written before the accelerator existed were computed without it
        check_meta({k: v for k, v in meta.items() if k != 'accelerate'}, dict(meta), 'f')
        with self.assertRaisesRegex(StaleAllocation, 'accelerator enabled'):
            check_meta(dict(meta), dict(meta, accelerate=True), 'f')


if __name__ == '__main__':
    unittest.main()
