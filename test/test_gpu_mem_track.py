import gc
import logging
import sys
from types import ModuleType
import unittest
from inspect import signature
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import weakref

import specula
specula.init(-1)

from specula import cp
import specula.base_time_obj as bto
from specula.base_time_obj import BaseTimeObj, gpu_mem_report, mem_registry_mark, top_level_objs
from specula.loop_control import LoopControl
import specula.simul as simul_module


class FakeArray:
    def __init__(self, ptr, size):
        self.data = SimpleNamespace(mem=SimpleNamespace(ptr=ptr, size=size))


def gpu_available():
    try:
        return cp is not None and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@unittest.skipUnless(gpu_available(), 'GPU memory integration test requires CuPy and a GPU')
class TestGpuMemoryIntegration(unittest.TestCase):
    def test_nested_object_allocations_are_tracked_on_gpu(self):
        class Child(BaseTimeObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.array = cp.zeros(2048, dtype=cp.uint8)

        class Parent(BaseTimeObj):
            def __init__(self):
                super().__init__(target_device_idx=0)
                self.array = cp.zeros(1024, dtype=cp.uint8)
                self.child = Child()

        parent = Parent()
        self.assertGreater(parent.gpu_bytes_used, 0)
        self.assertGreater(parent.gpu_bytes_children, 0)
        self.assertGreater(parent.gpu_bytes_own, 0)
        self.assertGreater(parent.gpu_bytes_held(), 1024 + 2048 - 1)


class TestGpuMemoryAccounting(unittest.TestCase):
    def setUp(self):
        self.stack = []
        self.registry = weakref.WeakValueDictionary()
        self.registry_mark = patch.object(bto, '_mem_count_stack', self.stack)
        self.registry_dict = patch.object(bto, '_top_level_objs', self.registry)
        self.registry_counter = patch.object(bto, '_top_level_counter', 0)
        self.registry_mark.start()
        self.registry_dict.start()
        self.registry_counter.start()
        self.pool = {0: 0, 1: 0, 2: 0, 3: 0}
        self.pool_reader = bto._pool_used_bytes
        self.pool_patch = patch.object(bto, '_pool_used_bytes',
                                       side_effect=lambda device: self.pool[device])
        self.pool_patch.start()
        self.cp_patch = patch.object(bto, 'cp', SimpleNamespace(ndarray=FakeArray))
        self.cp_patch.start()
        self.addCleanup(self.cp_patch.stop)
        self.addCleanup(self.pool_patch.stop)
        self.addCleanup(self.registry_counter.stop)
        self.addCleanup(self.registry_dict.stop)
        self.addCleanup(self.registry_mark.stop)

    @staticmethod
    def make_obj(device=0):
        obj = BaseTimeObj(target_device_idx=-1)
        obj.target_device_idx = device
        obj.gpu_bytes_used = 0
        obj.gpu_bytes_children = 0
        return obj

    def test_nested_allocations_and_later_child_updates(self):
        parent = self.make_obj()
        child = self.make_obj()

        parent.startMemUsageCount()
        self.pool[0] += 1
        child.startMemUsageCount()
        self.pool[0] += 2
        child.stopMemUsageCount()
        self.pool[0] += 3
        parent.stopMemUsageCount()

        self.assertEqual(parent.gpu_bytes_used, 6)
        self.assertEqual(parent.gpu_bytes_children, 2)
        self.assertEqual(parent.gpu_bytes_own, 4)
        self.assertEqual(child.gpu_bytes_used, 2)
        self.assertEqual(top_level_objs(), [parent])

        child.startMemUsageCount()
        self.pool[0] += 4
        child.stopMemUsageCount()
        self.assertEqual(child.gpu_bytes_used, 6)
        self.assertEqual(parent.gpu_bytes_used, 10)
        self.assertEqual(parent.gpu_bytes_children, 6)
        self.assertEqual(parent.gpu_bytes_own, 4)

    def test_device_is_added_to_pre_init_count_when_not_hinted(self):
        class Device:
            use = MagicMock()

        device = Device()
        fake_cp = SimpleNamespace(
            cuda=SimpleNamespace(Device=MagicMock(return_value=device)),
            ndarray=FakeArray)
        modules = {}
        for name in ('cupyx', 'cupyx.scipy'):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        for name, exports in (
                ('cupyx.scipy.ndimage', ('rotate', 'shift', 'center_of_mass')),
                ('cupyx.scipy.fft', ('ifft2', 'idct', 'dct')),
                ('cupyx.scipy.linalg', ('lu_factor', 'lu_solve'))):
            module = ModuleType(name)
            for export in exports:
                setattr(module, export, MagicMock())
            modules[name] = module
        cupy = ModuleType('cupy')
        cupy.__path__ = []
        modules['cupy'] = cupy
        util = ModuleType('cupy._util')
        util.PerformanceWarning = RuntimeWarning
        modules['cupy._util'] = util

        with patch.object(bto, 'cp', fake_cp), \
             patch.object(bto, 'default_target_device_idx', -1), \
             patch.object(bto, '_used_devices', set()), \
             patch.object(bto, '_pool_used_bytes', side_effect=[50, 75]) as pool_used, \
             patch.dict(sys.modules, modules):
            class DeviceSelectedInConstructor(BaseTimeObj):
                def __init__(self, device_idx):
                    super().__init__(target_device_idx=device_idx)

            obj = DeviceSelectedInConstructor(0)

        self.assertEqual(obj.gpu_bytes_used, 25)
        self.assertEqual(pool_used.call_args_list[0].args, (0,))
        self.assertEqual(pool_used.call_args_list[1].args, (0,))
        device.use.assert_called_once_with()
        self.assertEqual(obj.PerformanceWarning, RuntimeWarning)

    def test_child_counted_during_parent_count_is_not_double_counted(self):
        parent = self.make_obj()
        child = self.make_obj()
        parent.startMemUsageCount()
        child.startMemUsageCount()
        self.pool[0] += 5
        child.stopMemUsageCount()
        parent.stopMemUsageCount()

        self.assertEqual(parent.gpu_bytes_used, 5)
        self.assertEqual(parent.gpu_bytes_children, 5)
        self.assertEqual(parent.gpu_bytes_own, 0)

    def test_child_on_different_device_is_a_separate_top_level_object(self):
        parent = self.make_obj(device=0)
        child = self.make_obj(device=1)
        parent.startMemUsageCount()
        child.startMemUsageCount()
        self.pool[1] += 8
        child.stopMemUsageCount()
        parent.stopMemUsageCount()

        self.assertEqual(parent.gpu_bytes_used, 0)
        self.assertEqual(parent.gpu_bytes_children, 0)
        self.assertEqual(child.gpu_bytes_used, 8)
        self.assertCountEqual(top_level_objs(), [parent, child])

    def test_cpu_and_untracked_device_counts_do_not_add_gpu_bytes(self):
        obj = self.make_obj(device=-1)
        obj.startMemUsageCount()
        self.pool[0] += 10
        obj.stopMemUsageCount()
        self.assertEqual(obj.gpu_bytes_used, 0)

        obj = self.make_obj(device=2)
        obj.startMemUsageCount()
        self.pool[2] += 7
        obj.stopMemUsageCount()
        self.assertEqual(obj.gpu_bytes_used, 7)

    def test_before_init_reads_only_used_default_and_hinted_devices(self):
        obj = BaseTimeObj.__new__(BaseTimeObj)
        read = MagicMock(return_value=0)
        with patch.object(bto, '_pool_used_bytes', read), \
             patch.object(bto, '_used_devices', {1}), \
             patch.object(bto, 'default_target_device_idx', 0):
            obj.startMemUsageCount(device_hint=3)

        self.assertEqual({call.args[0] for call in read.call_args_list}, {0, 1, 3})
        obj.target_device_idx = -1
        obj.stopMemUsageCount()

    def test_pool_reader_enters_the_requested_device(self):
        events = []

        class Device:
            def __init__(self, index):
                events.append(('device', index))

            def __enter__(self):
                events.append(('enter',))

            def __exit__(self, *args):
                events.append(('exit',))

        pool = MagicMock()
        pool.used_bytes.return_value = 123
        cp = SimpleNamespace(cuda=SimpleNamespace(Device=Device),
                             get_default_memory_pool=lambda: pool)
        with patch.object(bto, 'cp', cp):
            self.assertEqual(self.pool_reader(2), 123)
        self.assertEqual(events, [('device', 2), ('enter',), ('exit',)])
        pool.used_bytes.assert_called_once_with()

    def test_nested_wrapped_method_stops_counting_after_exception(self):
        calls = []

        class Meter:
            def startMemUsageCount(self, device_hint=None):
                calls.append(('start', device_hint))

            def stopMemUsageCount(self):
                calls.append(('stop',))

        def setup(self, target_device_idx=None):
            calls.append(('run', target_device_idx))
            raise ValueError('setup failed')

        wrapped = BaseTimeObj.monitorMem(setup)
        with self.assertRaisesRegex(ValueError, 'setup failed'):
            wrapped(Meter(), target_device_idx=4)
        self.assertEqual(calls, [('start', 4), ('run', 4), ('stop',)])

    def test_pickle_state_omits_weak_references(self):
        obj = self.make_obj()
        obj._mem_owner = weakref.ref(obj)
        obj._mem_parent = weakref.ref(obj)
        state = obj.__getstate__()
        self.assertNotIn('_mem_owner', state)
        self.assertNotIn('_mem_parent', state)

    def test_registry_does_not_keep_top_level_objects_alive(self):
        obj = self.make_obj()
        obj.startMemUsageCount()
        obj.stopMemUsageCount()
        mark = mem_registry_mark() - 1
        ref = weakref.ref(obj)

        del obj
        gc.collect()

        self.assertIsNone(ref())
        self.assertEqual(top_level_objs(since=mark), [])

    def test_gpu_bytes_held_deduplicates_arrays_and_excludes_inputs(self):
        parent = self.make_obj()
        child = self.make_obj()
        child._mem_owner = weakref.ref(parent)
        parent.child = child
        parent.array = FakeArray(10, 100)
        parent.view = FakeArray(10, 100)
        child.array = FakeArray(20, 200)
        parent.arrays = [FakeArray(40, 50)]
        parent.inputs = {'ignored': FakeArray(30, 300)}
        parent.local_inputs = {}

        self.assertEqual(parent.gpu_bytes_held(), 350)
        seen = set()
        self.assertEqual(parent.gpu_bytes_held(seen), 350)
        self.assertEqual(parent.gpu_bytes_held(seen), 0)

    def test_gpu_memory_report_groups_gpu_objects_and_uses_names(self):
        parent = self.make_obj(device=0)
        other = self.make_obj(device=1)
        cpu_obj = self.make_obj(device=-1)
        parent.gpu_bytes_used = 6 * bto.MB
        parent.gpu_bytes_children = 2 * bto.MB
        parent.array = FakeArray(1, 4 * bto.MB)
        other.gpu_bytes_used = 3 * bto.MB
        other.array = FakeArray(2, 3 * bto.MB)
        with patch.object(bto, '_pool_used_bytes', side_effect=lambda dev: 10 * bto.MB):
            report = gpu_mem_report([parent, other, cpu_obj], names={id(parent): 'named-parent'})

        self.assertIn('GPU 0:', report)
        self.assertIn('GPU 1:', report)
        self.assertIn('named-parent', report)
        self.assertIn('sum of totals 6.00 MB', report)
        self.assertIn('unattributed 4.00 MB', report)
        self.assertNotIn('GPU -1:', report)

    def test_gpu_bytes_held_returns_zero_when_cupy_is_unavailable(self):
        obj = self.make_obj()
        with patch.object(bto, 'cp', None):
            self.assertEqual(obj.gpu_bytes_held(), 0)

    def test_print_memory_usage_logs_object_name_and_own_total(self):
        obj = self.make_obj()
        obj.name = 'memory-owner'
        obj.gpu_bytes_used = 4 * bto.MB
        obj.gpu_bytes_children = 1 * bto.MB
        with self.assertLogs(obj.logger.logger, logging.INFO) as logs:
            obj.printMemUsage()
        self.assertIn('memory-owner', logs.output[0])
        self.assertIn('4.00 MB', logs.output[0])
        self.assertIn('own 3.00 MB', logs.output[0])

    def test_subclass_init_and_setup_methods_are_wrapped_with_original_signature(self):
        class Derived(BaseTimeObj):
            def __init__(self, value):
                self.value = value

            def setup(self, value=1):
                self.setup_value = value

        self.assertEqual(list(signature(Derived.__init__).parameters), ['self', 'value'])
        self.assertEqual(list(signature(Derived.setup).parameters), ['self', 'value'])
        obj = Derived(9)
        obj.setup(3)
        self.assertEqual(obj.value, 9)
        self.assertEqual(obj.setup_value, 3)

    def test_top_level_registry_mark_filters_and_weakly_holds_objects(self):
        first = self.make_obj()
        first.startMemUsageCount()
        first.stopMemUsageCount()
        mark = mem_registry_mark()
        later = self.make_obj()
        later.startMemUsageCount()
        later.stopMemUsageCount()
        self.assertEqual(top_level_objs(since=mark), [later])

    def test_nested_start_stop_only_samples_outermost_call(self):
        obj = self.make_obj()
        obj.startMemUsageCount()
        self.pool[0] += 3
        obj.startMemUsageCount()
        self.pool[0] += 4
        obj.stopMemUsageCount()
        self.assertEqual(obj.gpu_bytes_used, 0)
        self.assertEqual(self.stack, [obj])
        obj.stopMemUsageCount()
        self.assertEqual(obj.gpu_bytes_used, 7)
        self.assertEqual(self.stack, [])


class DummyLoopObject:
    def __init__(self, name='dummy', ready=True):
        self.name = name
        self.ready = ready
        self.inputs_changed = False
        self.remote_outputs = {}
        self.count_starts = 0
        self.count_stops = 0
        self.triggers = 0

    def send_outputs(self, **kwargs):
        pass

    def startMemUsageCount(self):
        self.count_starts += 1

    def stopMemUsageCount(self):
        self.count_stops += 1

    def setup(self):
        pass

    def sanity_check(self):
        pass

    def check_ready(self, t):
        self.inputs_changed = self.ready

    def trigger(self):
        self.triggers += 1

    def post_trigger(self):
        self.inputs_changed = False

    def finalize(self):
        pass


class TestLoopMemoryTracking(unittest.TestCase):
    def test_setup_exception_balances_memory_count(self):
        class FailingSetup(DummyLoopObject):
            def setup(self):
                raise RuntimeError('setup failed')

        obj = FailingSetup()
        loop = LoopControl()
        loop.add(obj, 0)
        with self.assertRaisesRegex(RuntimeError, 'setup failed'):
            loop.start(run_time=0.001, dt=0.001)
        self.assertEqual((obj.count_starts, obj.count_stops), (1, 1))

    def test_counts_until_first_trigger_then_reports_once(self):
        obj = DummyLoopObject()
        loop = LoopControl()
        report = MagicMock()
        loop.mem_report = report
        loop.add(obj, 0)
        loop.run(run_time=0.003, dt=0.001)

        titles = [call.args[0] for call in report.call_args_list]
        self.assertEqual(titles, ['after setup', 'after the first trigger of all objects'])
        self.assertEqual(obj.triggers, 3)
        self.assertEqual(obj.count_starts, 4)  # setup, readiness, trigger and post-trigger
        self.assertEqual(obj.count_stops, 4)

    def test_never_triggered_object_is_reported_at_finish(self):
        obj = DummyLoopObject(name='never', ready=False)
        loop = LoopControl()
        report = MagicMock()
        loop.mem_report = report
        loop.add(obj, 0)
        loop.run(run_time=0.001, dt=0.001)

        self.assertEqual([call.args[0] for call in report.call_args_list],
                         ['after setup', 'at the end (never triggered: never)'])
        self.assertEqual(obj.triggers, 0)
        self.assertFalse(loop._mem_pending)

    def test_count_wrapper_always_stops_on_exception_and_untracked_calls_bypass_it(self):
        obj = DummyLoopObject()
        loop = LoopControl()
        loop._mem_pending.add(obj)
        with self.assertRaisesRegex(RuntimeError, 'failed'):
            loop._counted(obj, lambda: (_ for _ in ()).throw(RuntimeError('failed')))
        self.assertEqual((obj.count_starts, obj.count_stops), (1, 1))

        loop._mem_pending.clear()
        self.assertEqual(loop._counted(obj, lambda: 'result'), 'result')
        self.assertEqual((obj.count_starts, obj.count_stops), (1, 1))

    def test_empty_pending_set_reports_first_trigger_stage(self):
        loop = LoopControl()
        report = MagicMock()
        loop.mem_report = report
        loop.run(run_time=0.001, dt=0.001)
        self.assertEqual([call.args[0] for call in report.call_args_list],
                         ['after setup', 'after the first trigger of all objects'])

    def test_preroll_completing_first_trigger_reports_without_main_iteration(self):
        obj = DummyLoopObject()
        loop = LoopControl()
        report = MagicMock()
        loop.mem_report = report
        loop.add(obj, 0)
        loop.run(run_time=0, dt=0.001, t0=0.002, preroll_objs=['dummy'])
        self.assertEqual([call.args[0] for call in report.call_args_list],
                         ['after setup', 'after the first trigger of all objects'])
        self.assertEqual(obj.triggers, 2)


class TestSimulationMemoryReporting(unittest.TestCase):
    def test_logs_only_nonempty_simulation_report(self):
        sim = simul_module.Simul.__new__(simul_module.Simul)
        sim._mem_mark = 7
        obj = object()
        sim.objs = {'processing': obj}
        sim.logger = MagicMock()
        with patch.object(simul_module, 'top_level_objs', return_value=[obj]) as top_objs, \
             patch.object(simul_module, 'gpu_mem_report', return_value='GPU report') as report:
            sim.gpu_mem_report('after setup')
        top_objs.assert_called_once_with(since=7)
        report.assert_called_once_with([obj], {id(obj): 'processing'})
        sim.logger.info.assert_called_once_with('GPU memory after setup:\nGPU report')

        sim.logger.info.reset_mock()
        with patch.object(simul_module, 'top_level_objs', return_value=[]), \
             patch.object(simul_module, 'gpu_mem_report', return_value=''):
            sim.gpu_mem_report('empty')
        sim.logger.info.assert_not_called()


if __name__ == '__main__':
    unittest.main()
