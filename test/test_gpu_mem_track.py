import pickle
import unittest
from unittest.mock import MagicMock, patch

import specula
specula.init(0)  # Default target device

from specula import cp
import specula.base_time_obj as bto
from specula.base_time_obj import BaseTimeObj
from specula.base_time_obj import gpu_mem_report, mem_registry_mark, top_level_objs
from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj
from specula.loop_control import LoopControl

MB = 1024 * 1024


def alloc(nbytes, device_idx=None):
    if device_idx is None:
        return cp.zeros(nbytes, dtype=cp.uint8)
    with cp.cuda.Device(device_idx):
        return cp.zeros(nbytes, dtype=cp.uint8)


class DataWithArray(BaseDataObj):
    def __init__(self, nbytes, target_device_idx=None):
        super().__init__(target_device_idx=target_device_idx)
        if self.target_device_idx >= 0:
            self.value = alloc(nbytes, self.target_device_idx)


class AllocBeforeSuper(BaseProcessingObj):
    def __init__(self, target_device_idx=None):
        self.early = alloc(2 * MB, target_device_idx if target_device_idx is not None else 0)
        super().__init__(target_device_idx=target_device_idx)
        self.late = alloc(1 * MB)


class AllocInSetup(BaseProcessingObj):
    def setup(self):
        self.before_super = alloc(2 * MB)
        super().setup()
        self.after_super = alloc(1 * MB)


class Parent(BaseProcessingObj):
    '''Builds a nested data object and an output'''
    def __init__(self, target_device_idx=None, child_device_idx=None):
        super().__init__(target_device_idx=target_device_idx)
        self.own = alloc(1 * MB)
        self.child = DataWithArray(2 * MB, target_device_idx=child_device_idx)
        self.outputs['out'] = DataWithArray(3 * MB)


class LazyTrigger(BaseProcessingObj):
    '''Allocates a buffer on the first trigger, and a temporary one on the second'''
    def __init__(self):
        super().__init__()
        self.buffer = None
        self.second = None
        self.ntriggers = 0

    def trigger_code(self):
        self.ntriggers += 1
        if self.buffer is None:
            self.buffer = alloc(2 * MB)
        elif self.second is None:
            self.second = alloc(1 * MB)


class DeviceNotHinted(BaseProcessingObj):
    '''The device is not passed as target_device_idx keyword to __init__'''
    def __init__(self, device_idx):
        self.early = alloc(2 * MB, device_idx)
        super().__init__(target_device_idx=device_idx)
        self.late = alloc(1 * MB)


class NeverTriggered(BaseProcessingObj):
    def checkInputTimes(self):
        return False


class TestDevicesRead(unittest.TestCase):
    '''Runs without a GPU: the memory pools are mocked'''

    def test_reads_only_used_devices_before_init(self):
        # Reading the memory pool of an unused GPU would create a CUDA context on it
        read = MagicMock(return_value=0)
        with patch.object(bto, 'cp', MagicMock()), \
             patch.object(bto, '_pool_used_bytes', read), \
             patch.object(bto, '_mem_count_stack', []), \
             patch.object(bto, '_used_devices', {1}):

            # Before __init__ the device is not known
            with patch.object(bto, 'default_target_device_idx', 0):
                BaseTimeObj.__new__(BaseTimeObj).startMemUsageCount(device_hint=3)
            self.assertEqual({c.args[0] for c in read.call_args_list}, {0, 1, 3})

            read.reset_mock()
            with patch.object(bto, 'default_target_device_idx', -1):
                BaseTimeObj.__new__(BaseTimeObj).startMemUsageCount(device_hint=-1)
            self.assertEqual({c.args[0] for c in read.call_args_list}, {1})


@unittest.skipIf(cp is None, 'GPU memory tracking needs cupy')
class TestGpuMemTrack(unittest.TestCase):

    def test_alloc_before_super_init_is_counted(self):
        obj = AllocBeforeSuper()
        self.assertEqual(obj.gpu_bytes_used, 3 * MB)
        self.assertEqual(obj.gpu_bytes_own, 3 * MB)

    def test_device_not_read_is_counted_from_base_init(self):
        # Device neither hinted nor used yet: counting starts in BaseTimeObj.__init__
        with patch.object(bto, '_used_devices', set()), \
             patch.object(bto, 'default_target_device_idx', -1):
            obj = DeviceNotHinted(0)
        self.assertEqual(obj.gpu_bytes_used, 1 * MB)

    def test_alloc_before_super_setup_is_counted(self):
        obj = AllocInSetup()
        obj.setup()
        self.assertEqual(obj.gpu_bytes_used, 3 * MB)

    def test_nested_objects(self):
        mark = mem_registry_mark()
        parent = Parent()
        self.assertEqual(parent.gpu_bytes_used, 6 * MB)
        self.assertEqual(parent.gpu_bytes_children, 5 * MB)
        self.assertEqual(parent.gpu_bytes_own, 1 * MB)
        self.assertEqual(parent.child.gpu_bytes_used, 2 * MB)

        # Only the parent is a top-level object
        objs = top_level_objs(since=mark)
        self.assertIn(parent, objs)
        self.assertNotIn(parent.child, objs)
        self.assertNotIn(parent.outputs['out'], objs)

    def test_nested_object_counted_later_updates_owner(self):
        parent = Parent()
        child = parent.child
        child.startMemUsageCount()
        child.more = alloc(1 * MB)
        child.stopMemUsageCount()
        self.assertEqual(child.gpu_bytes_used, 3 * MB)
        self.assertEqual(parent.gpu_bytes_used, 7 * MB)
        self.assertEqual(parent.gpu_bytes_own, 1 * MB)

    def test_nested_object_counted_during_owner_count(self):
        parent = Parent()
        child = parent.child
        parent.startMemUsageCount()
        child.startMemUsageCount()
        child.more = alloc(1 * MB)
        child.stopMemUsageCount()
        parent.stopMemUsageCount()
        # Counted once in the parent total, as memory of the child
        self.assertEqual(parent.gpu_bytes_used, 7 * MB)
        self.assertEqual(parent.gpu_bytes_own, 1 * MB)

    def test_start_stop_with_exception(self):
        class Failing(BaseProcessingObj):
            def setup(self):
                super().setup()
                raise RuntimeError('setup failed')

        obj = Failing()
        with self.assertRaises(RuntimeError):
            obj.setup()
        self.assertEqual(obj._mem_count_depth, 0)
        obj.startMemUsageCount()
        obj.x = alloc(1 * MB)
        obj.stopMemUsageCount()
        self.assertEqual(obj.gpu_bytes_used, 1 * MB)

    def test_pickle_nested_object(self):
        parent = Parent()
        parent.child.xp = 0     # modules cannot be pickled, as in send_remote_output()
        copy = pickle.loads(pickle.dumps(parent.child))
        self.assertIsNone(copy.__dict__.get('_mem_owner'))

    def test_gpu_bytes_held(self):
        parent = Parent()
        parent.view = parent.own[:10]                   # same memory as parent.own
        other = DataWithArray(4 * MB)
        parent.other = other                            # another top-level object
        parent.inputs['in'] = MagicMock()
        parent.local_inputs['in'] = other               # inputs are not held
        self.assertEqual(parent.gpu_bytes_held(), 6 * MB)
        self.assertEqual(other.gpu_bytes_held(), 4 * MB)

        # A shared set of pointers counts arrays once
        seen = set()
        self.assertEqual(parent.gpu_bytes_held(seen), 6 * MB)
        parent2 = Parent()
        parent2.shared = parent.own
        self.assertEqual(parent2.gpu_bytes_held(seen), 6 * MB)

    def test_loop_counts_first_trigger(self):
        obj = LazyTrigger()
        obj.name = 'lazy'
        never = NeverTriggered()
        never.name = 'never'
        loop = LoopControl()
        loop.mem_report = MagicMock()
        loop.add(obj, 0)
        loop.add(never, 0)
        loop.run(run_time=0.003, dt=0.001)

        self.assertEqual(obj.ntriggers, 3)
        # The first trigger is counted, the following ones are not
        self.assertEqual(obj.gpu_bytes_used, 2 * MB)

        titles = [call.args[0] for call in loop.mem_report.call_args_list]
        self.assertEqual(titles[0], 'after setup')
        self.assertEqual(titles[1], 'at the end (never triggered: never)')
        self.assertEqual(len(titles), 2)

    def test_loop_reports_after_first_trigger(self):
        obj = LazyTrigger()
        obj.name = 'lazy'
        loop = LoopControl()
        loop.mem_report = MagicMock()
        loop.add(obj, 0)
        loop.run(run_time=0.002, dt=0.001)
        titles = [call.args[0] for call in loop.mem_report.call_args_list]
        self.assertEqual(titles, ['after setup', 'after the first trigger of all objects'])

    def test_gpu_mem_report(self):
        parent = Parent()
        other = DataWithArray(4 * MB)
        cpu_obj = DataWithArray(0, target_device_idx=-1)
        report = gpu_mem_report([parent, other, cpu_obj], names={id(parent): 'my_parent'})
        lines = report.splitlines()
        self.assertEqual(lines[0], '  GPU 0:')
        self.assertIn('my_parent', lines[2])
        self.assertEqual(lines[2].split()[-3:], ['1.00', '6.00', '6.00'])
        self.assertIn('DataWithArray()', lines[3])
        self.assertEqual(lines[3].split()[-3:], ['4.00', '4.00', '4.00'])
        self.assertTrue(lines[4].strip().startswith('sum of totals 10.00 MB'))
        self.assertEqual(len(lines), 5)   # no CPU section

    @unittest.skipIf(cp is None or cp.cuda.runtime.getDeviceCount() < 2, 'needs two GPUs')
    def test_object_on_other_device(self):
        cp.cuda.Device(0).use()
        mark = mem_registry_mark()
        parent = Parent(target_device_idx=1, child_device_idx=0)
        # The child and the output are on device 0: not nested
        self.assertEqual(parent.gpu_bytes_used, 1 * MB)
        self.assertEqual(parent.gpu_bytes_children, 0)
        self.assertEqual(parent.child.gpu_bytes_used, 2 * MB)
        self.assertIn(parent.child, top_level_objs(since=mark))
        self.assertIn(parent.outputs['out'], top_level_objs(since=mark))

        # Counted on the object's device, whatever the current one
        cp.cuda.Device(0).use()
        parent.startMemUsageCount()
        parent.more = alloc(1 * MB, 1)
        cp.cuda.Device(0).use()
        parent.stopMemUsageCount()
        self.assertEqual(parent.gpu_bytes_used, 2 * MB)

        obj = AllocBeforeSuper(target_device_idx=1)
        self.assertEqual(obj.gpu_bytes_used, 3 * MB)
        cp.cuda.Device(0).use()


if __name__ == '__main__':
    unittest.main()
