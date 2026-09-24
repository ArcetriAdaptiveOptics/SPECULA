import os
import tempfile
import unittest

import specula
specula.init(-1)  # Default target device

from specula.tracing import Tracer


class _Obj:
    def __init__(self, name):
        self.name = name
        self.target_device_idx = -1


class TestTracer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.tmpdir.name, 'trace.tsv')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _read_rows(self, filename):
        with open(filename) as f:
            lines = [l.rstrip('\n') for l in f if not l.startswith('#')]
        header = lines[0].split('\t')
        return header, [dict(zip(header, l.split('\t'))) for l in lines[1:]]

    def test_not_recording_by_default(self):
        tracer = Tracer()
        self.assertFalse(tracer.recording)
        obj = _Obj('a')
        tracer.begin(obj, 'trigger')   # must be harmless
        tracer.end(obj, 'trigger')
        self.assertIsNone(tracer.close())

    def test_file_rows_and_summary(self):
        tracer = Tracer()
        tracer.open(self.filename)
        a, b = _Obj('a'), _Obj('b')
        tracer.begin(a, 'setup')
        tracer.end(a, 'setup')
        for i in range(3):
            tracer.begin_iteration(i, i * 0.001)
            for obj in (a, b):
                for phase in ('prepare_trigger', 'trigger', 'post_trigger'):
                    tracer.begin(obj, phase)
                    tracer.end(obj, phase)
            tracer.end_iteration()
        summary = tracer.close()

        header, rows = self._read_rows(self.filename)
        self.assertEqual(header, ['iter', 't_sim_s', 'object', 'class', 'phase', 'start_us', 'dur_us'])
        self.assertEqual(len(rows), 1 + 3 * 2 * 3)
        self.assertEqual(rows[0]['iter'], '-1')
        self.assertEqual(rows[0]['phase'], 'setup')
        self.assertEqual(rows[-1]['iter'], '2')
        self.assertEqual(rows[-1]['t_sim_s'], '0.002000')
        self.assertEqual(rows[-1]['object'], 'b')
        self.assertEqual(rows[-1]['class'], '_Obj')
        self.assertEqual(rows[-1]['phase'], 'post_trigger')

        self.assertIn('Iterations: 3', summary)
        self.assertIn('trigger', summary)
        summary_file = os.path.join(self.tmpdir.name, 'trace.summary.txt')
        with open(summary_file) as f:
            self.assertEqual(f.read(), summary)

    def test_nested_phases(self):
        tracer = Tracer()
        tracer.open(self.filename)
        outer, inner = _Obj('outer'), _Obj('inner')
        tracer.begin_iteration(0, 0.0)
        tracer.begin(outer, 'prepare_trigger')
        tracer.begin(inner, 'prepare_trigger')
        tracer.end(inner, 'prepare_trigger')
        tracer.end(outer, 'prepare_trigger')
        tracer.end_iteration()
        tracer.close()
        _, rows = self._read_rows(self.filename)
        self.assertEqual([r['object'] for r in rows], ['inner', 'outer'])
        self.assertGreaterEqual(float(rows[1]['dur_us']), float(rows[0]['dur_us']))

    def test_rank_suffix(self):
        tracer = Tracer()
        tracer.open(self.filename, rank=3)
        tracer.close()
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir.name, 'trace.rank3.tsv')))
        self.assertTrue(os.path.exists(os.path.join(self.tmpdir.name, 'trace.rank3.summary.txt')))

    def test_skip_iterations(self):
        tracer = Tracer()
        tracer.open(self.filename, skip=2)
        obj = _Obj('a')
        tracer.begin(obj, 'setup')
        tracer.end(obj, 'setup')
        for i in range(5):
            tracer.begin_iteration(i, 0.0)
            tracer.begin(obj, 'trigger')
            tracer.end(obj, 'trigger')
            tracer.end_iteration()
        summary = tracer.close()
        _, rows = self._read_rows(self.filename)
        self.assertEqual([r['iter'] for r in rows], ['-1', '2', '3', '4'])
        self.assertIn('Iterations: 3', summary)
        self.assertIn('skipped iterations: 2', summary)


class TestContextDecorator(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.tmpdir.name, 'trace.tsv')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _rows(self):
        return TestTracer._read_rows(self, self.filename)[1]

    def test_context_manager(self):
        tracer = Tracer()
        tracer.open(self.filename)
        obj = _Obj('a')
        with tracer(obj=obj, phase='interpolation'):
            with tracer('toccd', obj):
                pass
        tracer.close()
        rows = self._rows()
        self.assertEqual([(r['object'], r['phase']) for r in rows],
                         [('a', 'toccd'), ('a', 'interpolation')])

    def test_decorator_uses_self(self):
        tracer = Tracer()

        class Foo(_Obj):
            @tracer('compute')
            def compute(self, x):
                return x + 1

        foo = Foo('foo')
        tracer.open(self.filename)
        self.assertEqual(foo.compute(1), 2)
        self.assertEqual(Foo.compute.__name__, 'compute')
        tracer.close()
        rows = self._rows()
        self.assertEqual([(r['object'], r['class'], r['phase']) for r in rows],
                         [('foo', 'Foo', 'compute')])

    def test_name_only(self):
        tracer = Tracer()

        @tracer('plain_function')
        def f(x):
            return x * 2

        tracer.open(self.filename)
        self.assertEqual(f(3), 6)
        with tracer('section'):
            pass
        tracer.close()
        rows = self._rows()
        self.assertEqual([(r['object'], r['phase']) for r in rows],
                         [('-', 'plain_function'), ('-', 'section')])

    def test_exception_closes_range(self):
        tracer = Tracer()
        tracer.open(self.filename)
        obj = _Obj('a')

        @tracer('bad')
        def bad(self):
            raise ValueError

        with self.assertRaises(ValueError):
            with tracer('section', obj):
                raise ValueError
        with self.assertRaises(ValueError):
            bad(obj)
        self.assertEqual(tracer._stack, [])
        tracer.close()
        self.assertEqual([r['phase'] for r in self._rows()], ['section', 'bad'])

    def test_nvtx_ranges(self):
        from unittest.mock import MagicMock
        tracer = Tracer()
        tracer._nvtx = MagicMock()
        with tracer('trigger', _Obj('a')):
            pass
        with tracer('section', color_id=7):
            pass
        push = tracer._nvtx.RangePush.call_args_list
        self.assertEqual([c.args for c in push], [('a.trigger', 3), ('section', 7)])
        self.assertEqual(tracer._nvtx.RangePop.call_count, 2)


# Index of the current (fake) device, changed only by _FakeDevice as a context manager
_current_device = [None]


class _FakeEvent:
    created = 0

    def __init__(self):
        _FakeEvent.created += 1
        self.stream = None
        self.device = None
        self.done = False
        self.synchronized = False

    def record(self, stream=None):
        self.stream = stream
        self.device = _current_device[0]
        self.done = False

    def synchronize(self):
        self.synchronized = True
        self.done = True


class _FakeDevice:
    # No use(): the tracer must not change the current device
    def __init__(self, idx=0):
        self.idx = idx
        self._prev = []

    def __enter__(self):
        self._prev.append(_current_device[0])
        _current_device[0] = self.idx
        return self

    def __exit__(self, *exc):
        _current_device[0] = self._prev.pop()
        return False


class _FakeStream:
    def __init__(self):
        self.capturing = False

    def is_capturing(self):
        return self.capturing


def _make_fake_cp(elapsed_ms=0.25):
    from types import SimpleNamespace
    current_stream = _FakeStream()
    runtime = SimpleNamespace(n_sync=0, sync_devices=[])

    def deviceSynchronize():
        runtime.n_sync += 1
        runtime.sync_devices.append(_current_device[0])
    runtime.deviceSynchronize = deviceSynchronize
    cuda = SimpleNamespace(
        Event=_FakeEvent,
        get_current_stream=lambda: current_stream,
        get_elapsed_time=lambda start, end: elapsed_ms,
        runtime=runtime,
    )
    return SimpleNamespace(cuda=cuda), current_stream


class _GpuObj(_Obj):
    def __init__(self, name, with_graph=False):
        super().__init__(name)
        self.target_device_idx = 0
        self._target_device = _FakeDevice()
        self.stream = object()
        self.cuda_graph = object() if with_graph else None


class TestGpuEvents(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.filename = os.path.join(self.tmpdir.name, 'trace.tsv')

    def tearDown(self):
        self.tmpdir.cleanup()

    def _run(self, tracer, objs, n_iter, complete=True):
        for i in range(n_iter):
            tracer.begin_iteration(i, 0.0)
            for obj in objs:
                tracer.begin(obj, 'trigger')
                tracer.end(obj, 'trigger')
                if complete:
                    for entry in tracer._gpu_pending:
                        entry[5].done = True
            tracer.end_iteration()

    def _rows(self):
        with open(self.filename) as f:
            lines = [l.rstrip('\n').split('\t') for l in f if not l.startswith('#')]
        return lines[1:]

    def test_gpu_rows_and_streams(self):
        from unittest.mock import patch
        fake_cp, current_stream = _make_fake_cp(elapsed_ms=0.25)
        with patch('specula.tracing.cp', fake_cp):
            tracer = Tracer()
            tracer.open(self.filename, gpu_events=True)
            graph_obj = _GpuObj('graph', with_graph=True)
            plain_obj = _GpuObj('plain')
            cpu_obj = _Obj('cpu')
            tracer.begin_iteration(0, 0.0)
            tracer.begin(graph_obj, 'trigger')
            tracer.begin(plain_obj, 'trigger')
            tracer.end(plain_obj, 'trigger')
            tracer.end(graph_obj, 'trigger')
            streams = [(e[2].name, e[4].stream, e[5].stream) for e in tracer._gpu_pending]
            tracer.begin(cpu_obj, 'trigger')
            tracer.end(cpu_obj, 'trigger')
            tracer.end_iteration()
            summary = tracer.close()

        self.assertIn(('graph', graph_obj.stream, graph_obj.stream), streams)
        self.assertIn(('plain', current_stream, current_stream), streams)
        gpu_rows = [r for r in self._rows() if r[4] == 'trigger_gpu']
        self.assertEqual(sorted(r[2] for r in gpu_rows), ['graph', 'plain'])
        self.assertTrue(all(r[6] == '250.0' for r in gpu_rows))
        self.assertIn('trigger_gpu', summary)
        self.assertIn('GPU events: True', summary)

    def test_events_are_reused(self):
        from unittest.mock import patch
        fake_cp, _ = _make_fake_cp()
        with patch('specula.tracing.cp', fake_cp):
            _FakeEvent.created = 0
            tracer = Tracer()
            tracer.open(self.filename, gpu_events=True)
            self._run(tracer, [_GpuObj('a'), _GpuObj('b')], n_iter=20)
            tracer.close()
        # Two events per object, reused across iterations
        self.assertLessEqual(_FakeEvent.created, 4)

    def test_lagged_wait(self):
        from unittest.mock import patch
        from specula.tracing import GPU_EVENTS_LAG
        fake_cp, _ = _make_fake_cp()
        with patch('specula.tracing.cp', fake_cp):
            tracer = Tracer()
            tracer.open(self.filename, gpu_events=True)
            obj = _GpuObj('a')
            # Events never complete by themselves: the tracer waits only
            # for entries older than GPU_EVENTS_LAG iterations
            self._run(tracer, [obj], n_iter=GPU_EVENTS_LAG + 3, complete=False)
            self.assertEqual(len(tracer._gpu_pending), GPU_EVENTS_LAG)
            self.assertTrue(all(not e[5].synchronized for e in tracer._gpu_pending))
            tracer.close()
            self.assertEqual(tracer._gpu_pending, [])
        gpu_rows = [r for r in self._rows() if r[4] == 'trigger_gpu']
        self.assertEqual([r[0] for r in gpu_rows], [str(i) for i in range(GPU_EVENTS_LAG + 3)])

    def test_no_events_during_setup_or_skip(self):
        from unittest.mock import patch
        fake_cp, _ = _make_fake_cp()
        with patch('specula.tracing.cp', fake_cp):
            tracer = Tracer()
            tracer.open(self.filename, gpu_events=True, skip=2)
            obj = _GpuObj('a')
            tracer.begin(obj, 'trigger')   # iteration -1, e.g. setup-time trigger
            tracer.end(obj, 'trigger')
            self.assertEqual(tracer._gpu_pending, [])
            self._run(tracer, [obj], n_iter=4)
            tracer.close()
        gpu_rows = [r for r in self._rows() if r[4] == 'trigger_gpu']
        self.assertEqual([r[0] for r in gpu_rows], ['2', '3'])


class TestSync(unittest.TestCase):

    def test_no_sync_during_capture(self):
        from unittest.mock import patch
        fake_cp, stream = _make_fake_cp()
        with tempfile.TemporaryDirectory() as tmpdir, patch('specula.tracing.cp', fake_cp):
            tracer = Tracer()
            tracer.open(os.path.join(tmpdir, 'trace.tsv'), sync=True)
            obj = _GpuObj('a')
            with tracer('toccd', obj):
                pass
            self.assertEqual(fake_cp.cuda.runtime.n_sync, 1)
            stream.capturing = True
            with tracer('toccd', obj):
                pass
            self.assertEqual(fake_cp.cuda.runtime.n_sync, 1)
            tracer.close()

    def test_current_device_unchanged(self):
        from unittest.mock import patch
        fake_cp, _ = _make_fake_cp()
        with tempfile.TemporaryDirectory() as tmpdir, patch('specula.tracing.cp', fake_cp):
            tracer = Tracer()
            tracer.open(os.path.join(tmpdir, 'trace.tsv'), sync=True, gpu_events=True)
            obj = _GpuObj('a')
            obj._target_device = _FakeDevice(1)
            _current_device[0] = 0
            try:
                tracer.begin_iteration(0, 0.0)
                with tracer('trigger', obj):
                    self.assertEqual(_current_device[0], 0)
                self.assertEqual(_current_device[0], 0)
                tracer.end_iteration()
                ev_start, ev_end = tracer._gpu_pending[0][4:6]
                self.assertEqual((ev_start.device, ev_end.device), (1, 1))
                self.assertEqual(fake_cp.cuda.runtime.sync_devices, [1])
                tracer.close()
            finally:
                _current_device[0] = None
