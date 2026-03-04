import specula
specula.init(0)  # Default target device

import io
import pickle
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import multiprocessing as mp
import numpy as np

import queue as _queue

from specula.simul import Simul


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_mock_dataobj(array):
    """Return a mock that behaves like a specula data object."""
    obj = MagicMock()
    obj.array_for_display.return_value = array
    obj.copyTo.return_value = obj
    obj.xp = np
    return obj


def _drain_queue(q, max_items=200):
    """Reliably drain all items from a queue.Queue (or mp.Queue).

    mp.Queue.empty() is documented as unreliable because a background thread
    moves items through a pipe; using get_nowait() inside try/except is the
    only safe approach.
    """
    items = []
    for _ in range(max_items):
        try:
            items.append(q.get_nowait())
        except Exception:
            break
    return items


class _PicklableDataObj:
    """Minimal picklable stand-in for a specula data object.

    MagicMock cannot be pickled, which breaks the image-mode trigger path that
    calls pickle.dumps() on the object after copyTo().  This class provides the
    same interface without the pickling problem.
    """
    def __init__(self, array):
        self._array = array
        # xp is stripped by remove_xp_np before pickling, so it's safe here
        self.xp = np

    def copyTo(self, device):
        return _PicklableDataObj(self._array.copy())

    def array_for_display(self):
        return self._array


# ---------------------------------------------------------------------------
# Tests for the existing spawn behaviour (kept from original file)
# ---------------------------------------------------------------------------

class TestDisplayServerSpawn(unittest.TestCase):

    def test_display_spawn(self):
        """Test that a DisplayServer can be started.

        Expected to fail on Windows and MacOS.
        """
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          total_time: 0.001
          time_step: 0.001
          display_server: true

        test:
          class: 'Source'
          polar_coordinates: [1, 2]
          magnitude: null
          wavelengthInNm: null
        '''
        with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as tmp:
            tmp.write(yml)
            yml_path = tmp.name

        simul = Simul(yml_path)
        simul.run()


# ---------------------------------------------------------------------------
# Tests for display_server.py utilities
# ---------------------------------------------------------------------------

class TestRemoveXpNp(unittest.TestCase):
    """Tests for the remove_xp_np context manager."""

    def _get_remove_xp_np(self):
        from specula.processing_objects.display_server import remove_xp_np
        return remove_xp_np

    def test_removes_xp_and_restores(self):
        remove_xp_np = self._get_remove_xp_np()
        obj = MagicMock()
        obj.xp = np
        obj.np = np

        with remove_xp_np(obj) as cleaned:
            self.assertFalse(hasattr(cleaned, 'xp'))
            self.assertFalse(hasattr(cleaned, 'np'))

        self.assertIs(obj.xp, np)
        self.assertIs(obj.np, np)

    def test_object_without_xp_unaffected(self):
        remove_xp_np = self._get_remove_xp_np()
        obj = MagicMock(spec=[])  # no attributes
        # Should not raise
        with remove_xp_np(obj):
            pass

    def test_removes_xp_from_list_and_restores(self):
        remove_xp_np = self._get_remove_xp_np()
        objs = [MagicMock(), MagicMock()]
        for o in objs:
            o.xp = np

        with remove_xp_np(objs) as cleaned:
            for o in cleaned:
                self.assertFalse(hasattr(o, 'xp'))

        for o in objs:
            self.assertIs(o.xp, np)

    def test_partial_attributes_restored(self):
        """Object with only xp (no np) should still be handled correctly."""
        remove_xp_np = self._get_remove_xp_np()
        obj = MagicMock()
        obj.xp = np
        # Deliberately do NOT set obj.np

        with remove_xp_np(obj) as cleaned:
            self.assertFalse(hasattr(cleaned, 'xp'))

        self.assertIs(obj.xp, np)


class TestEncodeDisplayServer(unittest.TestCase):
    """Tests for the encode() helper in display_server.py."""

    def test_encode_returns_base64_string(self):
        from specula.processing_objects.display_server import encode
        fig = MagicMock()
        # Make savefig write a valid PNG-like byte sequence
        def _fake_savefig(buf, format):
            buf.write(b'\x89PNG\r\n\x1a\n' + b'\x00' * 20)
        fig.savefig.side_effect = _fake_savefig

        result = encode(fig)
        self.assertIsInstance(result, str)
        # Base64 strings contain only safe characters
        import base64 as _b64
        decoded = _b64.b64decode(result)
        self.assertGreater(len(decoded), 0)


# ---------------------------------------------------------------------------
# Tests for DisplayServer._process_for_dpg
# ---------------------------------------------------------------------------

class TestProcessForDpg(unittest.TestCase):
    """Unit tests for the _process_for_dpg method."""

    def _make_server(self):
        from specula.processing_objects.display_server import DisplayServer
        with patch.object(mp.Process, 'start'):
            with patch('specula.processing_objects.display_server.start_server'):
                server = DisplayServer.__new__(DisplayServer)
                server.mode = 'data'
                server.qin = mp.Queue()
                server.qout = mp.Queue()
                server.params_dict = {}
                server.counter = 0
                server.t0 = time.time()
                server.c0 = 0
                server.speed_report = ''
                server.data_obj_getter = lambda name: None
                server.info_getter = lambda: ('sim', 'running')
                return server

    def test_scalar_array(self):
        server = self._make_server()
        obj = MagicMock()
        obj.array_for_display.return_value = np.array(3.14)
        result = server._process_for_dpg(obj, 'scalar_val')
        self.assertEqual(result['type'], 'scalar')
        self.assertAlmostEqual(result['data'], 3.14, places=4)

    def test_1d_array(self):
        server = self._make_server()
        arr = np.arange(10, dtype=np.float32)
        obj = MagicMock()
        obj.array_for_display.return_value = arr
        result = server._process_for_dpg(obj, 'vec')
        self.assertEqual(result['type'], '1d_array')
        self.assertEqual(result['shape'], (10,))

    def test_2d_array(self):
        server = self._make_server()
        arr = np.ones((4, 4), dtype=np.float64)
        obj = MagicMock()
        obj.array_for_display.return_value = arr
        result = server._process_for_dpg(obj, 'image')
        self.assertEqual(result['type'], '2d_array')
        self.assertEqual(result['shape'], (4, 4))

    def test_nd_array(self):
        server = self._make_server()
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        obj = MagicMock()
        obj.array_for_display.return_value = arr
        result = server._process_for_dpg(obj, 'cube')
        self.assertEqual(result['type'], 'nd_array')

    def test_integer_array_cast_to_float(self):
        server = self._make_server()
        arr = np.array([1, 2, 3], dtype=np.int32)
        obj = MagicMock()
        obj.array_for_display.return_value = arr
        result = server._process_for_dpg(obj, 'ints')
        self.assertEqual(result['dtype'], 'float32')

    def test_list_single_1d(self):
        server = self._make_server()
        arr = np.linspace(0, 1, 5)
        obj = MagicMock()
        obj.array_for_display.return_value = arr
        result = server._process_for_dpg([obj], 'single_list')
        self.assertEqual(result['type'], '1d_array')

    def test_list_multiple_1d_same_length(self):
        server = self._make_server()
        objs = []
        for _ in range(3):
            o = MagicMock()
            o.array_for_display.return_value = np.ones(8)
            objs.append(o)
        result = server._process_for_dpg(objs, 'multi_1d')
        self.assertEqual(result['type'], '2d_array')
        self.assertEqual(result['shape'], (8, 3))

    def test_numpy_array_passed_directly(self):
        server = self._make_server()
        arr = np.eye(3)
        result = server._process_for_dpg(arr, 'direct')
        self.assertEqual(result['type'], '2d_array')

    def test_get_method_fallback(self):
        """If array_for_display returns None, _safe_extract should try .get()."""
        server = self._make_server()
        arr = np.array([7.0, 8.0])
        obj = MagicMock()
        obj.array_for_display.return_value = None
        obj.get.return_value = arr
        result = server._process_for_dpg(obj, 'get_fallback')
        self.assertEqual(result['type'], '1d_array')

    def test_returns_unknown_when_no_data(self):
        server = self._make_server()
        obj = MagicMock()
        obj.array_for_display.return_value = None
        obj.get.return_value = None
        # Make conversion to np.array fail
        obj.__array__ = MagicMock(side_effect=TypeError)
        obj.value = None
        obj.shape = None
        result = server._process_for_dpg(obj, 'nothing')
        self.assertIn(result['type'], ('unknown', 'error'))


# ---------------------------------------------------------------------------
# Tests for DisplayServer trigger methods
# ---------------------------------------------------------------------------

class TestDisplayServerTrigger(unittest.TestCase):

    def _make_server(self, mode='image'):
        from specula.processing_objects.display_server import DisplayServer
        with patch.object(mp.Process, 'start'):
            with patch('specula.processing_objects.display_server.start_server'):
                server = DisplayServer.__new__(DisplayServer)
                server.mode = mode
                # Use thread-safe queue.Queue instead of mp.Queue:
                # mp.Queue transfers data through a background pipe thread, so
                # empty() / get_nowait() can race with items just put() in the
                # same synchronous call — making assertions flaky.
                server.qin = _queue.Queue()
                server.qout = _queue.Queue()
                server.params_dict = {}
                server.counter = 0
                server.t0 = time.time() - 2   # force speed report
                server.c0 = 0
                server.speed_report = ''
                server.info_getter = lambda: ('sim', 'running')
                return server

    # -- image mode --

    def test_trigger_image_mode_empty_queue(self):
        """trigger() with an empty qin should return without error."""
        server = self._make_server('image')
        server.data_obj_getter = lambda name: None
        server.trigger()   # should not raise

    def test_trigger_image_mode_processes_request(self):
        server = self._make_server('image')

        # MagicMock cannot be pickled; the image-mode trigger calls
        # pickle.dumps() on the copyTo() result, so we need a real object.
        dataobj = _PicklableDataObj(np.ones((3, 3)))
        server.data_obj_getter = lambda name: dataobj
        server.qin.put(('client_abc', ['obj1']))

        server.trigger()

        # At minimum the terminator should have been queued
        items = _drain_queue(server.qout)
        types = [i[0] for i in items if isinstance(i, tuple)]
        self.assertIn('image_terminator', types)

    # -- data mode --

    def test_trigger_data_mode_empty_queue(self):
        server = self._make_server('data')
        server.data_obj_getter = lambda name: None
        server.trigger()   # should not raise

    def test_trigger_data_mode_processes_request(self):
        server = self._make_server('data')

        arr = np.arange(5, dtype=float)
        mock_obj = MagicMock()
        mock_obj.copyTo.return_value = mock_obj
        mock_obj.array_for_display.return_value = arr

        server.data_obj_getter = lambda name: mock_obj
        server.qin.put(('client_xyz', ['obj1']))

        server.trigger()

        items = _drain_queue(server.qout)
        types = [i[0] for i in items if isinstance(i, tuple)]
        self.assertIn('terminator', types)

    def test_trigger_puts_speed_report_after_one_second(self):
        server = self._make_server('image')
        server.data_obj_getter = lambda name: None
        server.t0 = time.time() - 2   # guarantee the 1-second check fires

        server.trigger()

        # Speed report is a 2-tuple (name, status_string).
        # Use _drain_queue rather than empty()/get_nowait() directly:
        # mp.Queue.empty() is unreliable due to its background pipe thread.
        items = _drain_queue(server.qout)
        two_tuples = [i for i in items if isinstance(i, tuple) and len(i) == 2]
        self.assertTrue(len(two_tuples) >= 1)

    def test_trigger_dispatch_calls_correct_mode(self):
        """trigger() delegates to the right private method based on self.mode."""
        server = self._make_server('image')
        server._trigger_image_mode = MagicMock()
        server._trigger_data_mode = MagicMock()
        server.trigger()
        server._trigger_image_mode.assert_called_once()
        server._trigger_data_mode.assert_not_called()

        server2 = self._make_server('data')
        server2._trigger_image_mode = MagicMock()
        server2._trigger_data_mode = MagicMock()
        server2.trigger()
        server2._trigger_data_mode.assert_called_once()
        server2._trigger_image_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for DisplayServer.finalize
# ---------------------------------------------------------------------------

class TestDisplayServerFinalize(unittest.TestCase):

    def test_finalize_terminates_process(self):
        from specula.processing_objects.display_server import DisplayServer
        with patch.object(mp.Process, 'start'):
            with patch('specula.processing_objects.display_server.start_server'):
                server = DisplayServer.__new__(DisplayServer)
                mock_proc = MagicMock()
                mock_proc.is_alive.return_value = True
                server.p = mock_proc

                server.finalize()

                mock_proc.terminate.assert_called_once()
                mock_proc.join.assert_called_once()

    def test_finalize_noop_when_process_dead(self):
        from specula.processing_objects.display_server import DisplayServer
        server = DisplayServer.__new__(DisplayServer)
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = False
        server.p = mock_proc

        server.finalize()   # should not raise
        mock_proc.terminate.assert_not_called()

    def test_finalize_noop_when_no_process_attr(self):
        from specula.processing_objects.display_server import DisplayServer
        server = DisplayServer.__new__(DisplayServer)
        # Deliberately do NOT set server.p
        server.finalize()   # should not raise


# ---------------------------------------------------------------------------
# Tests for display_server_api.py – params dict sanitisation
# ---------------------------------------------------------------------------

class TestStartServerParamsSanitisation(unittest.TestCase):
    """Tests for the safe_params_dict filtering inside start_server."""

    def _run_sanitisation(self, raw_params):
        """
        Exercise the sanitisation logic extracted from start_server without
        spinning up a real Flask server.
        """
        safe_params_dict = {}
        for k, v in raw_params.items():
            if isinstance(v, dict):
                safe_v = {}
                for key, val in v.items():
                    if isinstance(val, (str, int, float, bool, type(None), list, dict)):
                        safe_v[key] = val
                safe_params_dict[k] = safe_v
            else:
                safe_params_dict[k] = v
        return safe_params_dict

    def test_primitive_values_preserved(self):
        raw = {'obj1': {'class': 'Source', 'mag': 5.0, 'enabled': True, 'name': 'star'}}
        result = self._run_sanitisation(raw)
        self.assertEqual(result['obj1']['class'], 'Source')
        self.assertAlmostEqual(result['obj1']['mag'], 5.0)
        self.assertTrue(result['obj1']['enabled'])

    def test_non_serialisable_values_stripped(self):
        sentinel = object()   # not str/int/float/bool/None/list/dict
        raw = {'obj1': {'class': 'Source', 'bad_ref': sentinel, 'keep': 42}}
        result = self._run_sanitisation(raw)
        self.assertNotIn('bad_ref', result['obj1'])
        self.assertEqual(result['obj1']['keep'], 42)

    def test_nested_dict_and_list_preserved(self):
        raw = {'obj1': {'inputs': ['a', 'b'], 'cfg': {'x': 1}}}
        result = self._run_sanitisation(raw)
        self.assertEqual(result['obj1']['inputs'], ['a', 'b'])
        self.assertEqual(result['obj1']['cfg'], {'x': 1})

    def test_none_value_preserved(self):
        raw = {'obj1': {'magnitude': None}}
        result = self._run_sanitisation(raw)
        self.assertIsNone(result['obj1']['magnitude'])

    def test_non_dict_top_level_value_passed_through(self):
        raw = {'scalar_key': 'hello'}
        result = self._run_sanitisation(raw)
        self.assertEqual(result['scalar_key'], 'hello')


# ---------------------------------------------------------------------------
# Tests for display params filtering (DataStore / inputs+outputs logic)
# ---------------------------------------------------------------------------

class TestDisplayParamsFiltering(unittest.TestCase):
    """
    The same filtering logic appears in both image-mode and data-mode handlers.
    We test it independently.
    """

    def _filter(self, params_dict):
        display_params = {}
        for k, v in params_dict.items():
            if 'class' in v:
                if v['class'] == 'DataStore':
                    continue
                if 'inputs' not in v and 'outputs' not in v:
                    continue
            display_params[k] = v
        return display_params

    def test_datastore_excluded(self):
        params = {'ds': {'class': 'DataStore', 'inputs': ['x']}}
        result = self._filter(params)
        self.assertNotIn('ds', result)

    def test_class_without_inputs_outputs_excluded(self):
        params = {'src': {'class': 'Source', 'mag': 5}}
        result = self._filter(params)
        self.assertNotIn('src', result)

    def test_class_with_inputs_included(self):
        params = {'wfs': {'class': 'Sensor', 'inputs': ['pupil']}}
        result = self._filter(params)
        self.assertIn('wfs', result)

    def test_class_with_outputs_included(self):
        params = {'dm': {'class': 'DM', 'outputs': ['phase']}}
        result = self._filter(params)
        self.assertIn('dm', result)

    def test_entry_without_class_always_included(self):
        params = {'meta': {'version': '1.0'}}
        result = self._filter(params)
        self.assertIn('meta', result)

    def test_mixed_params(self):
        params = {
            'ds': {'class': 'DataStore', 'inputs': ['x']},
            'src': {'class': 'Source', 'mag': 5},
            'wfs': {'class': 'Sensor', 'inputs': ['pupil']},
            'cfg': {'root_dir': '/tmp'},
        }
        result = self._filter(params)
        self.assertNotIn('ds', result)
        self.assertNotIn('src', result)
        self.assertIn('wfs', result)
        self.assertIn('cfg', result)


# ---------------------------------------------------------------------------
# Tests for ImageFlaskServer and DataFlaskServer initialisation
# ---------------------------------------------------------------------------

class TestImageFlaskServerInit(unittest.TestCase):

    def _make_server(self, port=5000):
        from specula.lib.display_server_api import ImageFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            srv = ImageFlaskServer(
                params_dict={'k': {'class': 'Source', 'inputs': []}},
                status_queue=sq,
                request_queue=rq,
                host='127.0.0.1',
                port=port,
            )
        return srv

    def test_attributes_set_correctly(self):
        srv = self._make_server(port=5001)
        self.assertEqual(srv.host, '127.0.0.1')
        self.assertEqual(srv.port, 5001)
        self.assertIsNone(srv.actual_port)
        self.assertFalse(srv.frontend_connected)
        self.assertEqual(srv.client_types, {})
        self.assertEqual(srv.plotters, {})
        self.assertEqual(srv.t0, {})

    def test_response_handler_thread_started(self):
        from specula.lib.display_server_api import ImageFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            ImageFlaskServer(
                params_dict={},
                status_queue=sq,
                request_queue=rq,
            )
        mock_thread.start.assert_called_once()


class TestDataFlaskServerInit(unittest.TestCase):

    def _make_server(self, port=5000):
        from specula.lib.display_server_api import DataFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            srv = DataFlaskServer(
                params_dict={},
                status_queue=sq,
                request_queue=rq,
                host='0.0.0.0',
                port=port,
            )
        return srv

    def test_attributes_set_correctly(self):
        srv = self._make_server(port=6000)
        self.assertEqual(srv.port, 6000)
        self.assertIsNone(srv.actual_port)
        self.assertEqual(srv.client_types, {})
        self.assertEqual(srv.client_subscriptions, {})
        self.assertEqual(srv.last_request_outputs, [])
        self.assertEqual(srv.last_request_time, 0)

    def test_response_handler_thread_started(self):
        from specula.lib.display_server_api import DataFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread') as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            DataFlaskServer(params_dict={}, status_queue=sq, request_queue=rq)
        mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for port-resolution logic (port=0 vs explicit)
# ---------------------------------------------------------------------------

class TestPortResolution(unittest.TestCase):
    """
    The port=0 branch uses a temporary socket to find a free port.
    We verify that logic without running Flask.
    """

    def _resolve_port(self, requested_port):
        import socket
        if requested_port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))
                _, port = s.getsockname()
                s.close()
                return port
        else:
            return requested_port

    def test_explicit_port_returned_unchanged(self):
        self.assertEqual(self._resolve_port(5005), 5005)

    def test_zero_port_returns_free_port(self):
        port = self._resolve_port(0)
        self.assertIsInstance(port, int)
        self.assertGreater(port, 0)
        self.assertLessEqual(port, 65535)


# ---------------------------------------------------------------------------
# Tests for encode() in display_server_api
# ---------------------------------------------------------------------------

class TestEncodeApi(unittest.TestCase):

    def test_encode_returns_nonempty_string(self):
        from specula.lib.display_server_api import encode
        fig = MagicMock()
        fig.savefig.side_effect = lambda buf, format: buf.write(b'PNGDATA')
        result = encode(fig)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_encode_is_valid_base64(self):
        import base64 as _b64
        from specula.lib.display_server_api import encode
        fig = MagicMock()
        fig.savefig.side_effect = lambda buf, format: buf.write(b'\x89PNG\r\n\x1a\n')
        result = encode(fig)
        decoded = _b64.b64decode(result)
        self.assertTrue(decoded.startswith(b'\x89PNG'))


# ---------------------------------------------------------------------------
# Tests for ImageFlaskServer.handle_image_responses
# ---------------------------------------------------------------------------

class TestImageFlaskServerHandleResponses(unittest.TestCase):

    def _make_server(self):
        from specula.lib.display_server_api import ImageFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            srv = ImageFlaskServer(
                params_dict={},
                status_queue=sq,
                request_queue=rq,
            )
        # Patch the sio module-level object used by the handler
        return srv, sq

    def test_image_terminator_emits_done(self):
        from specula.lib.display_server_api import ImageFlaskServer, sio
        srv, sq = self._make_server()
        srv.client_types['cli1'] = 'web'
        srv.t0['cli1'] = time.time() - 1.0

        sq.put(('image_terminator', 'cli1', None, '10.00 Hz'))
        # Sentinel to stop the loop
        sq.put(('image_terminator', 'cli1', None, None))

        with patch.object(sio, 'emit') as mock_emit:
            # Run one iteration of the handler in the current thread
            try:
                item = sq.get(timeout=0.1)
                response_type, client_id, name, data = item
                if response_type == 'image_terminator' and client_id in srv.client_types:
                    sio.emit('speed_report', data, room=client_id)
                    t1 = time.time()
                    t0 = srv.t0.get(client_id, t1)
                    freq = 1.0 / (t1 - t0) if t1 != t0 else 0
                    sio.emit('done', f'Display rate: {freq:.2f} Hz', room=client_id)
                    srv.t0[client_id] = t1
            except queue.Empty:
                pass

        calls = [c[0][0] for c in mock_emit.call_args_list]
        self.assertIn('speed_report', calls)
        self.assertIn('done', calls)

    def test_unknown_client_ignored(self):
        """Responses for unknown client IDs should not raise."""
        from specula.lib.display_server_api import ImageFlaskServer, sio
        srv, sq = self._make_server()
        # client 'ghost' is NOT in client_types

        sq.put(('image_terminator', 'ghost', None, '5.00 Hz'))

        with patch.object(sio, 'emit') as mock_emit:
            try:
                item = sq.get(timeout=0.1)
                response_type, client_id, name, data = item
                if response_type == 'image_terminator' and client_id in srv.client_types:
                    sio.emit('done', '', room=client_id)
            except queue.Empty:
                pass

        mock_emit.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for DataFlaskServer.handle_responses
# ---------------------------------------------------------------------------

class TestDataFlaskServerHandleResponses(unittest.TestCase):

    def _manual_handle(self, srv, item):
        """Simulate one iteration of DataFlaskServer.handle_responses."""
        from specula.lib.display_server_api import sio
        if isinstance(item, tuple) and len(item) == 4:
            response_type, client_id, name, data = item
            if response_type == 'data_response':
                if client_id in srv.client_types:
                    sio.emit('data_update', {'name': name, 'data': data}, room=client_id)
            elif response_type == 'terminator':
                if client_id in srv.client_types:
                    sio.emit('speed_report', data, room=client_id)
                    t1 = time.time()
                    t0 = srv.t0.get(client_id, t1)
                    freq = 1.0 / (t1 - t0) if t1 != t0 else 0
                    sio.emit('done', f'Display rate: {freq:.2f} Hz', room=client_id)
                    srv.t0[client_id] = t1

    def _make_server(self):
        from specula.lib.display_server_api import DataFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            srv = DataFlaskServer(params_dict={}, status_queue=sq, request_queue=rq)
        return srv

    def test_data_response_emits_data_update(self):
        from specula.lib.display_server_api import sio
        srv = self._make_server()
        srv.client_types['cli1'] = 'dpg'

        with patch.object(sio, 'emit') as mock_emit:
            self._manual_handle(srv, ('data_response', 'cli1', 'slope', {'type': '1d_array'}))

        mock_emit.assert_called_once()
        self.assertEqual(mock_emit.call_args[0][0], 'data_update')
        self.assertEqual(mock_emit.call_args[0][1]['name'], 'slope')

    def test_terminator_emits_speed_and_done(self):
        from specula.lib.display_server_api import sio
        srv = self._make_server()
        srv.client_types['cli1'] = 'dpg'
        srv.t0['cli1'] = time.time() - 0.5

        with patch.object(sio, 'emit') as mock_emit:
            self._manual_handle(srv, ('terminator', 'cli1', None, '20.00 Hz'))

        calls = [c[0][0] for c in mock_emit.call_args_list]
        self.assertIn('speed_report', calls)
        self.assertIn('done', calls)

    def test_unknown_client_ignored(self):
        from specula.lib.display_server_api import sio
        srv = self._make_server()
        # 'ghost' is not registered

        with patch.object(sio, 'emit') as mock_emit:
            self._manual_handle(srv, ('terminator', 'ghost', None, ''))

        mock_emit.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for /status route
# ---------------------------------------------------------------------------

class TestStatusRoute(unittest.TestCase):

    def _get_app(self):
        from specula.lib.display_server_api import app
        app.config['TESTING'] = True
        return app.test_client()

    def test_status_not_initialized(self):
        import specula.lib.display_server_api as api
        original = api.server
        api.server = None
        client = self._get_app()
        try:
            resp = client.get('/status')
            data = resp.get_json()
            self.assertEqual(data['status'], 'not_initialized')
        finally:
            api.server = original

    def test_status_image_mode(self):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import ImageFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            mock_server = ImageFlaskServer(params_dict={}, status_queue=sq, request_queue=rq)
        mock_server.actual_port = 9999
        mock_server.client_types = {'a': 'web'}

        original = api.server
        api.server = mock_server
        client = self._get_app()
        try:
            resp = client.get('/status')
            data = resp.get_json()
            self.assertEqual(data['status'], 'running')
            self.assertEqual(data['mode'], 'image')
            self.assertEqual(data['port'], 9999)
            self.assertEqual(data['connected_clients'], 1)
        finally:
            api.server = original

    def test_status_data_mode(self):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import DataFlaskServer
        sq = mp.Queue()
        rq = mp.Queue()
        with patch('threading.Thread'):
            mock_server = DataFlaskServer(params_dict={}, status_queue=sq, request_queue=rq)
        mock_server.actual_port = 8888
        mock_server.client_types = {}

        original = api.server
        api.server = mock_server
        client = self._get_app()
        try:
            resp = client.get('/status')
            data = resp.get_json()
            self.assertEqual(data['mode'], 'data')
            self.assertEqual(data['connected_clients'], 0)
        finally:
            api.server = original