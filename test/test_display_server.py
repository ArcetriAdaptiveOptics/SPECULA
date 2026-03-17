import specula
specula.init(0)  # Default target device

import os
import sys
import pickle
import queue as _queue_module
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call
import multiprocessing as mp
import numpy as np

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

    def test_numpy_array_passed_directly_return_unknown(self):
        server = self._make_server()
        arr = np.eye(3)
        result = server._process_for_dpg(arr, 'direct')
        self.assertEqual(result['type'], 'unknown')

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
                server.qin = mp.Queue()
                server.qout = mp.Queue()
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

    @unittest.skipIf(os.environ.get('CI') == 'true', "Disabled for CI issues with multiprocessing.Queue")
    def test_trigger_image_mode_processes_request(self):
        from specula.base_value import BaseValue
        server = self._make_server('image')

        arr = np.ones((3, 3))
        mock_obj = MagicMock()
        mock_obj.copyTo.return_value = mock_obj
        mock_obj.array_for_display.return_value = arr
        # Remove xp so pickle works
        if hasattr(mock_obj, 'xp'):
            del mock_obj.xp

        server.data_obj_getter = lambda name: mock_obj
        server.qin.put(('client_abc', ['obj1']))

        server.trigger()
        time.sleep(0.001)  # Queue context switch

        types = []
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                item = server.qout.get(timeout=0.05)
            except _queue_module.Empty:
                continue

            if isinstance(item, tuple):
                types.append(item[0])
                if item[0] == 'image_terminator':
                    break

        self.assertIn('image_terminator', types)

    # -- data mode --

    def test_trigger_data_mode_empty_queue(self):
        server = self._make_server('data')
        server.data_obj_getter = lambda name: None
        server.trigger()   # should not raise

    @unittest.skipIf(os.environ.get('CI') == 'true', "Disabled for CI issues with multiprocessing.Queue")
    def test_trigger_data_mode_processes_request(self):
        server = self._make_server('data')

        arr = np.arange(5, dtype=float)
        mock_obj = MagicMock()
        mock_obj.copyTo.return_value = mock_obj
        mock_obj.array_for_display.return_value = arr

        server.data_obj_getter = lambda name: mock_obj
        server.qin.put(('client_xyz', ['obj1']))

        server.trigger()
        time.sleep(0.001) # Queue context switch

        types = []
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                item = server.qout.get(timeout=0.05)
            except _queue_module.Empty:
                continue

            if isinstance(item, tuple):
                types.append(item[0])
                if item[0] == 'terminator':
                    break

        self.assertIn('terminator', types)

    def test_trigger_puts_speed_report_after_one_second(self):
        server = self._make_server('image')
        server.data_obj_getter = lambda name: None
        server.t0 = time.time() - 2   # guarantee the 1-second check fires

        server.trigger()
        time.sleep(0.001) # Queue context switch

        # Speed report is a 2-tuple (name, status_string)
        items = []
        while not server.qout.empty():
            items.append(server.qout.get_nowait())

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

from specula.lib.display_server_api import filter_params_for_display as _filter

class TestDisplayParamsFiltering(unittest.TestCase):
    """
    The same filtering logic appears in both image-mode and data-mode handlers.
    We test it independently.
    """ 

    def test_datastore_excluded(self):
        params = {'ds': {'class': 'DataStore', 'inputs': ['x']}}
        result = _filter(params)
        self.assertNotIn('ds', result)

    def test_class_without_inputs_outputs_excluded(self):
        params = {'src': {'class': 'Source', 'mag': 5}}
        result = _filter(params)
        self.assertNotIn('src', result)

    def test_class_with_inputs_included(self):
        params = {'wfs': {'class': 'Sensor', 'inputs': ['pupil']}}
        result = _filter(params)
        self.assertIn('wfs', result)

    def test_class_with_outputs_included(self):
        params = {'dm': {'class': 'DM', 'outputs': ['phase']}}
        result = _filter(params)
        self.assertIn('dm', result)

    def test_entry_without_class_always_included(self):
        params = {'meta': {'version': '1.0'}}
        result = _filter(params)
        self.assertIn('meta', result)

    def test_mixed_params(self):
        params = {
            'ds': {'class': 'DataStore', 'inputs': ['x']},
            'src': {'class': 'Source', 'mag': 5},
            'wfs': {'class': 'Sensor', 'inputs': ['pupil']},
            'cfg': {'root_dir': '/tmp'},
        }
        result = _filter(params)
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

    @unittest.skipIf(sys.platform == 'darwin', reason='Not implemented on MacOSX')
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

    @unittest.skipIf(sys.platform == 'darwin', reason='Not implemented on MacOSX')
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


# ===========================================================================
# ADDITIONAL TESTS FOR HIGHER COVERAGE
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared picklable stub and helpers
# ---------------------------------------------------------------------------

class _PicklableDataObj:
    """Minimal picklable stand-in for a specula data object.

    MagicMock cannot be pickled — the image-mode trigger calls pickle.dumps()
    on the object returned by copyTo(), so we need a real Python class.
    """
    def __init__(self, array):
        self._array = array
        self.xp = np   # stripped by remove_xp_np before pickling

    def copyTo(self, device):
        return _PicklableDataObj(self._array.copy())

    def array_for_display(self):
        return self._array


def _drain_queue(q, max_items=200):
    """Reliably drain all items from a queue without relying on .empty()."""
    items = []
    for _ in range(max_items):
        try:
            items.append(q.get_nowait())
        except Exception:
            break
    return items


# ---------------------------------------------------------------------------
# DisplayServer.__init__ and data_obj_getter closure
# ---------------------------------------------------------------------------

class TestDisplayServerInit(unittest.TestCase):

    def _build(self, mode='image', input_getter=None, output_getter=None,
               info_getter=None):
        from specula.processing_objects.display_server import DisplayServer
        input_getter = input_getter or (lambda name: object())
        output_getter = output_getter or (lambda name: object())
        info_getter = info_getter or (lambda: ('sim', 'running'))
        with patch('specula.processing_objects.display_server.start_server'), \
             patch('multiprocessing.Process') as mock_proc_cls:
            mock_proc = MagicMock()
            mock_proc.start.return_value = None
            mock_proc_cls.return_value = mock_proc
            srv = DisplayServer(
                params_dict={'key': 'val'},
                input_ref_getter=input_getter,
                output_ref_getter=output_getter,
                info_getter=info_getter,
                mode=mode,
            )
        return srv

    def test_init_image_mode_sets_attributes(self):
        srv = self._build(mode='image')
        self.assertEqual(srv.mode, 'image')
        self.assertEqual(srv.counter, 0)
        self.assertEqual(srv.speed_report, '')
        self.assertIsNotNone(srv.data_obj_getter)

    def test_init_data_mode(self):
        srv = self._build(mode='data')
        self.assertEqual(srv.mode, 'data')

    def test_data_obj_getter_in_prefix_uses_input_getter(self):
        sentinel = object()
        srv = self._build(input_getter=lambda name: sentinel)
        self.assertIs(srv.data_obj_getter('wfs.in_slopes'), sentinel)

    def test_data_obj_getter_no_in_prefix_uses_output_getter(self):
        sentinel = object()
        srv = self._build(output_getter=lambda name: sentinel)
        self.assertIs(srv.data_obj_getter('wfs.slopes'), sentinel)

    def test_data_obj_getter_output_valueerror_falls_back_to_input(self):
        fallback = object()
        def raise_value_error(name):
            raise ValueError('not found')
        srv = self._build(
            output_getter=raise_value_error,
            input_getter=lambda name: fallback,
        )
        self.assertIs(srv.data_obj_getter('wfs.something'), fallback)

    def test_data_obj_getter_outer_exception_returns_none(self):
        def boom(name):
            raise RuntimeError('unexpected')
        srv = self._build(input_getter=boom, output_getter=boom)
        self.assertIsNone(srv.data_obj_getter('bad.in_obj'))


# ---------------------------------------------------------------------------
# _process_for_dpg – previously uncovered branches
# ---------------------------------------------------------------------------

class TestProcessForDpgExtra(unittest.TestCase):

    def _make_server(self):
        from specula.processing_objects.display_server import DisplayServer
        srv = DisplayServer.__new__(DisplayServer)
        srv.mode = 'data'
        srv.qin = _queue_module.Queue()
        srv.qout = _queue_module.Queue()
        srv.params_dict = {}
        srv.counter = 0
        srv.t0 = time.time()
        srv.c0 = 0
        srv.speed_report = ''
        return srv

    def test_list_mixed_dimensions_returns_multi_data(self):
        srv = self._make_server()
        o1, o2 = MagicMock(), MagicMock()
        o1.array_for_display.return_value = np.ones(5)
        o2.array_for_display.return_value = np.ones((3, 3))
        result = srv._process_for_dpg([o1, o2], 'mixed')
        self.assertEqual(result['type'], 'multi_data')
        self.assertIn('shapes', result)
        self.assertIn('dtypes', result)

    def test_list_different_length_1d_returns_multi_data(self):
        srv = self._make_server()
        o1, o2 = MagicMock(), MagicMock()
        o1.array_for_display.return_value = np.ones(5)
        o2.array_for_display.return_value = np.ones(7)
        result = srv._process_for_dpg([o1, o2], 'diff_len')
        self.assertEqual(result['type'], 'multi_data')

    def test_list_all_none_returns_unknown(self):
        srv = self._make_server()
        obj = MagicMock()
        obj.array_for_display.return_value = None
        obj.get.return_value = None
        obj.__array__ = MagicMock(side_effect=TypeError)
        obj.shape = None
        obj.value = None
        result = srv._process_for_dpg([obj], 'empty_list')
        self.assertIn(result['type'], ('unknown', 'error'))

    def test_random_class_returns_unknown(self):
        """Test that an object not deriving from BaseDataObject is handled gracefully."""
        srv = self._make_server()

        class ObjWithValueOnly:
            def __init__(self):
                self.value = np.array([1.0, 2.0, 3.0])

        result = srv._process_for_dpg(ObjWithValueOnly(), 'val_fb')
        self.assertEqual(result['type'], 'unknown')

    def test_list_integer_arrays_cast_to_float32(self):
        srv = self._make_server()
        obj = MagicMock()
        obj.array_for_display.return_value = np.array([1, 2, 3], dtype=np.int16)
        result = srv._process_for_dpg([obj], 'int_list')
        self.assertEqual(result['dtype'], 'float32')


# ---------------------------------------------------------------------------
# Trigger: list dataobj and error handling paths
# ---------------------------------------------------------------------------

class TestDisplayServerTriggerExtra(unittest.TestCase):

    def _make_server(self, mode='image'):
        from specula.processing_objects.display_server import DisplayServer
        srv = DisplayServer.__new__(DisplayServer)
        srv.mode = mode
        srv.qin = _queue_module.Queue()
        srv.qout = _queue_module.Queue()
        srv.params_dict = {}
        srv.counter = 0
        srv.t0 = time.time() - 2
        srv.c0 = 0
        srv.speed_report = ''
        srv.info_getter = lambda: ('sim', 'running')
        return srv

    def test_trigger_image_list_dataobj(self):
        srv = self._make_server('image')
        elems = [_PicklableDataObj(np.zeros(3)), _PicklableDataObj(np.ones(3))]
        srv.data_obj_getter = lambda name: elems
        srv.qin.put(('cli_list', ['multi_obj']))
        srv.trigger()
        items = _drain_queue(srv.qout)
        self.assertIn('image_terminator', [i[0] for i in items if isinstance(i, tuple)])

    def test_trigger_image_copyto_error_swallowed(self):
        srv = self._make_server('image')
        bad = MagicMock()
        bad.copyTo.side_effect = RuntimeError('copy failed')
        srv.data_obj_getter = lambda name: bad
        srv.qin.put(('cli_err', ['bad_obj']))
        srv.trigger()  # must not raise
        items = _drain_queue(srv.qout)
        self.assertIn('image_terminator', [i[0] for i in items if isinstance(i, tuple)])

    def test_trigger_data_list_dataobj(self):
        srv = self._make_server('data')
        elems = [MagicMock(), MagicMock()]
        for e in elems:
            e.copyTo.return_value = e
            e.array_for_display.return_value = np.ones(4)
        srv.data_obj_getter = lambda name: elems
        srv.qin.put(('cli_list', ['multi_obj']))
        srv.trigger()
        items = _drain_queue(srv.qout)
        self.assertIn('terminator', [i[0] for i in items if isinstance(i, tuple)])

    def test_trigger_data_xp_set_on_list_elements(self):
        """copyTo results in a list that lack .xp must get xp=np injected."""
        srv = self._make_server('data')

        class NoXp:
            def copyTo(self, device):
                return self
            def array_for_display(self):
                return np.ones(3)

        objs = [NoXp(), NoXp()]
        srv.data_obj_getter = lambda name: objs
        srv.qin.put(('cli_xp', ['obj']))
        srv.trigger()
        for o in objs:
            self.assertIs(o.xp, np)

    def test_trigger_data_copyto_error_appends_error_response(self):
        srv = self._make_server('data')
        bad = MagicMock()
        bad.copyTo.side_effect = RuntimeError('copy failure')
        srv.data_obj_getter = lambda name: bad
        srv.qin.put(('cli_err', ['bad_obj']))
        srv.trigger()
        items = _drain_queue(srv.qout)
        types = [i[0] for i in items if isinstance(i, tuple)]
        self.assertIn('terminator', types)
        error_items = [
            i for i in items
            if isinstance(i, tuple) and len(i) == 4
            and i[0] == 'data_response'
            and isinstance(i[3], dict) and i[3].get('type') == 'error'
        ]
        self.assertTrue(len(error_items) >= 1)


# ---------------------------------------------------------------------------
# start_server() – both modes
# ---------------------------------------------------------------------------

class TestStartServer(unittest.TestCase):

    def _call(self, mode):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import (
            start_server, ImageFlaskServer, DataFlaskServer,
        )
        sq = mp.Queue()
        rq = mp.Queue()
        params = {'wfs': {'class': 'Sensor', 'inputs': ['p'], 'bad': object()}}
        with patch.object(ImageFlaskServer, 'run'), \
             patch.object(DataFlaskServer, 'run'), \
             patch('threading.Thread'):
            start_server(params, sq, rq, '0.0.0.0', 5000, mode)
        return api.server

    def test_image_mode_creates_image_server(self):
        from specula.lib.display_server_api import ImageFlaskServer
        self.assertIsInstance(self._call('image'), ImageFlaskServer)

    def test_data_mode_creates_data_server(self):
        from specula.lib.display_server_api import DataFlaskServer
        self.assertIsInstance(self._call('data'), DataFlaskServer)

    def test_non_serialisable_values_stripped(self):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import start_server, ImageFlaskServer
        sq, rq = mp.Queue(), mp.Queue()
        params = {'obj': {'class': 'Sensor', 'inputs': ['p'], 'bad': object(), 'good': 42}}
        with patch.object(ImageFlaskServer, 'run'), patch('threading.Thread'):
            start_server(params, sq, rq, '0.0.0.0', 5000, 'image')
        self.assertNotIn('bad', api.server.params_dict.get('obj', {}))
        self.assertEqual(api.server.params_dict['obj']['good'], 42)


# ---------------------------------------------------------------------------
# shutdown() on both server types
# ---------------------------------------------------------------------------

class TestServerShutdown(unittest.TestCase):

    def _make_image(self):
        from specula.lib.display_server_api import ImageFlaskServer
        with patch('threading.Thread'):
            return ImageFlaskServer(params_dict={}, status_queue=mp.Queue(),
                                    request_queue=mp.Queue())

    def _make_data(self):
        from specula.lib.display_server_api import DataFlaskServer
        with patch('threading.Thread'):
            return DataFlaskServer(params_dict={}, status_queue=mp.Queue(),
                                   request_queue=mp.Queue())

    def test_image_server_shutdown_calls_os_exit(self):
        srv = self._make_image()
        with patch('os._exit') as mock_exit:
            srv.shutdown()
        mock_exit.assert_called_once_with(0)

    def test_data_server_shutdown_calls_os_exit(self):
        srv = self._make_data()
        with patch('os._exit') as mock_exit:
            srv.shutdown()
        mock_exit.assert_called_once_with(0)


# ---------------------------------------------------------------------------
# ImageFlaskServer.status_update – logic branches
# ---------------------------------------------------------------------------

class TestImageFlaskServerStatusUpdate(unittest.TestCase):

    def _make_server(self):
        from specula.lib.display_server_api import ImageFlaskServer
        sq = _queue_module.Queue()
        with patch('threading.Thread'):
            srv = ImageFlaskServer(params_dict={}, status_queue=sq,
                                   request_queue=_queue_module.Queue())
        srv.actual_port = 5000
        return srv, sq

    def test_breaks_on_none_data(self):
        srv, sq = self._make_server()
        srv.shutdown = lambda: 0  # prevent actual shutdown
        sq.put(('sim', None))
        with patch('socketio.Client'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        self.assertFalse(t.is_alive())

    def test_emits_simul_update_on_status_tuple(self):
        srv, sq = self._make_server()
        srv.shutdown = lambda: 0  # prevent actual shutdown
        mock_client = MagicMock()
        sq.put(('sim_name', 'running at 10 Hz'))
        sq.put(('sim_name', None))
        with patch('socketio.Client', return_value=mock_client):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        mock_client.emit.assert_called()
        self.assertEqual(mock_client.emit.call_args[0][0], 'simul_update')

    def test_requeues_non_2tuple_items(self):
        srv, sq = self._make_server()
        srv.shutdown = lambda: 0  # prevent actual shutdown
        sq.put(('image_terminator', 'cli1', None, '5 Hz'))  # 4-tuple → requeue
        sq.put(('sim', None))
        with patch('socketio.Client'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        items = _drain_queue(sq)
        self.assertTrue(any(isinstance(i, tuple) and len(i) == 4 for i in items))

    def test_connection_error_resets_frontend_flag(self):
        import socketio.exceptions
        srv, sq = self._make_server()
        srv.frontend_connected = True
        srv.shutdown = lambda: 0  # prevent actual shutdown
        mock_client = MagicMock()
        mock_client.emit.side_effect = socketio.exceptions.ConnectionError()
        sq.put(('sim', 'running'))
        sq.put(('sim', None))
        with patch('socketio.Client', return_value=mock_client), \
             patch('time.sleep'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        self.assertFalse(srv.frontend_connected)


# ---------------------------------------------------------------------------
# ImageFlaskServer.handle_image_responses – actual thread execution
# ---------------------------------------------------------------------------

class TestImageFlaskServerHandleResponsesActual(unittest.TestCase):

    def _make_server(self):
        from specula.lib.display_server_api import ImageFlaskServer
        sq = _queue_module.Queue()
        # Let the real daemon thread start so coverage is recorded
        srv = ImageFlaskServer(params_dict={}, status_queue=sq,
                               request_queue=_queue_module.Queue())
        return srv, sq

    def test_image_data_branch_emits_plot(self):
        from specula.lib.display_server_api import sio
        from specula.processing_objects.display_server import remove_xp_np
        srv, sq = self._make_server()
        srv.client_types['cli1'] = 'web'
        obj = _PicklableDataObj(np.eye(3))
        with remove_xp_np(obj):
            obj_bytes = pickle.dumps(obj)
            mock_fig = MagicMock()
            mock_fig.savefig.side_effect = lambda buf, format: buf.write(b'\x89PNG\r\n\x1a\n')
            with patch.object(sio, 'emit') as mock_emit, \
                patch('specula.lib.display_server_api.DataPlotter') as mock_dp:
                mock_dp.plot_best_effort.return_value = mock_fig
                sq.put(('image_data', 'cli1', 'slopes', obj_bytes))
                time.sleep(0.3)
        self.assertIn('plot', [c[0][0] for c in mock_emit.call_args_list])

    def test_image_data_unknown_client_not_emitted(self):
        from specula.lib.display_server_api import sio
        from specula.processing_objects.display_server import remove_xp_np
        srv, sq = self._make_server()
        obj = _PicklableDataObj(np.zeros(4))
        with remove_xp_np(obj):
            obj_bytes = pickle.dumps(obj)
            with patch.object(sio, 'emit') as mock_emit:
                sq.put(('image_data', 'ghost', 'slopes', obj_bytes))
                time.sleep(0.3)
        self.assertEqual(
            len([c for c in mock_emit.call_args_list if c[0][0] == 'plot']), 0
        )

    def test_image_terminator_emits_done_in_thread(self):
        from specula.lib.display_server_api import sio
        srv, sq = self._make_server()
        srv.client_types['cli1'] = 'web'
        srv.t0['cli1'] = time.time() - 1.0
        with patch.object(sio, 'emit') as mock_emit:
            sq.put(('image_terminator', 'cli1', None, '10 Hz'))
            time.sleep(0.3)
        self.assertIn('done', [c[0][0] for c in mock_emit.call_args_list])

    def test_2tuple_item_requeued_thread_survives(self):
        srv, sq = self._make_server()
        sq.put(('status_name', 'running'))
        time.sleep(0.3)
        self.assertTrue(srv.response_handler_thread.is_alive())


# ---------------------------------------------------------------------------
# DataFlaskServer.handle_responses – actual thread execution
# ---------------------------------------------------------------------------

class TestDataFlaskServerHandleResponsesActual(unittest.TestCase):

    def _make_server(self):
        from specula.lib.display_server_api import DataFlaskServer
        sq = _queue_module.Queue()
        srv = DataFlaskServer(params_dict={}, status_queue=sq,
                              request_queue=_queue_module.Queue())
        return srv, sq

    def test_data_response_emits_data_update(self):
        from specula.lib.display_server_api import sio
        srv, sq = self._make_server()
        srv.client_types['cli1'] = 'dpg'
        with patch.object(sio, 'emit') as mock_emit:
            sq.put(('data_response', 'cli1', 'slope',
                    {'type': '1d_array', 'data': [1, 2]}))
            time.sleep(0.3)
        self.assertIn('data_update', [c[0][0] for c in mock_emit.call_args_list])

    def test_terminator_emits_done(self):
        from specula.lib.display_server_api import sio
        srv, sq = self._make_server()
        srv.client_types['cli1'] = 'dpg'
        srv.t0['cli1'] = time.time() - 1.0
        with patch.object(sio, 'emit') as mock_emit:
            sq.put(('terminator', 'cli1', None, '15 Hz'))
            time.sleep(0.3)
        self.assertIn('done', [c[0][0] for c in mock_emit.call_args_list])

    def test_2tuple_item_requeued_thread_survives(self):
        srv, sq = self._make_server()
        sq.put(('sim_name', 'running'))
        time.sleep(0.3)
        self.assertTrue(srv.response_handler_thread.is_alive())

    def test_unknown_format_thread_survives(self):
        srv, sq = self._make_server()
        sq.put(('only_one',))  # 1-tuple → "Unknown item format" log
        time.sleep(0.3)
        self.assertTrue(srv.response_handler_thread.is_alive())

    def test_unknown_client_not_emitted(self):
        from specula.lib.display_server_api import sio
        srv, sq = self._make_server()
        with patch.object(sio, 'emit') as mock_emit:
            sq.put(('data_response', 'ghost', 'slope', {}))
            time.sleep(0.3)
        self.assertEqual(len(mock_emit.call_args_list), 0)


# ---------------------------------------------------------------------------
# DataFlaskServer.status_update – logic branches
# ---------------------------------------------------------------------------

class TestDataFlaskServerStatusUpdate(unittest.TestCase):

    def _make_server(self):
        from specula.lib.display_server_api import DataFlaskServer
        sq = _queue_module.Queue()
        with patch('threading.Thread'):
            srv = DataFlaskServer(params_dict={}, status_queue=sq,
                                  request_queue=_queue_module.Queue())
        srv.actual_port = 5000
        return srv, sq

    def test_breaks_on_none_data(self):
        srv, sq = self._make_server()
        sq.put(('sim', None))
        with patch('socketio.Client'), patch('os._exit'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        self.assertFalse(t.is_alive())

    def test_skips_data_response_2tuples(self):
        srv, sq = self._make_server()
        sq.put(('data_response', 'x'))   # must be requeued, not consumed
        sq.put(('sim', None))            # then break
        with patch('socketio.Client'), patch('os._exit'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        items = _drain_queue(sq)
        self.assertIn(('data_response', 'x'), items)

    def test_emits_simul_update_on_status(self):
        srv, sq = self._make_server()
        mock_client = MagicMock()
        sq.put(('sim_name', 'running at 10 Hz'))
        sq.put(('sim_name', None))
        with patch('socketio.Client', return_value=mock_client), \
             patch('os._exit'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        mock_client.emit.assert_called()

    def test_requeues_non_2tuple_items(self):
        srv, sq = self._make_server()
        sq.put(('terminator', 'cli1', None, '5 Hz'))  # 4-tuple → requeue
        sq.put(('sim', None))
        with patch('socketio.Client'), patch('os._exit'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=3.0)
        items = _drain_queue(sq)
        self.assertTrue(any(isinstance(i, tuple) and len(i) == 4 for i in items))

    def test_queue_empty_calls_shutdown(self):
        srv, sq = self._make_server()
        mock_q = MagicMock()
        mock_q.get.side_effect = _queue_module.Empty()
        srv.status_queue = mock_q
        with patch.object(srv, 'shutdown') as mock_shutdown, \
             patch('socketio.Client'):
            
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
        mock_shutdown.assert_called()

    def test_eoferror_calls_shutdown(self):
        srv, sq = self._make_server()
        mock_q = MagicMock()
        mock_q.get.side_effect = EOFError()
        srv.status_queue = mock_q
        with patch.object(srv, 'shutdown') as mock_shutdown, \
             patch('socketio.Client'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
        mock_shutdown.assert_called()

    def test_generic_exception_calls_shutdown(self):
        srv, sq = self._make_server()
        mock_q = MagicMock()
        mock_q.get.side_effect = RuntimeError('unexpected')
        srv.status_queue = mock_q
        with patch.object(srv, 'shutdown') as mock_shutdown, \
             patch('socketio.Client'):
            t = threading.Thread(target=srv.status_update,
                                 args=(MagicMock(),), daemon=True)
            t.start()
            t.join(timeout=2.0)
        mock_shutdown.assert_called()


# ---------------------------------------------------------------------------
# Socket.IO event handlers – image mode (via Flask-SocketIO test client)
# ---------------------------------------------------------------------------

class TestImageModeSocketHandlers(unittest.TestCase):

    def setUp(self):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import (
            sio, app, ImageFlaskServer, setup_image_mode_handlers,
        )
        sio.handlers = {}
        sio.namespace_handlers = {'/': {}}

        sq = _queue_module.Queue()
        rq = _queue_module.Queue()
        with patch('threading.Thread'):
            self.srv = ImageFlaskServer(
                params_dict={
                    'wfs': {'class': 'Sensor', 'inputs': ['pupil']},
                    'ds':  {'class': 'DataStore', 'inputs': ['x']},
                    'src': {'class': 'Source'},
                    'cfg': {'root_dir': '/tmp'},
                },
                status_queue=sq,
                request_queue=rq,
            )
        self.original_server = api.server
        api.server = self.srv
        setup_image_mode_handlers()
        self.sio = sio
        self.app = app
        self.api = api
        self.rq = rq

    def tearDown(self):
        self.api.server = self.original_server

    def test_connect_registers_web_client_type(self):
        tc = self.sio.test_client(self.app)
        self.assertTrue(any(v == 'web' for v in self.srv.client_types.values()))
        tc.disconnect()

    def test_connect_emits_filtered_params(self):
        tc = self.sio.test_client(self.app)
        received = tc.get_received()
        params_events = [e for e in received if e['name'] == 'params']
        self.assertTrue(len(params_events) >= 1)
        data = params_events[0]['args'][0]
        self.assertIn('wfs', data)
        self.assertNotIn('ds', data)
        self.assertNotIn('src', data)
        self.assertIn('cfg', data)
        tc.disconnect()

    def test_newdata_puts_request_on_queue(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['obj1', 'obj2'])
        item = self.rq.get(timeout=1.0)
        client_id, names = item
        self.assertIsInstance(client_id, str)
        self.assertEqual(names, ['obj1', 'obj2'])
        tc.disconnect()

    def test_newdata_sets_t0(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['obj1'])
        time.sleep(0.05)
        self.assertTrue(len(self.srv.t0) >= 1)
        tc.disconnect()

    def test_disconnect_clears_client_type_and_t0(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['obj1'])
        time.sleep(0.05)
        registered_ids = list(self.srv.client_types.keys())
        tc.disconnect()
        for cid in registered_ids:
            self.assertNotIn(cid, self.srv.client_types)
            self.assertNotIn(cid, self.srv.t0)


# ---------------------------------------------------------------------------
# Socket.IO event handlers – data mode (via Flask-SocketIO test client)
# ---------------------------------------------------------------------------

class TestDataModeSocketHandlers(unittest.TestCase):

    def setUp(self):
        import specula.lib.display_server_api as api
        from specula.lib.display_server_api import (
            sio, app, DataFlaskServer, setup_data_mode_handlers,
        )
        sio.handlers = {}
        sio.namespace_handlers = {'/': {}}

        sq = _queue_module.Queue()
        rq = _queue_module.Queue()
        with patch('threading.Thread'):
            self.srv = DataFlaskServer(
                params_dict={
                    'wfs': {'class': 'Sensor', 'inputs': ['pupil']},
                    'ds':  {'class': 'DataStore', 'inputs': ['x']},
                },
                status_queue=sq,
                request_queue=rq,
            )
        self.original_server = api.server
        api.server = self.srv
        setup_data_mode_handlers()
        self.sio = sio
        self.app = app
        self.api = api
        self.rq = rq

    def tearDown(self):
        self.api.server = self.original_server

    def test_connect_registers_dpg_client(self):
        tc = self.sio.test_client(self.app)
        self.assertTrue(any(v == 'dpg' for v in self.srv.client_types.values()))
        tc.disconnect()

    def test_connect_creates_subscription_set(self):
        tc = self.sio.test_client(self.app)
        for sid in self.srv.client_types:
            self.assertIn(sid, self.srv.client_subscriptions)
            self.assertIsInstance(self.srv.client_subscriptions[sid], set)
        tc.disconnect()

    def test_connect_emits_filtered_params(self):
        tc = self.sio.test_client(self.app)
        received = tc.get_received()
        params_events = [e for e in received if e['name'] == 'params']
        self.assertTrue(len(params_events) >= 1)
        data = params_events[0]['args'][0]
        self.assertIn('wfs', data)
        self.assertNotIn('ds', data)
        tc.disconnect()

    def test_newdata_puts_request_on_queue(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['slope', 'phase'])
        item = self.rq.get(timeout=1.0)
        _, names = item
        self.assertIn('slope', names)
        tc.disconnect()

    def test_newdata_updates_subscriptions(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['slope', 'phase'])
        time.sleep(0.05)
        self.assertTrue(
            any('slope' in s for s in self.srv.client_subscriptions.values())
        )
        tc.disconnect()

    def test_newdata_updates_last_request_outputs(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['slope'])
        time.sleep(0.05)
        self.assertIn('slope', self.srv.last_request_outputs)
        tc.disconnect()

    def test_get_params_emits_params(self):
        tc = self.sio.test_client(self.app)
        tc.get_received()  # drain connect events
        tc.emit('get_params')
        time.sleep(0.05)
        received = tc.get_received()
        self.assertTrue(any(e['name'] == 'params' for e in received))
        tc.disconnect()

    def test_set_client_type_updates_type(self):
        tc = self.sio.test_client(self.app)
        tc.emit('set_client_type', {'type': 'web'})
        time.sleep(0.05)
        self.assertTrue(any(v == 'web' for v in self.srv.client_types.values()))
        tc.disconnect()

    def test_set_client_type_default_is_dpg(self):
        tc = self.sio.test_client(self.app)
        tc.emit('set_client_type', {})
        time.sleep(0.05)
        self.assertTrue(any(v == 'dpg' for v in self.srv.client_types.values()))
        tc.disconnect()

    def test_unsubscribe_removes_output(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['slope', 'phase'])
        time.sleep(0.05)
        tc.emit('unsubscribe', {'output': 'slope'})
        time.sleep(0.05)
        self.assertTrue(
            all('slope' not in s for s in self.srv.client_subscriptions.values())
        )
        tc.disconnect()

    def test_unsubscribe_nonexistent_output_no_error(self):
        tc = self.sio.test_client(self.app)
        tc.emit('unsubscribe', {'output': 'nonexistent'})
        time.sleep(0.05)
        tc.disconnect()  # no exception expected

    def test_disconnect_removes_from_all_dicts(self):
        tc = self.sio.test_client(self.app)
        tc.emit('newdata', ['slope'])
        time.sleep(0.05)
        registered_ids = list(self.srv.client_types.keys())
        tc.disconnect()
        for cid in registered_ids:
            self.assertNotIn(cid, self.srv.client_types)
            self.assertNotIn(cid, self.srv.t0)
            self.assertNotIn(cid, self.srv.client_subscriptions)
