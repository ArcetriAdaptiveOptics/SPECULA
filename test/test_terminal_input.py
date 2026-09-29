import io
import logging
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

import specula
specula.init(-1)  # Default target device

from specula.processing_objects.terminal_input import TerminalInput, TerminalReader
from specula.processing_objects.terminal_input import _redirect_log_handlers, _restore_log_handlers, _isatty, print_help


class TestTerminalInput(unittest.TestCase):

    # Do not start the reader thread, which would open a real
    # prompt when running from a terminal with "pytest -s"
    @patch.object(TerminalReader, 'start')
    def test_singleton(self, _):
        a = TerminalInput(output_list=["a:int", "b:float"])

        with self.assertRaises(RuntimeError):
            b = TerminalInput(output_list=["a:int", "b:float"])

        a.finalize()

    @patch.object(TerminalReader, 'start')
    def test_new_instance_after_finalize(self, _):
        a = TerminalInput(output_list=["a:int"])
        a.finalize()
        b = TerminalInput(output_list=["a:int"])
        b.finalize()

    def test_handle_line(self):
        received = []
        reader = TerminalReader(lambda name, value: received.append((name, value)))
        reader._handle_line('gain 0.3\n')
        reader._handle_line('   ')
        reader._handle_line('stop')
        with patch('builtins.print') as mock_print:
            reader._handle_line('too many tokens')
            mock_print.assert_called_once_with('Input not recognized')
        self.assertEqual(received, [('gain', '0.3'), ('stop', False)])

    def test_rejected_input_is_reported(self):
        def put(name, value):
            raise ValueError(f'Rejected input {value} for output {name}')
        reader = TerminalReader(put)
        with patch('builtins.print') as mock_print:
            reader._handle_line('gain abc')
            mock_print.assert_called_once_with('Rejected input abc for output gain')

    def test_plain_input_from_pipe(self):
        received = []
        reader = TerminalReader(lambda name, value: received.append((name, value)))
        with patch('sys.stdin', io.StringIO('a 1\nb 2\n')):
            reader._run()
        self.assertEqual(received, [('a', '1'), ('b', '2')])

    def test_redirect_log_handlers(self):
        import sys
        root = logging.getLogger()
        console = logging.StreamHandler(sys.__stderr__)
        other = logging.StreamHandler(io.StringIO())
        root.addHandler(console)
        root.addHandler(other)
        try:
            new_stream = io.StringIO()
            with patch('specula.processing_objects.terminal_input._isatty', return_value=True):
                redirected = _redirect_log_handlers(new_stream)
            self.assertIs(console.stream, new_stream)
            self.assertIsNot(other.stream, new_stream)
            _restore_log_handlers(redirected)
            self.assertIs(console.stream, sys.__stderr__)
        finally:
            root.removeHandler(console)
            root.removeHandler(other)

    def test_redirect_log_handlers_skips_non_tty(self):
        # e.g. stderr redirected to a file with "2> log.txt"
        import sys
        root = logging.getLogger()
        console = logging.StreamHandler(sys.__stderr__)
        root.addHandler(console)
        try:
            with patch('specula.processing_objects.terminal_input._isatty', return_value=False):
                redirected = _redirect_log_handlers(io.StringIO())
            self.assertEqual(redirected, [])
            self.assertIs(console.stream, sys.__stderr__)
        finally:
            root.removeHandler(console)

    def test_help(self):
        reader = TerminalReader(lambda name, value: None)
        with patch('specula.processing_objects.terminal_input.print_help') as mock_help:
            reader._handle_line('help')
            mock_help.assert_called_once()

    @patch.object(TerminalReader, 'start')
    def test_print_help(self, _):
        a = TerminalInput(output_list=["a:int"])
        try:
            with patch('builtins.print') as mock_print:
                print_help()
                mock_print.assert_called_once_with(["a:int"])
        finally:
            a.finalize()

    def test_run_interactive(self):
        reader = TerminalReader(lambda name, value: None)
        reader._interactive = True
        with patch.object(reader, '_run_prompt_toolkit') as mock_run:
            reader._run()
            mock_run.assert_called_once()

    def test_isatty_without_isatty_method(self):
        self.assertFalse(_isatty(object()))

    def test_start_plain_mode(self):
        received = []
        reader = TerminalReader(lambda name, value: received.append((name, value)))
        with patch('sys.stdin', io.StringIO('a 1\n')), \
             patch('specula.processing_objects.terminal_input._isatty', return_value=False):
            reader.start()
            reader.thread.join(timeout=5)
        self.assertFalse(reader._interactive)
        self.assertEqual(received, [('a', '1')])
        reader.stop()

    def test_run_after_stop_does_nothing(self):
        received = []
        reader = TerminalReader(lambda name, value: received.append((name, value)))
        reader.stop()
        with patch('sys.stdin', io.StringIO('a 1\n')):
            reader._run()
        self.assertEqual(received, [])

    def test_plain_input_stops_on_request(self):
        reader = TerminalReader(None)
        def put(name, value):
            reader.stop()
        reader.put = put
        with patch('sys.stdin', io.StringIO('a 1\nb 2\n')), \
             patch.object(reader, '_handle_line', wraps=reader._handle_line) as mock_handle:
            reader._run_plain()
            mock_handle.assert_called_once_with('a 1\n')

    def test_plain_input_closed_stdin(self):
        stdin = io.StringIO()
        stdin.close()
        reader = TerminalReader(lambda name, value: None)
        with patch('sys.stdin', stdin):
            reader._run_plain()    # Must not raise

    def test_stop_exits_running_app(self):
        reader = TerminalReader(lambda name, value: None)
        reader._app = MagicMock(is_running=True)
        reader.stop()
        reader._app.loop.call_soon_threadsafe.assert_called_once_with(reader._app.exit)

    def test_stop_ignores_app_errors(self):
        reader = TerminalReader(lambda name, value: None)
        reader._app = MagicMock(is_running=True)
        reader._app.loop.call_soon_threadsafe.side_effect = RuntimeError('loop closed')
        reader.stop()    # Must not raise

    def test_stop_waits_for_interactive_thread(self):
        reader = TerminalReader(lambda name, value: None)
        reader._interactive = True
        reader.thread = MagicMock()
        reader.stop(timeout=2.0)
        reader.thread.join.assert_called_once_with(timeout=2.0)

    def test_exit_if_stopping(self):
        reader = TerminalReader(lambda name, value: None)
        reader._app = MagicMock()
        reader._exit_if_stopping()
        reader._app.exit.assert_not_called()
        reader._stopping = True
        reader._exit_if_stopping()
        reader._app.exit.assert_called_once()


class TestPromptToolkit(unittest.TestCase):
    '''
    Run the prompt_toolkit loop in the test thread, reading
    keystrokes from a pipe instead of a real terminal.
    '''
    def _run_prompt(self, text, put=None):
        received = []
        reader = TerminalReader(put or (lambda name, value: received.append((name, value))))
        with create_pipe_input() as pipe_input:
            pipe_input.send_text(text)
            pipe_input.close()
            with create_app_session(input=pipe_input, output=DummyOutput()):
                reader._run_prompt_toolkit()
        self.assertIsNone(reader._app)
        return received

    def test_commands_until_eof(self):
        received = self._run_prompt('a 1\nb 2\n')
        self.assertEqual(received, [('a', '1'), ('b', '2')])

    def test_ctrl_c_interrupts_main_thread(self):
        with patch('specula.processing_objects.terminal_input._thread.interrupt_main') as mock_interrupt, \
             patch('builtins.print'):
            received = self._run_prompt('a 1\n\x03b 2\n')
            mock_interrupt.assert_called_once()
        self.assertEqual(received, [('a', '1')])

    def test_stop_from_command(self):
        reader = None
        received = []
        def put(name, value):
            received.append((name, value))
            reader.stop()
        reader = TerminalReader(put)
        with create_pipe_input() as pipe_input:
            pipe_input.send_text('a 1\nb 2\n')
            with create_app_session(input=pipe_input, output=DummyOutput()):
                reader._run_prompt_toolkit()
        self.assertEqual(received, [('a', '1')])

    def test_stop_from_other_thread(self):
        reader = TerminalReader(lambda name, value: None)
        with create_pipe_input() as pipe_input:
            def run():
                with create_app_session(input=pipe_input, output=DummyOutput()):
                    reader._run_prompt_toolkit()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            deadline = time.time() + 5
            while not (reader._app and reader._app.is_running) and time.time() < deadline:
                time.sleep(0.01)
            reader.stop()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertIsNone(reader._app)

    def test_log_handlers_restored(self):
        import sys
        root = logging.getLogger()
        console = logging.StreamHandler(sys.__stderr__)
        root.addHandler(console)
        try:
            with patch('specula.processing_objects.terminal_input._isatty', return_value=True):
                self._run_prompt('a 1\n')
            self.assertIs(console.stream, sys.__stderr__)
        finally:
            root.removeHandler(console)
