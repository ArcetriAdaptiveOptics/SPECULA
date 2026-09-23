import io
import logging
import unittest
from unittest.mock import patch

import specula
specula.init(-1)  # Default target device

from specula.processing_objects.terminal_input import TerminalInput, TerminalReader
from specula.processing_objects.terminal_input import _redirect_log_handlers, _restore_log_handlers


class TestTerminalInput(unittest.TestCase):

    def test_singleton(self):
        a = TerminalInput(output_list=["a:int", "b:float"])

        with self.assertRaises(RuntimeError):
            b = TerminalInput(output_list=["a:int", "b:float"])

        a.finalize()

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
            redirected = _redirect_log_handlers(new_stream)
            self.assertIs(console.stream, new_stream)
            self.assertIsNot(other.stream, new_stream)
            _restore_log_handlers(redirected)
            self.assertIs(console.stream, sys.__stderr__)
        finally:
            root.removeHandler(console)
            root.removeHandler(other)
