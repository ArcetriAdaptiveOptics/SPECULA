
import unittest

from specula.processing_objects.terminal_input import TerminalInput



class TestTerminalInput(unittest.TestCase):

    def test_singleton(self):
        a = TerminalInput(output_list=["a", "b"])

        with self.assertRaises(RuntimeError):
            b = TerminalInput(output_list=["a", "b"])
