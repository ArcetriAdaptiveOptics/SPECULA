import unittest
from unittest.mock import patch, MagicMock

import specula
specula.init(0)  # Default target device

from specula.lib.process_utils import killProcessByName


class TestKillProcessByName(unittest.TestCase):

    @patch("subprocess.Popen")
    @patch("subprocess.call")
    def test_kill_processes_successfully(self, mock_call, mock_popen):
        # Mock Popen to simulate three PIDs, skip own PID
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.stdout = [b"111\n", b"222\n", b"333\n"]
        mock_popen.return_value = mock_process
        mock_call.return_value = 0

        killProcessByName("myproc")

        # Should kill 222 and 333, not 111
        mock_call.assert_any_call("kill -KILL 222", shell=True)
        mock_call.assert_any_call("kill -KILL 333", shell=True)
        self.assertEqual(mock_call.call_count, 2)

    @patch("subprocess.Popen")
    @patch("subprocess.call")
    def test_kill_process_fails(self, mock_call, mock_popen):
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.stdout = [b"222\n"]
        mock_popen.return_value = mock_process
        mock_call.return_value = 1  # Simulate kill failure

        with self.assertRaises(AssertionError):
            killProcessByName("failingproc")

    @patch("subprocess.Popen")
    @patch("subprocess.call")
    def test_no_matching_processes(self, mock_call, mock_popen):
        mock_process = MagicMock()
        mock_process.pid = 111
        mock_process.stdout = []  # No processes found
        mock_popen.return_value = mock_process

        killProcessByName("emptyproc")
        mock_call.assert_not_called()


