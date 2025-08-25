import unittest
from unittest.mock import patch, MagicMock
import subprocess
import os
import sys

from specula.lib.process_utils import daemonize, killProcessByName


class TestDaemonize(unittest.TestCase):

    def setUp(self):
        # Patch sys.stdin/stdout/stderr with mocks that support fileno()
        self.stdin_patch = patch("sys.stdin", MagicMock())
        self.stdout_patch = patch("sys.stdout", MagicMock())
        self.stderr_patch = patch("sys.stderr", MagicMock())

        self.mock_stdin = self.stdin_patch.start()
        self.mock_stdout = self.stdout_patch.start()
        self.mock_stderr = self.stderr_patch.start()

        # Make sure mocked streams have a valid fileno() method
        self.mock_stdin.fileno.return_value = 0
        self.mock_stdout.fileno.return_value = 1
        self.mock_stderr.fileno.return_value = 2

    def tearDown(self):
        self.stdin_patch.stop()
        self.stdout_patch.stop()
        self.stderr_patch.stop()

    @patch("sys.stderr.flush")
    @patch("sys.stdout.flush")
    @patch("os.fork")
    @patch("os.setsid")
    @patch("os.dup2")
    @patch("sys.exit")
    def test_daemonize_parent_exit_on_first_fork(
        self, mock_exit, mock_dup2, mock_setsid, mock_fork, mock_stdout_flush, mock_stderr_flush
    ):
        mock_fork.side_effect = [123, 0]  # first fork -> parent exits
        daemonize()
        mock_exit.assert_called_once_with(0)
        mock_setsid.assert_called_once()

    @patch("sys.stderr.flush")
    @patch("sys.stdout.flush")
    @patch("os.fork")
    @patch("os.setsid")
    @patch("os.dup2")
    @patch("sys.exit")
    def test_daemonize_child_detach_and_second_fork(
        self, mock_exit, mock_dup2, mock_setsid, mock_fork, mock_stdout_flush, mock_stderr_flush
    ):
        mock_fork.side_effect = [0, 123]  # child, then parent exits on second fork
        daemonize()
        mock_setsid.assert_called_once()
        self.assertEqual(mock_fork.call_count, 2)
        mock_exit.assert_called_once_with(0)

    @patch("sys.stderr.flush")
    @patch("sys.stdout.flush")
    @patch("os.fork")
    @patch("os.setsid")
    @patch("os.dup2")
    @patch("builtins.open", new_callable=MagicMock)
    @patch("sys.exit")
    def test_daemonize_redirects_streams(
        self, mock_exit, mock_open, mock_dup2, mock_setsid, mock_fork, mock_stdout_flush, mock_stderr_flush
    ):
        mock_fork.side_effect = [0, 0]  # child on both forks
        daemonize()
        mock_open.assert_called_with("/dev/null", "w")
        self.assertEqual(mock_dup2.call_count, 3)  # stdin, stdout, stderr
        mock_setsid.assert_called_once()
        mock_stdout_flush.assert_called_once()
        mock_stderr_flush.assert_called_once()


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


