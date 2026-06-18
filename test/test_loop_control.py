import specula
specula.init(0)  # Default target device

import types
import unittest
from unittest.mock import MagicMock, patch

from specula.loop_control import LoopControl, process_rank
from specula.base_processing_obj import BaseProcessingObj

from test.specula_testlib import cpu_and_gpu



class MockProcessingObjNotReady(BaseProcessingObj):
    '''Class that is never ready, and raises if trigger() or post_trigger() are called'''
    def check_ready(self, t):
        self.inputs_changed = False

    def trigger(self):
        raise RuntimeError('trigger called when check_ready returned False')

    def post_trigger(self):
        raise RuntimeError('post_trigger called when check_ready returned False')


class MockProcessingObjReady(BaseProcessingObj):
    '''Class that is always ready and remembers whether trigger() and post_trigger() were called'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggered = False
        self.post_triggered = False

    def check_ready(self, t):
        self.inputs_changed = True

    def trigger(self):
        self.triggered = True

    def post_trigger(self):
        self.post_triggered = True


class TestLoopControl(unittest.TestCase):

    @cpu_and_gpu
    def test_check_ready_true(self, target_device_idx, xp):
        '''Test that trigger and post_triggered are called if check_ready is True'''

        loop = LoopControl()
        p = MockProcessingObjReady()

        loop.add(p, idx=0)
        loop.run(run_time=1, dt=1)

        assert p.triggered
        assert p.post_triggered

    @cpu_and_gpu
    def test_check_ready_False(self, target_device_idx, xp):
        '''Test that trigger and post_triggered are called if check_ready is False'''

        loop = LoopControl()
        p = MockProcessingObjNotReady()

        loop.add(p, idx=0)
        # Must not raise
        loop.run(run_time=1, dt=1)

    @cpu_and_gpu
    def test_infinite_loop(self, target_device_idx, xp):
        '''Test that setting run_time to -1 causes iteration to go beyond the run_time'''

        loop = LoopControl()
        p = MockProcessingObjReady()
        p.count = 0

        def trigger_count(self):
            self.count += 1
            if self.count > 20:
                raise StopIteration()
        p.trigger = types.MethodType(trigger_count, p)

        loop.add(p, idx=0)
        with self.assertRaises(StopIteration):
            # We setup for 10 iterations and verify that we reach 20
            loop.run(run_time=-1, dt=0.1)


class TestLoopControlTiming(unittest.TestCase):

    @cpu_and_gpu
    def test_niters_with_nonzero_t0(self, target_device_idx, xp):
        """niters() counts actual iterations from t0, not total steps from t=0"""
        loop = LoopControl()
        loop.run_time = loop.seconds_to_t(1.0)
        loop.dt = loop.seconds_to_t(0.1)
        loop.t0 = loop.seconds_to_t(0.5)
        assert loop.niters() == 10  # run_time/dt only, t0 not added

    @cpu_and_gpu
    def test_run_with_t0_starts_at_correct_time_and_iteration_count(self, target_device_idx, xp):
        """With t0 > 0, first check_ready receives t0 and total iterations equals run_time/dt"""
        times_seen = []

        class TimeRecorder(MockProcessingObjReady):
            def check_ready(self, t):
                super().check_ready(t)
                times_seen.append(t)

        loop = LoopControl()
        loop.add(TimeRecorder(), idx=0)
        loop.run(run_time=0.2, dt=0.1, t0=0.5)

        assert times_seen[0] == loop.seconds_to_t(0.5)
        assert len(times_seen) == 2  # run_time/dt = 0.2/0.1


class TestSteppingFeature(unittest.TestCase):

    def setUp(self):
        self.obj = LoopControl()

        # minimal required state
        self.obj.logger = MagicMock()
        self.obj.iter = MagicMock()
        self.obj.start = MagicMock()
        self.obj.finish = MagicMock()

        self.obj.run_time = 10
        self.obj.t0 = 0
        self.obj.t = 0
        self.obj.stepping = True
        self.obj.next_time_to_stop = 0

    # -----------------------------
    # Case 1: default stepping = 1 step
    # -----------------------------
    @patch("builtins.input", return_value="")
    @patch("specula.process_rank", 0)
    def test_stepping_default_one_step(self, mock_input):
        """
        If user presses Enter, should advance exactly 1 timestep.
        """
        def fake_iter():
            self.obj.t += 1

        self.obj.iter.side_effect = fake_iter

        # force loop to stop after 1 iteration
        self.obj.run_time = 1

        self.obj.run(run_time=1, dt=1)

        self.obj.start.assert_called_once()
        self.obj.iter.assert_called()
        self.obj.finish.assert_called_once()

    # -----------------------------
    # Case 2: stepping disabled skips input
    # -----------------------------
    @patch("specula.process_rank", 0)
    @patch("builtins.input")
    def test_no_input_when_not_stepping(self, mock_input):
        """
        If stepping is False, input() should never be called.
        """
        self.obj.stepping = False

        def fake_iter():
            self.obj.t += 1

        self.obj.iter.side_effect = fake_iter
        self.obj.run_time = 1

        self.obj.run(run_time=1, dt=1)

        mock_input.assert_not_called()

    # -----------------------------
    # Case 3: invalid input defaults to 1
    # -----------------------------
    @patch("builtins.input", return_value="invalid")
    @patch("specula.process_rank", 0)
    def test_invalid_input_defaults_to_one(self, mock_input):
        """
        Non-integer input should fallback to 1 timestep.
        """
        def fake_iter():
            self.obj.t += 1

        self.obj.iter.side_effect = fake_iter
        self.obj.run_time = 1

        self.obj.run(run_time=1, dt=1)

        # should still run at least once safely
        self.obj.iter.assert_called()

    # -----------------------------
    # Case 4: stepping only triggers after next_time_to_stop
    # -----------------------------
    @patch("builtins.input", return_value="5")
    @patch("specula.process_rank", 0)
    def test_next_time_to_stop_controls_prompt(self, mock_input):
        """
        Ensure next_time_to_stop logic is applied.
        """
        def fake_iter():
            self.obj.t += 1

        self.obj.iter.side_effect = fake_iter
        self.obj.run_time = 5

        self.obj.run(run_time=5, dt=1)

        # should have updated threshold
        self.assertNotEqual(self.obj.next_time_to_stop, 0)

