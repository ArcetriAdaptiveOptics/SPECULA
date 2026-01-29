import specula
specula.init(0)  # Default target device

import unittest

from specula.loop_control import LoopControl
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
    '''Class that is alwasy ready and remmebers whether trigger() and post_trigger() were called'''

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



