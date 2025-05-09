
import os
import specula
specula.init(0)  # Default target device

import unittest
import tempfile

from specula.processing_objects.data_store import DataStore

from test.specula_testlib import cpu_and_gpu


class TestDataStore(unittest.TestCase):

    @unittest.skip
    @cpu_and_gpu
    def test_replay_params_is_skipped_if_not_set(self, target_device_idx, xp):

        ds = DataStore(tempfile.gettempdir())
        ds.setParams({})
        # Does not raise
        ds.finalize()  

    @unittest.skip
    @cpu_and_gpu
    def test_replay_params(self, target_device_idx, xp):

        ds = DataStore(tempfile.gettempdir())
        params = {'foo': bar}
        ds.setParams({})
        # TODO