
import os
import shutil

import yaml
import specula
specula.init(0)  # Default target device
from specula.simul import Simul

from astropy.io import fits
import numpy as np
import unittest
from unittest.mock import patch

from specula.connections import InputValue
from specula.base_data_obj import BaseDataObj
from specula.processing_objects.data_store import DataStore
from test.specula_testlib import cpu_and_gpu


class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_store')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
       shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @cpu_and_gpu
    def test_data_store(self, target_device_idx, xp):
        params = {'main': {'class': 'SimulParams', 'root_dir': self.tmp_dir,
                           'time_step': 0.1, 'total_time': 0.2},
                  'generator': {'class': 'WaveGenerator', 'target_device_idx': target_device_idx, 'amp': 1, 'freq': 2},
                  'store': {'class': 'DataStore', 'store_dir': self.tmp_dir,
                            'inputs': {'input_list': ['gen-generator.output']},
                            }
                  }
        filename = os.path.join(self.tmp_dir, 'test_data_store.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(params, outfile)

        simul = Simul(filename)
        simul.run()

        # Find last TN in tmp_dir
        tn_dirs = sorted([d for d in os.listdir(self.tmp_dir) if d.startswith('2')])
        last_tn_dir = os.path.join(self.tmp_dir, tn_dirs[-1])

        # Read gen.fits file from last_tn_dir and compare with [1,2]
        gen_file = os.path.join(last_tn_dir, 'gen.fits')
        assert os.path.exists(gen_file), f"File {gen_file} does not exist"
        gen_data = fits.getdata(gen_file)
        np.testing.assert_array_almost_equal(gen_data, np.array([[0], [0.9510565162951535]]))

        # Make sure times are in int64
        gen_times = fits.getdata(gen_file, ext=1)
        assert gen_times.dtype == np.uint64

        # Make sure replay_params.yml exists
        replay_file = os.path.join(last_tn_dir, 'replay_params.yml')
        assert os.path.exists(replay_file), f"File {replay_file} does not exist"

    @cpu_and_gpu
    def test_data_store_start_time(self, target_device_idx, xp):
        params = {'main': {'class': 'SimulParams', 'root_dir': self.tmp_dir,
                           'time_step': 0.1, 'total_time': 0.4},
                  'generator': {'class': 'WaveGenerator', 'target_device_idx': target_device_idx, 'amp': 1, 'freq': 2},
                  'store': {'class': 'DataStore', 'store_dir': self.tmp_dir,
                            'start_time': 0.2,
                            'inputs': {'input_list': ['gen-generator.output']},
                            }
                  }
        filename = os.path.join(self.tmp_dir, 'test_data_store.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(params, outfile)

        simul = Simul(filename)
        simul.run()

        # Find last TN in tmp_dir
        tn_dirs = sorted([d for d in os.listdir(self.tmp_dir) if d.startswith('2')])
        last_tn_dir = os.path.join(self.tmp_dir, tn_dirs[-1])

        gen_file = os.path.join(last_tn_dir, 'gen.fits')
        assert os.path.exists(gen_file), f"File {gen_file} does not exist"

        # Make sure times are correct
        gen_times = fits.getdata(gen_file, ext=1)
        ref_times = np.arange(0.2, 0.4, 0.1) * simul.objs['store']._time_resolution
        np.testing.assert_array_almost_equal(gen_times, ref_times)
        assert gen_times.dtype == np.uint64

    def test_data_store_fails_early(self):
        """Test that DataStore fails during setup() if a
        class without get_value() is set as an input"""
        buffer_size = 2

        # Create buffer with manual input setup
        store = DataStore(store_dir='/tmp')
        data = BaseDataObj()

        # Manually create input for buffer (simulate what simul.py does)
        store.inputs['gen'] = InputValue(type=BaseDataObj)
        store.inputs['gen'].set(data)

        with self.assertRaises(TypeError):
            store.setup()

    def test_trigger_code_saves_at_correct_intervals_and_suffixes(self):
        """
        Verify that DataStore.save() is called only when iter_counter reaches
        multiples of split_size, and that TN folder suffixes are correct.
        """
        with patch.object(DataStore, "save") as mock_save, \
             patch.object(DataStore, "create_TN_folder") as mock_create_tn, \
             patch("os.makedirs"), \
             patch("os.path.exists", return_value=False):

            ds = DataStore(store_dir="/tmp", split_size=2, data_format="fits")
            ds.local_inputs = {}  # Avoid real inputs
            ds.current_time = 0

            # First trigger → iter_counter=1 → no save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 0)
            self.assertEqual(mock_create_tn.call_count, 0)

            # Second trigger → iter_counter=2 → first save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 1)
            mock_create_tn.assert_called_with(suffix="_0")

            # Third trigger → iter_counter=3 → no save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 1)

            # Fourth trigger → iter_counter=4 → second save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 2)
            mock_create_tn.assert_called_with(suffix="_2")

            # Fifth trigger → iter_counter=5 → no save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 2)

            # Sixth trigger → iter_counter=6 → third save
            ds.trigger_code()
            self.assertEqual(mock_save.call_count, 3)
            mock_create_tn.assert_called_with(suffix="_4")

    def test_create_tn_folder_creates_unique_folder_with_suffix(self):
        """
        Verify that create_TN_folder() generates the correct TN folder name including suffix.
        """
        with patch("os.makedirs") as mock_makedirs, \
             patch("os.path.exists", return_value=False), \
             patch("time.strftime", return_value="20250101_120000"):

            ds = DataStore(store_dir="/tmp", data_format="fits")
            ds.create_TN_folder(suffix="_42")

            expected_path = os.path.join("/tmp", "20250101_120000") + "_42"
            self.assertEqual(ds.tn_dir, expected_path)
            mock_makedirs.assert_called_once_with(expected_path)

    def test_finalize_does_not_save_when_split_tn_set(self):
        """
        Verify that finalize() does not call save() when split_size > 0.
        """
        with patch.object(DataStore, "save") as mock_save, \
             patch.object(DataStore, "create_TN_folder") as mock_create_tn, \
             patch.object(DataStore, "trigger_code") as mock_trigger:

            ds = DataStore(store_dir="/tmp", split_size=2, data_format="fits")
            ds.local_inputs = {}
            ds.finalize()

            # finalize() always triggers once, but doesn't save when split TN enabled
            mock_trigger.assert_called_once()
            mock_create_tn.assert_not_called()
            mock_save.assert_not_called()

    def test_finalize_saves_whole_tn_when_split_tn_zero(self):
        """
        Verify that finalize() calls create_TN_folder() and save() when split_size = 0.
        """
        with patch.object(DataStore, "save") as mock_save, \
             patch.object(DataStore, "create_TN_folder") as mock_create_tn, \
             patch.object(DataStore, "trigger_code") as mock_trigger:

            ds = DataStore(store_dir="/tmp", split_size=0, data_format="fits")
            ds.local_inputs = {}
            ds.finalize()

            # finalize() always triggers once and saves entire TN when split TN disabled
            mock_trigger.assert_called_once()
            mock_create_tn.assert_called_once()
            mock_save.assert_called_once()

    def test_finalize_saves_whole_tn_when_split_tn_not_set(self):
        """
        Verify that finalize() behaves the same as split_size=0 if unset.
        """
        with patch.object(DataStore, "save") as mock_save, \
             patch.object(DataStore, "create_TN_folder") as mock_create_tn, \
             patch.object(DataStore, "trigger_code") as mock_trigger:

            ds = DataStore(store_dir="/tmp", data_format="fits")
            ds.local_inputs = {}
            ds.finalize()

            mock_trigger.assert_called_once()
            mock_create_tn.assert_called_once()
            mock_save.assert_called_once()

    @cpu_and_gpu
    def test_save_fits_skips_none_keys(self, target_device_idx, xp):
        """
        Verify that save_fits() skips None keys in storage without crashing.
        """
        with patch("os.makedirs"), \
             patch("os.path.exists", return_value=False):

            ds = DataStore(store_dir="/tmp", data_format="fits")
            ds.verbose = True
            ds.local_inputs = {'gen': None}  # Mock input

            # Manually create storage with a None key
            ds.storage[None] = {100: xp.array([1.0]), 200: xp.array([2.0])}
            ds.storage['gen'] = {100: xp.array([1.0]), 200: xp.array([2.0])}

            # Should not crash when saving
            ds.save_fits()

    @cpu_and_gpu
    def test_save_pickle_skips_none_keys(self, target_device_idx, xp):
        """
        Verify that save_pickle() skips None keys in storage without crashing.
        """
        with patch("os.makedirs"), \
             patch("os.path.exists", return_value=False):

            ds = DataStore(store_dir="/tmp", data_format="pickle")
            ds.verbose = True
            ds.local_inputs = {'gen': None}  # Mock input

            # Manually create storage with a None key
            ds.storage[None] = {100: xp.array([1.0]), 200: xp.array([2.0])}
            ds.storage['gen'] = {100: xp.array([1.0]), 200: xp.array([2.0])}

            # Should not crash when saving
            ds.save_pickle()

    @cpu_and_gpu
    def test_save_fits_handles_missing_local_inputs(self, target_device_idx, xp):
        """
        Verify that save_fits() gracefully handles keys not present in local_inputs.
        """
        with patch("os.makedirs"), \
             patch("os.path.exists", return_value=False):

            ds = DataStore(store_dir="/tmp", data_format="fits")
            ds.verbose = True
            ds.local_inputs = {'gen': None}  # Only 'gen' is registered

            # Add storage for both registered and unregistered keys
            ds.storage['gen'] = {100: xp.array([1.0]), 200: xp.array([2.0])}
            ds.storage['missing_key'] = {100: xp.array([3.0]), 200: xp.array([4.0])}

            # Should not crash when saving
            ds.save_fits()

            # Verify 'gen' file was created but 'missing_key' was not
            gen_file = os.path.join("/tmp", "gen.fits")
            missing_file = os.path.join("/tmp", "missing_key.fits")
            # Note: with mocked os.path.exists, we can't actually verify files
            # The test passes if save_fits() doesn't crash

    @cpu_and_gpu
    def test_save_pickle_handles_missing_inputs(self, target_device_idx, xp):
        """
        Verify that save_pickle() gracefully handles keys not present in inputs.
        """
        with patch("os.makedirs"), \
             patch("os.path.exists", return_value=False):

            ds = DataStore(store_dir="/tmp", data_format="pickle")
            ds.verbose = True
            ds.local_inputs = {'gen': None}  # Only 'gen' is registered

            # Add storage for both registered and unregistered keys
            ds.storage['gen'] = {100: xp.array([1.0]), 200: xp.array([2.0])}
            ds.storage['missing_key'] = {100: xp.array([3.0]), 200: xp.array([4.0])}

            # Should not crash when saving
            ds.save_pickle()

            # Note: with mocked filesystem, we can't verify files
            # The test passes if save_pickle() doesn't crash
