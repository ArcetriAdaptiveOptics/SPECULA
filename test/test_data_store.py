import os
import shutil

import yaml
import specula
specula.init(0)  # Default target device
from specula.simul import Simul

from astropy.io import fits
import numpy as np
import pickle
import unittest
from unittest.mock import patch

from specula.connections import InputValue
from specula.loop_control import LoopControl
from specula.base_data_obj import BaseDataObj
from specula.base_value import BaseValue
from specula.processing_objects.data_store import DataStore
from test.specula_testlib import cpu_and_gpu


class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_store')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @staticmethod
    def _make_input_value(target_device_idx):
        return BaseValue(target_device_idx=target_device_idx)

    @staticmethod
    def _connect_input(store, name, value):
        store.inputs[name] = InputValue(type=BaseValue)
        store.inputs[name].set(value)

    @staticmethod
    def _stored_scalars(store, name):
        values = list(store.storage[name].values())
        if not values:
            return np.array([])
        return np.array([np.asarray(value).reshape(-1)[0] for value in values])

    @cpu_and_gpu
    def test_data_store_global_downsampling(self, target_device_idx, xp):
        store = DataStore(
            store_dir=self.tmp_dir,
            create_tn=False,
            downsample_factor=2
        )
        store.target_device_idx = target_device_idx

        fast = self._make_input_value(target_device_idx)
        slow = self._make_input_value(target_device_idx)
        self._connect_input(store, 'fast', fast)
        self._connect_input(store, 'slow', slow)
        store.setup()

        loop = LoopControl()
        loop.add(store, idx=0)
        loop.start(run_time=store.t_to_seconds(6), dt=store.t_to_seconds(1))

        for t in range(6):
            fast.set_value(np.array([10 + t], dtype=np.float32))
            fast.generation_time = t 
            slow.set_value(np.array([20 + t], dtype=np.float32))
            slow.generation_time = t
            loop.iter()

        self.assertEqual(list(store.storage['fast'].keys()), [0, 2, 4])
        self.assertEqual(list(store.storage['slow'].keys()), [0, 2, 4])
        np.testing.assert_array_equal(self._stored_scalars(store, 'fast'), np.array([10, 12, 14]))
        np.testing.assert_array_equal(self._stored_scalars(store, 'slow'), np.array([20, 22, 24]))

    @cpu_and_gpu
    def test_data_store_per_input_downsampling(self, target_device_idx, xp):
        store = DataStore(
            store_dir=self.tmp_dir,
            create_tn=False,
            downsample_factor_by_input={'slow': 3}
        )
        store.target_device_idx = target_device_idx

        fast = self._make_input_value(target_device_idx)
        slow = self._make_input_value(target_device_idx)
        self._connect_input(store, 'fast', fast)
        self._connect_input(store, 'slow', slow)
        store.setup()

        for t in range(6):
            fast.set_value(np.array([10 + t], dtype=np.float32))
            fast.generation_time = t
            slow.set_value(np.array([20 + t], dtype=np.float32))
            slow.generation_time = t

            store.check_ready(t)
            store.trigger()
            store.post_trigger()

        self.assertEqual(list(store.storage['fast'].keys()), [0, 1, 2, 3, 4, 5])
        self.assertEqual(list(store.storage['slow'].keys()), [0, 3])
        np.testing.assert_array_equal(self._stored_scalars(store, 'fast'),
                                      np.array([10, 11, 12, 13, 14, 15]))
        np.testing.assert_array_equal(self._stored_scalars(store, 'slow'),
                                      np.array([20, 23]))

    @cpu_and_gpu
    def test_data_store_downsampling_counts_samples_per_input(self, target_device_idx, xp):
        store = DataStore(
            store_dir=self.tmp_dir,
            create_tn=False,
            downsample_factor=1,
            downsample_factor_by_input={'sparse': 2}
        )
        store.target_device_idx = target_device_idx

        sparse = self._make_input_value(target_device_idx)
        self._connect_input(store, 'sparse', sparse)
        store.setup()

        for t in range(7):
            if t % 2 == 0:
                sparse.set_value(np.array([100 + t], dtype=np.float32))
                sparse.generation_time = t

            if store.check_ready(t):
                store.trigger()
                store.post_trigger()

        self.assertEqual(list(store.storage['sparse'].keys()), [0, 4])
        np.testing.assert_array_equal(self._stored_scalars(store, 'sparse'), np.array([100, 104]))

    def test_data_store_rejects_invalid_downsampling_factor(self):
        with self.assertRaises(ValueError):
            DataStore(store_dir=self.tmp_dir, downsample_factor=0)

        with self.assertRaises(ValueError):
            DataStore(store_dir=self.tmp_dir, downsample_factor_by_input={'fast': 0})

        with self.assertRaises(ValueError):
            DataStore(store_dir=self.tmp_dir,
                      downsample_factor=2,
                      downsample_factor_by_input={'fast': 3})

    @cpu_and_gpu
    def test_data_store(self, target_device_idx, xp):
        params = {'main': {'class': 'SimulParams', 'root_dir': self.tmp_dir,
                           'time_step': 0.1, 'total_time': 0.2},
                  'generator': {'class': 'WaveGenerator',
                                'target_device_idx': target_device_idx,
                                'amp': 1, 'freq': 2},
                  'store': {'class': 'DataStore',
                            'store_dir': self.tmp_dir,
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

        gen_header = fits.getheader(gen_file)
        self.assertEqual(gen_header['DOWNSAMP'], 1)
        self.assertEqual(gen_header['DSMODE'], 'SAMPLE')

        # Make sure replay_params.yml exists
        replay_file = os.path.join(last_tn_dir, 'replay_params.yml')
        assert os.path.exists(replay_file), f"File {replay_file} does not exist"

    def test_data_store_pickle_writes_downsampling_metadata(self):
        store = DataStore(
            store_dir=self.tmp_dir,
            data_format='pickle',
            create_tn=False,
            downsample_factor_by_input={'fast': 3}
        )
        value = self._make_input_value(-1)
        self._connect_input(store, 'fast', value)
        store.setup()

        value.set_value(np.array([1.0], dtype=np.float32))
        value.generation_time = 0
        store.check_ready(0)
        store.trigger()
        store.post_trigger()
        store.save_pickle()

        with open(os.path.join(self.tmp_dir, 'fast.pickle'), 'rb') as handle:
            payload = pickle.load(handle)

        self.assertEqual(payload['hdr']['DOWNSAMP'], 3)
        self.assertEqual(payload['hdr']['DSMODE'], 'SAMPLE')

    def test_save_params_writes_global_precision_in_replay_params(self):
        store = DataStore(store_dir=self.tmp_dir, create_tn=False)
        store.params = {'main': {'class': 'SimulParams'}}
        store.replay_params = {
            'data_source': {
                'class': 'DataSource',
                'store_dir': '/tmp',
                'outputs': []
            }
        }
        store.tn_dir = self.tmp_dir
        store.precision = 1

        store.save_params()

        replay_file = os.path.join(self.tmp_dir, 'replay_params.yml')
        self.assertTrue(os.path.exists(replay_file))
        with open(replay_file, 'r', encoding='utf-8') as handle:
            replay_cfg = yaml.safe_load(handle)
        self.assertEqual(replay_cfg['data_source']['global_precision'], 1)

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
