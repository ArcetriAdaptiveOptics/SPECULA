import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import yaml
import specula
specula.init(0)  # Default target device

from specula import np
from specula.simul import Simul
from specula.ef_replay import EfReplay
from astropy.io import fits


class TestEfReplayWeakSpots(unittest.TestCase):
    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self._created_tn_dirs = []

    def tearDown(self):
        for tn_dir in self._created_tn_dirs:
            if os.path.isdir(tn_dir):
                shutil.rmtree(tn_dir, ignore_errors=True)

    def _make_tn(self, tn_name, params=None):
        tn_dir = os.path.join(self.datadir, f'efreplay_unit_{tn_name}')
        os.makedirs(tn_dir, exist_ok=True)
        self._created_tn_dirs.append(tn_dir)

        if params is None:
            params = {
                'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
                'prop': {'class': 'AtmoPropagation'},
                'ef_combinator': {
                    'class': 'ElectricFieldCombinator',
                    'inputs': {'in_ef1': 'prop.out_ef'},
                    'outputs': ['out_ef'],
                },
            }

        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as handle:
            yaml.dump(params, handle)

        return tn_dir, f'efreplay_unit_{tn_name}'

    def test_unknown_target_object_raises_keyerror(self):
        tn_dir, tn_name = self._make_tn('unknown_target')

        with self.assertRaises(KeyError):
            EfReplay(
                data_dir=self.datadir,
                tracking_number=tn_name,
                output_refs=['does_not_exist.out_ef'],
                log_level=logging.INFO,
            )

    def test_targets_are_deduplicated_object_names(self):
        tn_dir, tn_name = self._make_tn('dedup_targets')

        replay = EfReplay(
            data_dir=self.datadir,
            tracking_number=tn_name,
            output_refs=['ef_combinator.out_ef', 'ef_combinator.out_ef', 'prop.out_ef'],
            log_level=logging.INFO,
        )

        self.assertEqual(replay._targets, ['ef_combinator', 'prop'])

    def test_output_dir_naming(self):
        tn_dir, tn_name = self._make_tn('output_dir_naming')

        replay = EfReplay(
            data_dir=self.datadir,
            tracking_number=tn_name,
            output_refs=['prop.out_ef'],
            log_level=logging.INFO,
        )

        self.assertEqual(replay.replay_output_dir, replay.base_output_dir / f'{tn_name}_EFREPLAY')

    def test_compute_replay_builds_expected_datastore_and_targets(self):
        tn_dir, tn_name = self._make_tn('compute_replay_config')

        replay = EfReplay(
            data_dir=self.datadir,
            tracking_number=tn_name,
            output_refs=['ef_combinator.out_ef', 'prop.out_ef'],
            log_level=logging.INFO,
        )

        fake_replay_params = {
            'main': {'class': 'SimulParams'},
            'prop': {'class': 'AtmoPropagation'},
            'ef_combinator': {'class': 'ElectricFieldCombinator'},
        }

        with patch.object(replay, '_build_replay_params_from_datastore',
                           return_value=fake_replay_params) as mock_build, \
             patch.object(replay, '_run_simulation_with_params') as mock_run, \
             patch.object(replay, '_load_efreplay_results', return_value={'ok': True}) as mock_load:
            result = replay.compute_replay(force_recompute=True)

        mock_build.assert_called_once_with('ef_combinator', 'prop')
        used_params = mock_run.call_args.args[0]
        self.assertEqual(
            set(used_params['data_store_efreplay']['inputs']['input_list']),
            {'ef_combinator_out_ef-ef_combinator.out_ef', 'prop_out_ef-prop.out_ef'}
        )
        self.assertEqual(result, {'ok': True})


class TestEfReplayEndToEnd(unittest.TestCase):
    """
    End-to-end proof that EfReplay reproduces an existing object's output
    exactly, bypassing anything not needed to recompute it -- here, a DM
    driven by a recorded command, entirely independent of RNG/noise (unlike
    the WFS-in-the-loop fixture used by TestShSimulation), so the comparison
    is a clean bit-for-bit match.
    """

    def setUp(self):
        self.orig_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.orig_dir, ignore_errors=True)

    def test_ef_replay_reproduces_dm_out_layer_from_recorded_command(self):
        yml = f'''
main:
  class: SimulParams
  root_dir: dummy
  total_time: 0.003
  time_step: 0.001
  pixel_pupil: 8
  pixel_pitch: 1.0

gen:
  class: WaveGenerator
  constant: [100.0, 50.0]
  outputs: ['output']

dm:
  class: DM
  simul_params_ref: main
  type_str: zernike
  nmodes: 2
  npixels: 8
  obsratio: 0.0
  height: 0
  inputs:
    in_command: gen.output
  outputs: ['out_layer']

store:
  class: DataStore
  store_dir: {self.orig_dir}
  create_tn: false
  inputs:
    input_list: ['comm-gen.output', 'origlayer-dm.out_layer']
'''
        fd, path = tempfile.mkstemp(suffix='.yml')
        with os.fdopen(fd, 'w') as f:
            f.write(yml)
        try:
            Simul(path).run()
        finally:
            os.unlink(path)

        data_dir = os.path.dirname(self.orig_dir)
        tracking_number = os.path.basename(self.orig_dir)

        replay = EfReplay(
            data_dir=data_dir,
            tracking_number=tracking_number,
            output_refs=['dm.out_layer'],
            start_time=0.0,
            log_level=logging.INFO,
        )
        results = replay.compute_replay(force_recompute=True)

        replayed = results['dm.out_layer']['data']
        original = fits.getdata(os.path.join(self.orig_dir, 'origlayer.fits'))

        np.testing.assert_array_equal(replayed, original)

        shutil.rmtree(replay.replay_output_dir, ignore_errors=True)
