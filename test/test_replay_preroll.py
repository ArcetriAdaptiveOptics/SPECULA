import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import specula
specula.init(0)  # Default target device

from specula import np
from specula.simul import Simul
from specula.loop_control import LoopControl
from specula.field_analyser import FieldAnalyser
from astropy.io import fits


class TestFindPrerollObjects(unittest.TestCase):
    """
    Unit tests for Simul.find_preroll_objects on a synthetic replay params dict.

    Graph (a typical build_targeted_replay output):
      gen -> atmo -> prop            (case a: ancestors of the replayed 'prop', not
                                       themselves fed by data_source)
      data_source -> dm -[:-1]-> prop (case b: 'dm' is fed by data_source directly,
                                       so it is 'replayed'; its output is consumed with
                                       a negative delay by 'prop', also replayed, so
                                       'dm' -and its own ancestor 'data_source'- must be
                                       pre-rolled too)
      pupilstop -> prop               (data object ancestor: must be excluded)
      gen -> gen_disp                 (display fed only by a generator: not an
                                       ancestor of prop, must be excluded)
      gen -> monitor_store             (DataStore fed only by a generator: excluded,
                                       and DataStore-like objects are never pre-rolled)
    """

    def setUp(self):
        self.simul = Simul('dummy.yaml')

    def test_find_preroll_objects(self):
        params = {
            'main': {'class': 'SimulParams'},
            'data_source': {'class': 'DataSource', 'store_dir': 'dummy', 'outputs': ['command']},
            'gen': {'class': 'WaveGenerator', 'outputs': ['output']},
            'gen_disp': {'class': 'PhaseDisplay', 'inputs': {'phase': 'gen.output'}},
            'atmo': {'class': 'AtmoEvolution', 'inputs': {'seeing': 'gen.output'}, 'outputs': ['layer_list']},
            'dm': {'class': 'DM', 'inputs': {'in_command': 'data_source.command'}, 'outputs': ['out_layer']},
            'pupilstop': {'class': 'Pupilstop'},
            'prop': {
                'class': 'AtmoPropagation',
                'inputs': {
                    'atmo_layer_list': ['atmo.layer_list'],
                    'common_layer_list': ['pupilstop', 'dm.out_layer:-1'],
                },
            },
            'monitor_store': {'class': 'DataStore', 'inputs': {'input_list': ['g-gen.output']}},
        }
        # Only 'pupilstop' is a data object among the objects reachable from 'prop';
        # everything else defaults to False (as it would for real processing objects).
        self.simul.is_dataobj = {'pupilstop': True}

        result = self.simul.find_preroll_objects(params)

        # Order must follow the params dict order, not discovery order.
        self.assertEqual(result, ['data_source', 'gen', 'atmo', 'dm'])
        self.assertNotIn('prop', result, "the replayed target itself must not be pre-rolled")
        self.assertNotIn('pupilstop', result, "data objects are stateless and must be excluded")
        self.assertNotIn('gen_disp', result, "a display fed only by a generator is not an ancestor of prop")
        self.assertNotIn('monitor_store', result, "DataStore objects must never be pre-rolled")

    def test_find_preroll_objects_empty_when_nothing_feeds_replayed_part(self):
        """A generator/display branch disconnected from the replayed object yields no pre-roll."""
        params = {
            'main': {'class': 'SimulParams'},
            'data_source': {'class': 'DataSource', 'store_dir': 'dummy', 'outputs': ['command']},
            'prop': {'class': 'AtmoPropagation', 'inputs': {'common_layer_list': ['data_source.command']}},
            'gen': {'class': 'WaveGenerator', 'outputs': ['output']},
            'gen_disp': {'class': 'PhaseDisplay', 'inputs': {'phase': 'gen.output'}},
        }
        result = self.simul.find_preroll_objects(params)
        self.assertEqual(result, [])


class TestLoopControlPreroll(unittest.TestCase):

    def test_preroll_raises_when_t0_not_multiple_of_dt(self):
        loop = LoopControl()
        loop.dt = loop.seconds_to_t(0.001)
        loop.t0 = loop.seconds_to_t(0.0035)  # 3.5 steps: not a multiple of dt

        with self.assertRaises(ValueError):
            loop.preroll(['some_obj'])

    def test_start_skips_preroll_with_empty_list(self):
        """Without objects to pre-roll (non-replay runs), t0 need not be a multiple of dt."""
        loop = LoopControl()
        loop.start(run_time=0.01, dt=0.001, t0=0.0035, preroll_objs=[])
        self.assertEqual(loop.t, loop.seconds_to_t(0.0035))


class TestCheckPrerollIsLocal(unittest.TestCase):

    def setUp(self):
        self.simul = Simul('dummy.yaml')

    def test_raises_when_preroll_object_is_on_a_remote_rank(self):
        self.simul.remote_objs_ranks = {'remote_gen': 1}

        with self.assertRaises(NotImplementedError):
            self.simul._check_preroll_is_local(['remote_gen'])

    def test_raises_when_preroll_object_has_remote_outputs(self):
        class FakeObj:
            remote_outputs = ['some_rank']

        self.simul.objs = {'local_but_shared': FakeObj()}

        with self.assertRaises(NotImplementedError):
            self.simul._check_preroll_is_local(['local_but_shared'])

    def test_does_not_raise_for_purely_local_objects(self):
        class FakeObj:
            remote_outputs = None

        self.simul.remote_objs_ranks = {}
        self.simul.objs = {'local': FakeObj()}

        # Must not raise
        self.simul._check_preroll_is_local(['local'])


class TestReplayPrerollEndToEnd(unittest.TestCase):
    """
    End-to-end regression for SPECULA pre-roll: FieldAnalyser.compute_phase_cube
    with start_time > 0 must reproduce the original 'prop' phase on every frame of
    the replay window, including the first one.

    The graph exercises both kinds of pre-rolled objects:
      - gen_time (TimeHistoryGenerator) -> dm_time -> prop           (iter_counter state)
      - rand (RandomGenerator)          -> dm_rand -> prop           (RNG draw state)
      - gen_store (recorded by DataStore) -> dm_cmd -[:-1]-> prop    (delayed replayed
        consumer: dm_cmd is fed by data_source in the replay, and its own output must
        be ready *before* the first replayed step, or prop's first frame reads a
        never-computed value)
    """

    @classmethod
    def setUpClass(cls):
        cls.dt = 0.001
        cls.total_time = 0.008
        cls.start_time = 0.004  # multiple of dt, > 0
        cls.n_steps = round(cls.total_time / cls.dt)
        cls.t0_idx = round(cls.start_time / cls.dt)

        cls.orig_dir = tempfile.mkdtemp()
        yml = f'''
main:
  class: SimulParams
  root_dir: dummy
  total_time: {cls.total_time}
  time_step: {cls.dt}
  pixel_pupil: 16
  pixel_pitch: 0.1

time_hist:
  class: TimeHistory
  time_history: [[10, 20], [11, 19], [12, 18], [13, 17], [14, 16],
                 [15, 15], [16, 14], [17, 13], [18, 12], [19, 11]]

gen_time:
  class: TimeHistoryGenerator
  time_hist_ref: time_hist
  outputs: ['output']

dm_time:
  class: DM
  simul_params_ref: main
  type_str: zernike
  nmodes: 2
  npixels: 16
  obsratio: 0.0
  height: 0
  inputs:
    in_command: gen_time.output
  outputs: ['out_layer']

rand:
  class: RandomGenerator
  seed: 123
  amp: [5.0, 5.0]
  output_size: 2
  outputs: ['output']

dm_rand:
  class: DM
  simul_params_ref: main
  type_str: zernike
  nmodes: 2
  npixels: 16
  obsratio: 0.0
  height: 0
  inputs:
    in_command: rand.output
  outputs: ['out_layer']

gen_store:
  class: RandomGenerator
  seed: 456
  amp: [3.0, 3.0]
  output_size: 2
  outputs: ['output']

dm_cmd:
  class: DM
  simul_params_ref: main
  type_str: zernike
  nmodes: 2
  npixels: 16
  obsratio: 0.0
  height: 0
  inputs:
    in_command: gen_store.output
  outputs: ['out_layer']

on_axis_source:
  class: Source
  polar_coordinates: [0.0, 0.0]
  magnitude: 8
  wavelengthInNm: 750

prop:
  class: AtmoPropagation
  simul_params_ref: main
  source_dict_ref: ['on_axis_source']
  inputs:
    common_layer_list: ['dm_time.out_layer', 'dm_rand.out_layer', 'dm_cmd.out_layer:-1']
  outputs: ['out_on_axis_source_ef']

data_store:
  class: DataStore
  store_dir: {cls.orig_dir}
  create_tn: false
  inputs:
    input_list: ['cmd-gen_store.output', 'phase-prop.out_on_axis_source_ef']
'''
        fd, path = tempfile.mkstemp(suffix='.yml')
        with os.fdopen(fd, 'w') as f:
            f.write(yml)
        try:
            Simul(path).run()
        finally:
            os.unlink(path)

        with fits.open(os.path.join(cls.orig_dir, 'phase.fits')) as hdul:
            cls.original_phase = hdul[0].data.copy()   # pylint: disable=no-member
        cls.expected_window = cls.original_phase[cls.t0_idx:]

        cls.data_dir = os.path.dirname(cls.orig_dir)
        cls.tracking_number = os.path.basename(cls.orig_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.orig_dir, ignore_errors=True)

    def _make_analyzer(self):
        return FieldAnalyser(
            data_dir=self.data_dir,
            tracking_number=self.tracking_number,
            polar_coordinates=np.array([[0.0, 0.0]]),
            wavelength_nm=750,
            start_time=self.start_time,
            end_time=None,
            log_level=logging.INFO,
        )

    def tearDown(self):
        for suffix in ('_PSF', '_MA', '_CUBE'):
            path = os.path.join(self.data_dir, f'{self.tracking_number}{suffix}')
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

    def test_replay_with_preroll_matches_original_phase_on_every_frame(self):
        """Regression test for the pre-roll fix: no monkeypatching, this is the
        behaviour that must hold with the library code as shipped."""
        analyzer = self._make_analyzer()

        cube_results = analyzer.compute_phase_cube(force_recompute=True)
        replayed_phase = cube_results['phase_cubes'][0]

        self.assertEqual(replayed_phase.shape, self.expected_window.shape)
        np.testing.assert_allclose(
            replayed_phase, self.expected_window, rtol=1e-4, atol=1e-4,
            err_msg='Replayed phase cube does not match the original run on the pre-rolled window'
        )

    def test_replay_without_any_preroll_diverges_from_original(self):
        """
        Negative/discriminating test: simulates pre-fix behaviour by forcing
        Simul.find_preroll_objects to return an empty list (as it effectively was
        before this feature existed). All pre-rolled generators/DMs then start their
        internal iteration/RNG state from scratch at t0 instead of matching the
        original run, so every frame of the replay window must now be wrong.
        """
        analyzer = self._make_analyzer()

        with patch.object(Simul, 'find_preroll_objects', return_value=[]):
            cube_results = analyzer.compute_phase_cube(force_recompute=True)
        replayed_phase = cube_results['phase_cubes'][0]

        per_frame_max_diff = np.abs(replayed_phase - self.expected_window).reshape(
            replayed_phase.shape[0], -1).max(axis=1)
        self.assertTrue(
            np.all(per_frame_max_diff > 1e-2),
            f'Expected every frame to diverge without pre-roll, got per-frame max diff {per_frame_max_diff}'
        )

    def test_replay_with_only_non_replayed_ancestors_diverges_on_first_frame(self):
        """
        Negative/discriminating test: pre-roll only the plain ancestors of 'prop'
        (gen_time, dm_time, gen_rand, dm_rand), i.e. skip the delayed-replayed-consumer
        extension that also pre-rolls 'dm_cmd' (and 'data_source'). 'dm_cmd' feeds prop
        with ':-1', so without pre-rolling it, only the very first replayed frame reads
        a never-computed (zero-initialized) 'dm_cmd' output; later frames are fine
        because dm_cmd gets triggered normally by the running loop from then on.
        """
        analyzer = self._make_analyzer()
        replay_params = analyzer._build_replay_params_cube()
        full_preroll = Simul('dummy.yaml').find_preroll_objects(replay_params)
        self.assertIn('dm_cmd', full_preroll)
        self.assertIn('data_source', full_preroll)
        partial_preroll = [name for name in full_preroll if name not in ('dm_cmd', 'data_source')]
        self.assertIn('dm_time', partial_preroll)
        self.assertIn('gen_time', partial_preroll)

        with patch.object(Simul, 'find_preroll_objects', return_value=partial_preroll):
            cube_results = analyzer.compute_phase_cube(force_recompute=True)
        replayed_phase = cube_results['phase_cubes'][0]

        per_frame_max_diff = np.abs(replayed_phase - self.expected_window).reshape(
            replayed_phase.shape[0], -1).max(axis=1)

        self.assertGreater(
            per_frame_max_diff[0], 1e-2,
            'First replayed frame should diverge when the delayed-consumer dm_cmd is not pre-rolled'
        )
        np.testing.assert_allclose(
            per_frame_max_diff[1:], 0.0, atol=1e-4,
            err_msg='Frames after the first should already match: dm_cmd catches up once the loop is running'
        )


if __name__ == '__main__':
    unittest.main()
