import os
import csv
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

import yaml

from specula.scripts import list_runs
from specula.scripts.list_runs import (
    chrono_key, collect_runs, changed_keys, flatten_dict, format_value, select_columns)


class TestListRuns(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='test_list_runs_')

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_tn(self, name, params=None, parent=None):
        """Create a TN folder with params.yml (default: minimal params)."""
        path = os.path.join(parent or self.root, name)
        os.makedirs(path)
        with open(os.path.join(path, 'params.yml'), 'w') as f:
            yaml.dump(params if params is not None else {'a': 1}, f)
        return path

    def _run_main(self, *argv):
        """Run main(argv), return captured stdout."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            list_runs.main([str(a) for a in argv])
        return buf.getvalue()

    def _read_csv(self, path):
        with open(path, newline='') as f:
            return list(csv.reader(f))

    # ---- collect_runs ----

    def test_chronological_order(self):
        """TN.0 follows TN; TN.9 precedes TN.10 (numeric, not string, order)."""
        names = ['20260101_120000.10', '20260101_120000', '20260101_120000.9',
                 '20260101_120000.0', '20251231_235959']
        for n in names:
            self._make_tn(n)
        runs, other, _ = collect_runs(self.root)
        self.assertEqual(list(runs), ['20251231_235959', '20260101_120000',
                                      '20260101_120000.0', '20260101_120000.9',
                                      '20260101_120000.10'])
        self.assertEqual(other, [])

    def test_chrono_key(self):
        self.assertEqual(chrono_key('20260805_080019.0'), ('20260805_080019', 0))
        self.assertEqual(chrono_key('20260805_080019'), ('20260805_080019', -1))

    def test_non_tn_dirs_not_recursive(self):
        """A sub-folder without params.yml is listed in other_dirs, its nested TN is not read."""
        self._make_tn('20260101_120000')
        scan = os.path.join(self.root, 'rtf_scan')
        os.mkdir(scan)
        self._make_tn('20260102_000000', parent=scan)
        runs, other, _ = collect_runs(self.root)
        self.assertEqual(list(runs), ['20260101_120000'])
        self.assertEqual(other, ['rtf_scan'])

    def test_psf_dir_and_plain_file_ignored(self):
        """'*_PSF' folders and plain files are neither runs nor other_dirs."""
        self._make_tn('20260101_120000')
        os.mkdir(os.path.join(self.root, '20260101_120000_PSF'))
        with open(os.path.join(self.root, 'notes.txt'), 'w') as f:
            f.write('x')
        runs, other, _ = collect_runs(self.root)
        self.assertEqual(list(runs), ['20260101_120000'])
        self.assertEqual(other, [])

    def test_since_drops_earlier_tns(self):
        self._make_tn('20251231_235959')
        self._make_tn('20260101_000000')
        self._make_tn('20260102_000000')
        runs, _, _ = collect_runs(self.root, since='20260101')
        self.assertEqual(list(runs), ['20260101_000000', '20260102_000000'])

    def test_since_keeps_non_standard_names_with_warning(self):
        # 'my_run' cannot be dated: kept, and reported in a warning
        self._make_tn('20251231_235959')
        self._make_tn('my_run')
        runs, _, warnings = collect_runs(self.root, since='20260101')
        self.assertEqual(list(runs), ['my_run'])
        self.assertEqual(warnings, ["non-standard TN names, not filtered by --since: ['my_run']"])

    def test_bad_params_yml_skipped_with_warning(self):
        # empty and unparsable params.yml: folder skipped, the others still read
        self._make_tn('20260101_120000')
        for name, content in [('20260102_120000', ''), ('20260103_120000', 'a: [1, 2\n')]:
            os.makedirs(os.path.join(self.root, name))
            with open(os.path.join(self.root, name, 'params.yml'), 'w') as f:
                f.write(content)
        runs, _, warnings = collect_runs(self.root)
        self.assertEqual(list(runs), ['20260101_120000'])
        self.assertEqual(len(warnings), 2)
        self.assertTrue(warnings[0].startswith('20260102_120000: params.yml empty'))
        self.assertTrue(warnings[1].startswith('20260103_120000: params.yml not readable'))

    def test_main_prints_collect_warnings(self):
        self._make_tn('20260101_120000')
        os.makedirs(os.path.join(self.root, '20260102_120000'))
        open(os.path.join(self.root, '20260102_120000', 'params.yml'), 'w').close()
        out = self._run_main(self.root)
        self.assertIn('WARNING: 20260102_120000: params.yml empty or not a dictionary, skipped',
                      out)

    def test_main_since_bad_format_exits(self):
        # argparse rejects a --since value that is not YYYYMMDD
        self._make_tn('20260101_120000')
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._run_main(self.root, '--since', '2026-01-01')

    def test_default_ignore_keys_dropped(self):
        self._make_tn('20260101_120000', {'main': {'root_dir': '/x', 'total_time': 1.0},
                                          'data_store': {'store_dir': '/y', 'inputs': 1}})
        runs, _, _ = collect_runs(self.root)
        self.assertEqual(runs['20260101_120000'],
                         {'main.total_time': 1.0, 'data_store.inputs': 1})

    def test_ignore_wildcards(self):
        params = {'atmo': {'seed': 1, 'r0': 0.1},
                  'detector': {'photon_seed': 2, 'size': 8},
                  'dm_disp': {'class': 'D', 'title': 't'},
                  'wfs_disp': {'class': 'D'}}
        self._make_tn('20260101_120000', params)
        runs, _, _ = collect_runs(self.root, ignore=['*seed*', '*_disp.*'])
        self.assertEqual(runs['20260101_120000'], {'atmo.r0': 0.1, 'detector.size': 8})

    def test_keys_selection(self):
        params = {'wfs': {'magnitude': 8, 'size': 4}, 'seeing': {'constant': 0.6, 'x': 1},
                  'other': 3}
        self._make_tn('20260101_120000', params)
        runs, _, _ = collect_runs(self.root, keys=['*magnitude*', 'seeing.constant'])
        self.assertEqual(runs['20260101_120000'],
                         {'wfs.magnitude': 8, 'seeing.constant': 0.6})

    # ---- flatten_dict ----

    def test_flatten_nested_and_lists(self):
        d = {'a': {'b': {'c': 1}, 'd': [1, 2, 3]}, 'e': 'x'}
        self.assertEqual(flatten_dict(d), {'a.b.c': 1, 'a.d': [1, 2, 3], 'e': 'x'})

    # ---- changed_keys / select_columns ----

    def test_changed_keys_absent_vs_present(self):
        prev = {'a': 1, 'b': 2}
        cur = {'a': 1, 'c': 3}
        self.assertEqual(changed_keys(prev, cur), ['b', 'c'])

    def test_changed_keys_identical(self):
        self.assertEqual(changed_keys({'a': 1, 'b': [1, 2]}, {'a': 1, 'b': [1, 2]}), [])

    def _columns_runs(self):
        return {
            't1': {'seeing.constant': 0.6, 'wfs.magnitude': 8, 'dm_disp.class': 'A', 'gain': 0.3},
            't2': {'seeing.constant': 1.0, 'wfs.magnitude': 8, 'dm_disp.class': 'A', 'gain': 0.5},
            't3': {'seeing.constant': 0.6, 'wfs.magnitude': 8},
            't4': {'seeing.constant': 1.0, 'wfs.magnitude': 8},
        }

    def test_select_columns_default(self):
        """Only keys with >= 2 distinct values among runs where present."""
        self.assertEqual(select_columns(self._columns_runs()), ['gain', 'seeing.constant'])

    def test_select_columns_all_keys(self):
        """Also keys that only appear/disappear, but not constant ones."""
        self.assertEqual(select_columns(self._columns_runs(), all_keys=True),
                         ['dm_disp.class', 'gain', 'seeing.constant'])

    def test_select_columns_selected(self):
        """Selected: every key, including constants."""
        self.assertEqual(select_columns(self._columns_runs(), selected=True),
                         ['dm_disp.class', 'gain', 'seeing.constant', 'wfs.magnitude'])

    # ---- format_value ----

    def test_format_value_short_unchanged(self):
        self.assertEqual(format_value('abc', 3), ('abc', False))
        self.assertEqual(format_value(0.6, 1000), ('0.6', False))

    def test_format_value_long_numeric_list(self):
        self.assertEqual(format_value(list(range(300)), 1000), ('[300] 0..299', True))

    def test_format_value_nested_numeric_list(self):
        """n is the outer length; min/max span all nested numbers."""
        v = [[5, 7, -2.5] for _ in range(2)] + [[100, 3]]
        self.assertEqual(format_value(v, 10), ('[3] -2.5..100', True))

    def test_format_value_long_string_truncated(self):
        s, cut = format_value('x' * 100, 50)
        self.assertTrue(cut)
        self.assertEqual(s, 'x' * 47 + '...')
        self.assertEqual(len(s), 50)

    def test_format_value_bool_list_not_numeric(self):
        """A long list of booleans is truncated as text, not summarised as numbers."""
        v = [True] * 20
        s, cut = format_value(v, 30)
        self.assertTrue(cut)
        self.assertEqual(s, str(v)[:27] + '...')

    # ---- main: CSV ----

    def _make_two_runs(self, extra1=None, extra2=None):
        p1 = {'seeing': {'constant': 0.6}, 'wfs': {'magnitude': 8}}
        p2 = {'seeing': {'constant': 1.0}, 'wfs': {'magnitude': 8}}
        p1.update(extra1 or {})
        p2.update(extra2 or {})
        self._make_tn('20260101_120000.0', p2)
        self._make_tn('20260101_120000', p1)

    def test_main_default_csv(self):
        """CSV in root_dir, header 'tn'+columns, one row per TN in chronological order."""
        self._make_two_runs(extra1={'gain': 0.3})
        self._run_main(self.root)
        rows = self._read_csv(os.path.join(self.root, 'list_runs.csv'))
        self.assertEqual(rows, [['tn', 'seeing.constant'],
                                ['20260101_120000', '0.6'],
                                ['20260101_120000.0', '1.0']])

    def test_main_absent_value(self):
        """Value absent in a run is written as '(absent)'."""
        self._make_two_runs(extra1={'gain': 0.3}, extra2={})
        self._make_tn('20260101_120001', {'seeing': {'constant': 1.0}, 'gain': 0.5,
                                          'wfs': {'magnitude': 8}})
        self._run_main(self.root, '--all-keys')
        rows = self._read_csv(os.path.join(self.root, 'list_runs.csv'))
        self.assertEqual(rows[0], ['tn', 'gain', 'seeing.constant'])
        self.assertEqual(rows[1:], [['20260101_120000', '0.3', '0.6'],
                                    ['20260101_120000.0', '(absent)', '1.0'],
                                    ['20260101_120001', '0.5', '1.0']])

    def test_main_out_option(self):
        out_dir = tempfile.mkdtemp(prefix='test_list_runs_out_')
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        out = os.path.join(out_dir, 'my.csv')
        self._make_two_runs()
        self._run_main(self.root, '--out', out)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.exists(os.path.join(self.root, 'list_runs.csv')))

    def test_main_short_truncates_and_warns(self):
        self._make_tn('20260101_120000', {'v': 'a' * 100})
        self._make_tn('20260101_120001', {'v': 'b' * 100})
        stdout = self._run_main(self.root, '--short')
        rows = self._read_csv(os.path.join(self.root, 'list_runs.csv'))
        self.assertEqual(rows[1][1], 'a' * 47 + '...')
        self.assertEqual(len(rows[1][1]), 50)
        self.assertIn('WARNING: values longer than 50 characters were shortened for:\n    v',
                      stdout)

    def test_main_default_max_no_truncation(self):
        self._make_tn('20260101_120000', {'v': 'a' * 100})
        self._make_tn('20260101_120001', {'v': 'b' * 100})
        stdout = self._run_main(self.root)
        rows = self._read_csv(os.path.join(self.root, 'list_runs.csv'))
        self.assertEqual(rows[1][1], 'a' * 100)
        self.assertNotIn('WARNING', stdout)

    def test_main_keys_no_match_warning(self):
        self._make_two_runs()
        stdout = self._run_main(self.root, '--keys', 'foo*', 'seeing.constant')
        self.assertIn("WARNING: no key matches ['foo*']", stdout)

    def test_main_keys_constant_column_kept(self):
        """With --keys, constant columns are written to the CSV."""
        self._make_two_runs()
        self._run_main(self.root, '--keys', 'wfs.magnitude')
        rows = self._read_csv(os.path.join(self.root, 'list_runs.csv'))
        self.assertEqual(rows, [['tn', 'wfs.magnitude'],
                                ['20260101_120000', '8'],
                                ['20260101_120000.0', '8']])

    def test_main_empty_root_exits(self):
        with self.assertRaises(SystemExit):
            self._run_main(self.root)

    def test_main_bad_out_dir_exits(self):
        self._make_two_runs()
        out = os.path.join(self.root, 'does_not_exist', 'x.csv')
        with self.assertRaises(SystemExit) as cm:
            self._run_main(self.root, '--out', out)
        self.assertIn('Use --out', str(cm.exception))

    # ---- main: terminal output ----

    def test_main_prints_change(self):
        self._make_two_runs()
        stdout = self._run_main(self.root)
        self.assertIn('20260101_120000  (first run)', stdout)
        self.assertIn('    seeing.constant: 0.6 -> 1.0', stdout.splitlines())

    def test_main_prints_same_params(self):
        self._make_tn('20260101_120000', {'a': 1})
        self._make_tn('20260101_120001', {'a': 1})
        stdout = self._run_main(self.root)
        self.assertIn('20260101_120001  (same params as previous)', stdout)


if __name__ == '__main__':
    unittest.main()
