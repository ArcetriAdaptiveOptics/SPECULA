#!/usr/bin/env python
"""List the SPECULA runs (TN folders) in a folder and show what changes between them.

Each TN folder written by DataStore contains a ``params.yml`` with the full
simulation parameters (overrides included). This tool reads the ``params.yml``
of each TN folder directly inside ``root_dir`` (not recursively) and:

- prints, for each TN in chronological order (``TN.0`` right after ``TN``),
  the parameters that changed with respect to the previous TN. Runs launched
  in a batch show up as consecutive TNs changing the same key(s). Keys that
  appear/disappear are shown as ``(absent)``;
- writes a CSV (default ``root_dir/list_runs.csv``) with one row per TN and
  one column per parameter taking at least two different values among the
  runs where it is present.

Values longer than 1000 characters are always shortened (50 with ``--short``),
and the shortened keys are listed at the end.
"""

import os
import re
import csv
import argparse
import fnmatch
from collections.abc import Callable, Sequence
from typing import Any

import yaml

# Keys that change in every run without meaning anything (paths).
DEFAULT_IGNORE = ['main.root_dir', 'data_store.store_dir']
# Hard limit on the length of a printed/saved value (no option to raise it).
MAX_VALUE_LEN = 1000
SHORT_VALUE_LEN = 50
MISSING = '(absent)'
CSV_NAME = 'list_runs.csv'
# Standard TN folder name written by DataStore: YYYYMMDD_HHMMSS[.N]
TN_PATTERN = re.compile(r'\d{8}_\d{6}(\.\d+)?')


def flatten_dict(d: dict, prefix: str = '') -> dict:
    """Nested dict -> {"a.b.c": value}. Lists are kept as a single value."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}.{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def numbers(v) -> list:
    """All numbers in a (nested) list."""
    if isinstance(v, list):
        return [x for sub in v for x in numbers(sub)]
    return [v] if isinstance(v, (int, float)) and not isinstance(v, bool) else []


def format_value(v, max_len: int) -> tuple[str, bool]:
    """Value as string, and whether it was shortened.

    If longer than ``max_len``, numeric lists become ``'[n] min..max'``,
    anything else is truncated.
    """
    s = str(v)
    if len(s) <= max_len:
        return s, False
    nums = numbers(v)
    if nums:
        return f'[{len(v)}] {min(nums):g}..{max(nums):g}', True
    return s[:max_len - 3] + '...', True


def chrono_key(tn: str) -> tuple[str, int]:
    """'20260805_080019.0' -> ('20260805_080019', 0).

    Sorts TN.0 right after TN, and TN.10 after TN.9 (which plain string
    sorting gets wrong). DataStore adds the .N suffix to TNs created in the
    same second.
    """
    base, _, it = tn.partition('.')
    return (base, int(it) if it.isdigit() else -1)


def matches(key: str, patterns: Sequence[str]) -> bool:
    """True if ``key`` matches any of the wildcard ``patterns``."""
    return any(fnmatch.fnmatchcase(key, p) for p in patterns)


def collect_runs(root_dir: str,
                 keys: Sequence[str] | None = None,
                 ignore: Sequence[str] = (),
                 since: str | None = None) -> tuple[dict[str, dict], list[str], list[str]]:
    """Read the params.yml of each TN folder directly inside ``root_dir``.

    Parameters
    ----------
    root_dir : str
        Folder containing the TN folders.
    keys : sequence of str, optional
        If given, keep only the keys matching these wildcard patterns.
    ignore : sequence of str
        Wildcard patterns of keys to drop (in addition to DEFAULT_IGNORE).
    since : str, optional
        'YYYYMMDD': skip TNs whose name starts with an earlier date. TNs
        whose name is not a standard TN name cannot be dated: they are kept,
        with a warning.

    Returns
    -------
    runs : dict
        {tn: {flattened key: value}}, TNs in chronological order.
    other_dirs : list of str
        Sorted sub-folders without params.yml (not read). ``*_PSF`` folders
        are skipped silently.
    warnings : list of str
        Messages about folders skipped because their params.yml cannot be
        read, and TNs with a non-standard name kept despite ``since``.
    """
    ignore = list(DEFAULT_IGNORE) + list(ignore)
    runs = {}
    other_dirs = []
    warnings = []
    undated = []
    for name in sorted(os.listdir(root_dir), key=chrono_key):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path) or name.endswith('_PSF'):
            continue
        yml_file = os.path.join(path, 'params.yml')
        if not os.path.isfile(yml_file):
            other_dirs.append(name)
            continue
        if since is not None:
            if not TN_PATTERN.fullmatch(name):
                undated.append(name)
            elif name[:8] < since:
                continue
        # An interrupted run can leave an empty or truncated params.yml:
        # skip that folder instead of aborting the whole listing.
        try:
            with open(yml_file, 'r', encoding='utf-8') as f:
                params = yaml.safe_load(f)
        except yaml.YAMLError as e:
            warnings.append(f'{name}: params.yml not readable ({e.__class__.__name__}), skipped')
            continue
        if not isinstance(params, dict):
            warnings.append(f'{name}: params.yml empty or not a dictionary, skipped')
            continue
        runs[name] = {k: v for k, v in flatten_dict(params).items()
                      if (keys is None or matches(k, keys)) and not matches(k, ignore)}
    if undated:
        warnings.append(f'non-standard TN names, not filtered by --since: {undated}')
    return runs, sorted(other_dirs), warnings


def changed_keys(prev: dict, cur: dict) -> list[str]:
    """Sorted keys whose value differs between two runs (absent counts as a value)."""
    return [k for k in sorted(set(cur) | set(prev))
            if str(cur.get(k, MISSING)) != str(prev.get(k, MISSING))]


def select_columns(runs: dict[str, dict], all_keys: bool = False,
                   selected: bool = False) -> list[str]:
    """CSV columns.

    Default: keys taking at least two different values among the runs where
    they are present (keys that only appear/disappear are structural changes
    and are skipped). ``all_keys``: also those. ``selected`` (keys chosen with
    --keys): all keys, even if constant, so that a constant value is not
    mistaken for a wrong pattern.
    """
    keys = sorted(set().union(*runs.values()))

    def n_values(k):
        return len({str(r.get(k, MISSING)) for r in runs.values() if all_keys or k in r})

    return [k for k in keys if selected or n_values(k) > 1]


def make_formatter(max_len: int) -> tuple[Callable[[str, Any], str], set[str]]:
    """Formatter fmt(key, value) -> str, and the set where it records the keys
    whose values were shortened (shared by terminal and CSV output)."""
    shortened = set()

    def fmt(key, v):
        s, cut = format_value(v, max_len)
        if cut:
            shortened.add(key)
        return s

    return fmt, shortened


def write_csv(csv_file: str, runs: dict[str, dict], columns: Sequence[str],
              fmt: Callable[[str, Any], str]):
    """Write one row per TN, values formatted with fmt."""
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tn', *columns])
        writer.writerows([tn, *(fmt(k, p.get(k, MISSING)) for k in columns)]
                         for tn, p in runs.items())


def since_date(value: str) -> str:
    """argparse type for --since: exactly 8 digits (YYYYMMDD)."""
    if not re.fullmatch(r'\d{8}', value):
        raise argparse.ArgumentTypeError(f"'{value}' is not a date in YYYYMMDD format")
    return value


def main(argv: Sequence[str] | None = None):
    '''
    Entry point for the specula_list_runs command.
    '''
    parser = argparse.ArgumentParser(
        description='List the TNs in a folder and the parameters that change between them.')
    parser.add_argument('root_dir', nargs='?', default='.',
                        help='Folder containing the TN folders (default: current folder).')
    parser.add_argument('--out', default=None,
                        help=f'CSV file to write (default: root_dir/{CSV_NAME}).')
    parser.add_argument('--all-keys', action='store_true',
                        help='CSV: also include keys present only in some runs.')
    parser.add_argument('--short', action='store_true',
                        help=f'Shorten values longer than {SHORT_VALUE_LEN} characters '
                             f'(default: {MAX_VALUE_LEN}).')
    parser.add_argument('--keys', nargs='+', default=None, metavar='PATTERN',
                        help="Only these keys, wildcards allowed, e.g. '*magnitude*' "
                             "seeing.constant. Their CSV columns are shown even if "
                             "they never change.")
    parser.add_argument('--ignore', nargs='+', default=[], metavar='PATTERN',
                        help="Keys to ignore, wildcards allowed, e.g. '*seed*' '*_disp.*'. "
                             f"Always ignored: {', '.join(DEFAULT_IGNORE)}.")
    parser.add_argument('--since', default=None, metavar='YYYYMMDD', type=since_date,
                        help='Only TNs from this date on.')
    args = parser.parse_args(argv)

    csv_file = args.out if args.out is not None else os.path.join(args.root_dir, CSV_NAME)
    max_len = SHORT_VALUE_LEN if args.short else MAX_VALUE_LEN
    fmt, shortened = make_formatter(max_len)

    runs, other_dirs, warnings = collect_runs(args.root_dir, keys=args.keys,
                                              ignore=args.ignore, since=args.since)
    for w in warnings:
        print(f'WARNING: {w}')
    if warnings:
        print()
    if other_dirs:
        print('Sub-folders without params.yml (not scanned):')
        for name in other_dirs:
            print(f'    {name}/')
        print()
    if not runs:
        raise SystemExit(f'No TN folder with params.yml in {args.root_dir}')
    print(f'{len(runs)} runs found in {args.root_dir}\n')

    if args.keys is not None:
        found = set().union(*runs.values())
        unmatched = [p for p in args.keys if not any(matches(k, [p]) for k in found)]
        if unmatched:
            print(f'WARNING: no key matches {unmatched}\n')

    # Terminal: changes with respect to the previous TN
    prev = None
    for tn, cur in runs.items():
        if prev is None:
            print(f'{tn}  (first run)')
        else:
            changed = changed_keys(prev, cur)
            if changed:
                print(tn)
                for k in changed:
                    old, new = fmt(k, prev.get(k, MISSING)), fmt(k, cur.get(k, MISSING))
                    print(f'    {k}: {old} -> {new}')
            else:
                print(f'{tn}  (same params as previous)')
        prev = cur

    columns = select_columns(runs, all_keys=args.all_keys, selected=args.keys is not None)
    try:
        write_csv(csv_file, runs, columns, fmt)
    except OSError as e:
        raise SystemExit(f'Cannot write {csv_file} ({e}). Use --out to write it elsewhere.')
    print(f'\n{len(columns)} parameters saved to {csv_file}')

    if shortened:
        print(f'\nWARNING: values longer than {max_len} characters were shortened for:')
        for k in sorted(shortened):
            print(f'    {k}')


if __name__ == '__main__':
    main()
