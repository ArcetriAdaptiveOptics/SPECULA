#!/usr/bin/env python
"""List the files a set of SPECULA simulations reads, one absolute path per line.

Each positional argument is one simulation: its YAML files joined by commas,
in the order they would be given to ``specula`` (base file first). Example,
from config/, where the SOUL runs are launched::

    python ../specula/scripts/collect_calib_files.py -o calib_files.txt \\
        SOUL/params_pyr_soul.yml \\
        SOUL/params_pyr_soul.yml,SOUL/params_pyr_soul_cnn_train_data.yml \\
        SOUL/params_pyr_soul.yml,SOUL/params_pyr_soul_cnn_closed_loop.yml

With --all DIR, the simulations are found in DIR (recursively) instead:
every YAML file with a SimulParams object is a simulation by itself, and
every other YAML file (an override file) is combined with each of the
former in its own directory. The pairs SPECULA would reject (an override
or remove of an object the base file doesn't have) are skipped; override
files that pair with nothing, and simulations that can't be resolved, are
reported on stderr.

The YAML files are merged with Simul's own combine_params / apply_overrides,
and the paths are built the way Simul.build_objects and the classes build them:

- ``tag``:                     CalibManager.filename(<class name>, tag)
- ``<name>_object``:           CalibManager.filename(<name>, value)
- ``<name>_list/dict_object``: CalibManager.filename(<type hint's class name>, tag)
- ``<name>_data``:             CalibManager.filename('data', value)
- ``data_dir`` (injected as root_dir/<subdir of the class>, unless the YAML
  sets it): the files inside it are class specific, see CLASS_HANDLERS.
  A class that takes data_dir but has no handler is reported on stderr.
- class-specific file parameters (e.g. ``network_filename``): CLASS_HANDLERS.
  Relative paths are resolved against --run-dir, the directory specula is
  launched from (default: the current directory).

Only input files are listed; outputs (DataStore's store_dir, the trainer's
diagnostics) are not. Files that don't exist are reported on stderr and
left out of the list (they'd be generated, e.g. phase screens, or are
missing); --include-missing keeps them.
"""

import argparse
import inspect
import os
import sys

import numpy as np
import yaml

import specula
specula.init(-1, precision=1)   # CPU: classes are only imported, never built

from specula.calib_manager import CalibManager
from specula.lib.utils import import_class, get_type_hints, remove_suffix, resolve_type
from specula.simul import Simul


# ---------------------------------------------------------------------------
# Class-specific handlers: (key, pars, ctx) -> list of paths.
# ctx holds main (SimulParams), cm, data_dir, run_dir, precision.
# ---------------------------------------------------------------------------

def _atmo_evolution_files(key, pars, ctx):
    """Phase screens read by AtmoEvolution.compute() through phasescreens_manager."""
    main = ctx['main']
    pixel_pupil = main['pixel_pupil']
    pixel_pitch = main['pixel_pitch']
    zenith = main.get('zenithAngleInDeg', None)
    airmass = 1.0 / np.cos(np.radians(zenith)) if zenith is not None else 1.0

    heights = np.array(pars['heights'], dtype=float)
    L0 = pars['L0']
    fov = pars.get('fov', 0.0)
    fov_in_m = pars.get('fov_in_m', None)
    pupil_position = pars.get('pupil_position', [0, 0])
    dimension = pars.get('pixel_phasescreens', 8192)
    seed0 = pars.get('seed', 1)
    n_phasescreens = len(heights)

    # Same formulas as AtmoEvolution.__init__ / compute()
    fov_rad = fov * np.pi / 180 / 3600
    pixel_layer = np.ceil(
        (pixel_pupil
         + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=float) * 2)) / pixel_pitch
         + abs(heights * airmass) / pixel_pitch * fov_rad) / 2.0) * 2.0
    if fov_in_m is not None:
        pixel_layer = np.full_like(heights, int(fov_in_m / pixel_pitch / 2.0) * 2)
    pixel_phasescreens = int(np.max(pixel_layer))

    L0_list = list(L0) if hasattr(L0, '__len__') else [L0]
    if len(np.unique(L0_list)) == 1:
        n_ps = int(np.ceil(float(n_phasescreens) / np.floor(dimension / pixel_phasescreens)))
        seeds = np.arange(seed0, seed0 + n_ps)
        L0s = [L0_list[0]] * n_ps
    else:
        seeds = seed0 + np.arange(n_phasescreens)
        L0s = L0_list

    precision = pars.get('precision', ctx['precision'])
    precision_str = 'single' if precision == 1 else 'double'
    data_dir = ctx['data_dir']
    files = []
    for seed, L0i in zip(seeds, L0s):
        # The four names phasescreens_manager accepts, in its order
        names = [
            f'ps_seed{np.around(seed)}_dim{np.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0i):.4f}_{precision_str}.fits',
            f'ps_seed{np.around(seed)}_dim{np.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{np.around(L0i):.4f}_{precision_str}.fits',
            f'ps_seed{float(seed)}_dim{np.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{float(L0i):.4f}_{precision_str}.fits',
            f'ps_seed{float(seed)}_dim{np.around(dimension)}_pixpit{pixel_pitch:.3f}_L0{np.around(L0i):.4f}_{precision_str}.fits',
        ]
        paths = [os.path.join(data_dir, n) for n in names]
        found = [p for p in paths if os.path.exists(p)]
        files.append(found[0] if found else paths[0])
    return files


def _checkpoint_files(network_filename, ctx):
    """A Conv2dNet checkpoint and its stats file (specula/lib/cnn_checkpoint.py)."""
    path = os.path.join(ctx['run_dir'], network_filename)
    return [path, os.path.splitext(path)[0] + '_stats.json']


def _conv2d_net_reader_files(key, pars, ctx):
    return _checkpoint_files(pars['network_filename'], ctx)


def _conv2d_net_trainer_files(key, pars, ctx):
    if not pars.get('load_from_file', False):
        return []   # the checkpoint is only written
    network_filename = pars['network_filename']
    if not os.path.splitext(network_filename)[1]:
        network_filename += '.pth'
    return _checkpoint_files(network_filename, ctx)


CLASS_HANDLERS = {
    'AtmoEvolution': _atmo_evolution_files,
    'AtmoEvolutionUpDown': _atmo_evolution_files,
    'Conv2dNetRec': _conv2d_net_reader_files,
    'Conv2dNetTester': _conv2d_net_reader_files,
    'Conv2dNetTrainer': _conv2d_net_trainer_files,
}

# Classes that take data_dir but read nothing from it as inputs (they only
# write calibrations there), so they don't need a handler.
DATA_DIR_WRITERS = {
    'SnCalibrator', 'RecCalibrator', 'MultiRecCalibrator', 'ImCalibrator',
    'MultiImCalibrator', 'PyrPupdataCalibrator', 'DynamicPyrPupdataCalibrator',
    'DynamicDarkCalibrator', 'ShSubapCalibrator', 'PyramidNonlinearCalibrator',
}


# ---------------------------------------------------------------------------

_warned = set()

def warn(message):
    """Print a warning on stderr, once."""
    if message not in _warned:
        _warned.add(message)
        print(f'WARNING: {message}', file=sys.stderr)


def merged_params(yml_files, overrides):
    """The parameter dict Simul.run() builds before creating the objects."""
    simul = Simul(*yml_files, overrides=overrides, log_level='warning')
    with open(yml_files[0]) as f:
        params = yaml.safe_load(f)
    for filename in yml_files[1:]:
        with open(filename) as f:
            simul.combine_params(params, yaml.safe_load(f))
    simul.apply_overrides(params)
    return params


def simulation_files(params, run_dir, precision):
    """(path, origin) for every file the objects in params read."""
    main = next(p for p in params.values() if p.get('class') == 'SimulParams')
    cm = CalibManager(main['root_dir'])
    additional_modules = main.get('add_modules', [])
    result = []

    for key, pars in params.items():
        classname = pars['class']
        origin = f'{key} ({classname})'

        if 'tag' in pars:
            result.append((cm.filename(classname, pars['tag']), f'{origin}.tag'))
            continue

        klass = import_class(classname, additional_modules)
        args = inspect_args(klass)

        for name, value in pars.items():
            if value is None:
                continue
            if name.endswith('_list_object') or name.endswith('_dict_object'):
                parname = remove_suffix(name, '_object')
                partype = resolve_type(get_type_hints(klass)[parname],
                                       require_list=name.endswith('_list_object'),
                                       require_dict=name.endswith('_dict_object'))
                tags = value if isinstance(value, (list, tuple)) else value.values()
                for tag in tags:
                    result.append((cm.filename(partype.__name__, tag), f'{origin}.{name}'))
            elif name.endswith('_object'):
                parname = remove_suffix(name, '_object')
                if parname in args and name not in args:
                    result.append((cm.filename(parname, value), f'{origin}.{name}'))
            elif name.endswith('_data'):
                parname = remove_suffix(name, '_data')
                if parname in args and name not in args:
                    result.append((cm.filename('data', value), f'{origin}.{name}'))

        data_dir = None
        if 'data_dir' in args:
            data_dir = pars.get('data_dir', cm.root_subdir(classname))
        if classname in CLASS_HANDLERS:
            ctx = dict(main=main, cm=cm, data_dir=data_dir, run_dir=run_dir, precision=precision)
            for path in CLASS_HANDLERS[classname](key, pars, ctx):
                result.append((path, origin))
        elif data_dir is not None and classname not in DATA_DIR_WRITERS:
            warn(f'{classname} takes data_dir={data_dir}, but there is no handler '
                 f'to know which files it reads: add one to CLASS_HANDLERS')
    return result


def is_base_config(filename):
    with open(filename) as f:
        params = yaml.safe_load(f)
    return isinstance(params, dict) and any(
        isinstance(p, dict) and p.get('class') == 'SimulParams' for p in params.values())


def discover_simulations(directory):
    """The simulations in directory, as lists of YAML files (see --all)."""
    by_dir = {}
    for root, _, names in os.walk(directory):
        for name in sorted(names):
            if name.endswith(('.yml', '.yaml')):
                path = os.path.join(root, name)
                try:
                    kind = 'base' if is_base_config(path) else 'override'
                except yaml.YAMLError as e:
                    warn(f'skipping {path}: not valid YAML ({e})')
                    continue
                by_dir.setdefault(root, {'base': [], 'override': []})[kind].append(path)

    simulations = []
    for root, found in sorted(by_dir.items()):
        simulations += [[base] for base in found['base']]
        for override in found['override']:
            pairs = [[base, override] for base in found['base']]
            if not pairs:
                warn(f'{override} is an override file with no base config in its '
                     f'directory: pass it explicitly, as base.yml,{os.path.basename(override)}')
            simulations += [p + ['?'] for p in pairs]   # '?': may be rejected
    return simulations


def inspect_args(klass):
    return inspect.getfullargspec(klass.__init__).args


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('simulations', nargs='*',
                        help='one per simulation: its YAML files, comma separated')
    parser.add_argument('--all', metavar='DIR',
                        help='also every simulation found in DIR, see above')
    parser.add_argument('-o', '--output', default='-', help='output list (default: stdout)')
    parser.add_argument('--run-dir', default=os.getcwd(),
                        help='directory specula is launched from, for relative paths '
                             '(default: the current directory)')
    parser.add_argument('--overrides', default=None,
                        help='same as specula --overrides, applied to every simulation')
    parser.add_argument('--precision', type=int, default=1,
                        help='same as specula --precision (0 double, 1 single; default 1)')
    parser.add_argument('--include-missing', action='store_true',
                        help='also list the files that do not exist')
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    simulations = [[os.path.normpath(os.path.join(run_dir, f)) for f in sim.split(',')]
                   for sim in args.simulations]
    if args.all:
        simulations += discover_simulations(os.path.normpath(os.path.join(run_dir, args.all)))
    if not simulations:
        parser.error('no simulations: give some, or --all DIR')

    files = {}
    discovered_overrides, paired_overrides = set(), set()
    for yml_files in simulations:
        discovered = yml_files[-1] == '?'
        if discovered:
            yml_files = yml_files[:-1]
            discovered_overrides.add(yml_files[-1])
        label = ','.join(os.path.relpath(f, run_dir) for f in yml_files)
        try:
            params = merged_params(yml_files, args.overrides)
        except (ValueError, KeyError) as e:
            if discovered:
                continue   # an override file that doesn't belong to this base
            raise
        if discovered:
            paired_overrides.add(yml_files[-1])
        try:
            sim_files = simulation_files(params, run_dir, args.precision)
        except Exception as e:
            if not args.all:
                raise
            warn(f'skipping {label}: {type(e).__name__}: {e}')
            continue
        for path, origin in sim_files:
            files.setdefault(os.path.normpath(os.path.abspath(path)), set()).add(
                f'{label}: {origin}')

    for override in sorted(discovered_overrides - paired_overrides):
        warn(f'{override} could not be combined with any base config in its '
             f'directory: pass it explicitly, as base.yml,{os.path.basename(override)}')

    missing = sorted(p for p in files if not os.path.exists(p))
    for path in missing:
        print(f'MISSING: {path}  <- {"; ".join(sorted(files[path]))}', file=sys.stderr)

    listed = sorted(p for p in files if args.include_missing or p not in missing)
    text = ''.join(p + '\n' for p in listed)
    if args.output == '-':
        sys.stdout.write(text)
    else:
        with open(args.output, 'w') as f:
            f.write(text)
        print(f'{len(listed)} files written to {args.output} '
              f'({len(missing)} missing)', file=sys.stderr)


if __name__ == '__main__':
    main()
