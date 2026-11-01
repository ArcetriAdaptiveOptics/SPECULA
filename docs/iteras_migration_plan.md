# Plan: extracting the SPECULA simulation engine into **iteras**

Status (2026-10-30):
- **Phase 0:** merged upstream (#828).
- **Phase 1b:** merged upstream (#843).
- **Phase 2:** in review, PR from branch `iteras-phase2`.
- **Phase 1** (separate repository): deferred; iteras stays inside the SPECULA repository for now.
- **Phase 3:** next.

`iteras` is a new package, published later as the GitHub repository
`ArcetriAdaptiveOptics/ITERAS`. It contains the general-purpose end-to-end simulation
framework that was embedded in SPECULA, plus the mathematical and control building
blocks. SPECULA becomes an adaptive-optics plugin on top of iteras.

The change must be transparent to SPECULA users (YAML files, CLI, stored data), and
almost transparent to SPECULA developers who only write data and processing objects.
The final SPECULA PR will add the dependency on ITERAS 1.0.

## 1. Coupling points found in the original code

| # | Problem | Where | Solved in |
|---|---|---|---|
| 1 | Runtime configuration (`xp`, `cp`, precision, device, MPI comm/rank) copied at import time with `from specula import ...` | `base_time_obj.py`, `connections.py`, `loop_control.py`, `base_processing_obj.py`, many AO modules | Phase 0 (live reads, lazy `fuse`) |
| 2 | Class discovery hard-coded to `specula.processing_objects`, `specula.data_objects`, `specula.display` | `lib/utils.py::import_class` | Phase 0 (registry) |
| 3 | `SimulParams` mixes generic and AO fields; `Simul` finds it by class name | `data_objects/simul_params.py`, `simul.py` | Phase 0 (`SimParams`) |
| 4 | `CalibManager` subdirectory map hard-coded, mostly AO entries | `calib_manager.py` | Phase 0 (registry) |
| 5 | Replay sanity check hard-codes `ElectricField`/`Layer` | `simul.py` | Phase 0 (`track_consumers_in_replay()` hook) |
| 6 | Base processing object imports `Layer` for debug logging | `base_processing_obj.py` | Phase 0 (`debug_summary()`) |
| 7 | `Simul` special-cases `DataStore`/`DataBuffer`/`DataSource` by name | `simul.py` | Phase 0 (hooks, see the Extension API page) |
| 8 | `BaseTimeObj` eagerly imports cupyx/scipy ndimage, fft, linalg helpers | `base_time_obj.py` | Phase 1b |
| 9 | Generic objects typed on AO data objects (`MVM`/`SoftLimiter` on `Recmat`, `LinearCombination` on `DM`, `PushPullGenerator` on Zernike) | processing_objects | Phase 2 for `MVM`/`SoftLimiter` (`Matrix`); the others stay in SPECULA |
| 10 | Tests patch module-level names of core modules | `test_base_time_obj.py`, `test_connections.py`, `test_field_analyser.py`, ... | Phase 0 |

## 2. Layout (current)

Class discovery keeps the convention that class `FooBar` lives in module
`<package>.foo_bar`.

```
iteras/
  __init__.py          init(), to_xp, cpuArray, fuse, dtype lists, main_simul
  base_time_obj.py  base_data_obj.py  base_processing_obj.py  base_value.py
  scalar_values.py  connections.py  loop_control.py  simul.py  simul_diagram.py
  calib_manager.py  log.py  tracing.py  registry.py  base_replay_analyser.py
  template_processing_obj.py
  data_objects/        sim_params, iir_filter_data, ssr_filter_data, time_history, matrix
  processing_objects/  data_store, data_source, data_buffer, data_print, display_server,
                       interactive_input, terminal_input, filters and integrators,
                       generators, base_operation/slicer/inserter, mvm, soft_limiter,
                       round_to_multiple, demodulator, windowed_integration
  display/             base_display, display_process, data_plotter, plot_display,
                       plot_vector_display, vector_values_display, display_recorder
  lib/                 utils, storage, fits_io, display_server_api, fast_pinv,
                       demodulate_signal, rebin, process_utils, interp2d
  scripts/             iteras_main (CLI), web_frontend, templates
  examples/            classical control problems (Phase 3)

specula/               AO plugin
  __init__.py          forwards runtime values to iteras (PEP 562), registers the plugin
  <old paths>          alias modules: sys.modules entry pointing to the iteras module
  data_objects/, processing_objects/, display/, lib/   AO code
  data_objects/simul_params.py   SimulParams(SimParams) + AO fields
  data_objects/recmat.py         Recmat(Matrix)
  display/modes_display.py       ModesDisplay(VectorValuesDisplay)
  calib_subdirs.py               AO calibration subdirectories (and Matrix -> rec/)
```

## 3. Design decisions

- **Old module paths are aliases of the iteras modules** (`sys.modules[__name__] = iteras.x`).
  - `specula.base_time_obj is iteras.base_time_obj`, so the classes are the same objects.
  - `isinstance` and type checks work.
  - `mock.patch('specula.connections.process_comm')` patches the real module.
  - Old pickles load.
- **Runtime values** are read at use time from iteras.
  - `specula/__init__.py` forwards `cp`, `xp`, `global_precision`, `process_comm`, ... through
    a module `__getattr__` (PEP 562).
  - Objects can be created before `init()` is called.
- **Plugin registry** (`iteras.registry`):
  - packages searched by `import_class`: iteras first, then plugins, then `add_modules`.
    Qualified names (`specula.Integrator`) select a plugin, and `AmbiguousClassError` reports
    same-named different classes;
  - plugin roots, used for logger names;
  - calibration subdirectories;
  - environment variable aliases (`SPECULA_DISABLE_GPU`, `SPECULA_PORT`, `SPECULA_DATA_FORMAT`).

  SPECULA registers itself when imported, and declares an `iteras.plugins` entry point.
- **Hooks instead of class names** (documented on the Extension API page):
  - `accepts_input_list()`, `input_list_created()`;
  - `set_simulation_params()`, `set_replay_params()`;
  - `track_consumers_in_replay()`, `can_skip_updates()`.

  The allocator work adds `runs_on_cpu_only()`, `devices_used()` and related hooks.
- **Main parameters**: iteras `SimParams` holds the generic fields; SPECULA `SimulParams`
  subclasses it.
  - `Simul` identifies the main section with `issubclass(klass, SimParams)`.
  - Parameters not given take the defaults of the class (Phase 2).
- **General names, SPECULA synonyms**: objects that model general concepts but have
  AO-specific names get a general name in iteras, and SPECULA keeps the old name.
  - Plain alias for processing objects: `SpeculaInput = InteractiveInput`.
  - Thin subclass for data objects and displays, because the class name is used for
    calibration subdirectories, `OBJ_TYPE` headers or different defaults: `Recmat(Matrix)`,
    `ModesDisplay(VectorValuesDisplay)`.
  - When a parameter name was AO-specific, the new generic name is added and the old one
    is still accepted: `MVM(matrix=..., recmat=...)`, `SoftLimiter(matrix_list=..., recmat_list=...)`.
- **Storage** (`iteras.lib.storage`): `write_object()` / `read_object()`, with FITS, npz,
  npy and HDF5 back ends.
  - FITS keeps the layout of each class, and FITS and npy arrays are streamed to the GPU
    in chunks.
  - The default format comes from `ITERAS_DATA_FORMAT` (alias `SPECULA_DATA_FORMAT`), and is
    `fits` if it is not set, for compatibility with older installations.
  - Non-FITS headers are read as a dictionary (`storage.Header`), without the FITS limits.
  - h5py is a requirement.
- Borderline objects (`GainOptimizer`, `CloseGainOptimizer`, `AVC`, `LinearCombination`,
  `PushPullGenerator`) stay in SPECULA for now.
- Old import paths are kept indefinitely, with no deprecation warnings (open decision, see section 6).
- Each phase is a series of commits in Fabio's name only. Pure renames go in their own
  commit, so that GitHub shows moves. Fabio pushes.

## 4. Phases

**Phase 0: boundary inside the SPECULA repository.** Done, #828.

**Phase 1b: core improvements.** Done, #843.

**Phase 2: mathematical and control code.** In review, branch `iteras-phase2`; see section 5.

**Phase 3: new building blocks and examples.** Next.
- New objects:
  - `PID`;
  - `StateSpacePlant` (continuous to discrete);
  - `ODEPlant` (RK4 base);
  - `Saturation`.
- Examples, each a YAML file plus a plotting script, smoke-tested in CI:
  - mass-spring-damper with PID;
  - DC motor with saturation and anti-windup;
  - inverted pendulum with LQR;
  - Kalman tracking;
  - vibration rejection vs. theoretical RTF/NTF;
  - delay vs. stability margin;
  - 10^5 parallel loops on GPU with CUDA graphs;
  - plant and controller on two MPI ranks;
  - record and replay.
- Possibly: an `examples/` directory with package data, and a docs page "iteras without SPECULA".

**Phase 1: separate repository.** Deferred.
1. `git filter-repo` the `iteras/` tree (with history) into `ArcetriAdaptiveOptics/ITERAS`.
2. Publish, then make SPECULA depend on ITERAS (versions in lockstep at first). The final SPECULA PR adds
   the dependency on ITERAS 1.0.
3. Move the core tests to ITERAS, rewriting their AO fixtures with generic objects.
4. CI matrix for ITERAS (OS x MPI x CPU/GPU), with SPECULA's CI running against ITERAS main.

Alternative suggested by a reviewer: publish iteras as its own distribution from the
SPECULA repository (see section 6).

**Phase 4: documentation and release.**
- Framework docs move to iteras; SPECULA docs link them via intersphinx.
- A "Writing a domain plugin" guide, using SPECULA as the reference.
- ITERAS 1.0 release.

## 5. Outcomes

### Phase 0 (#828)
- Framework modules moved with `git mv`, leaving alias modules.
- Registry, `SimParams` split, CalibManager registration, hooks, `debug_summary()`.
- `iteras` command line and `iteras.plugins` entry point.
- Live runtime values and lazy `fuse`.
- Boundary test (iteras never imports specula) and compatibility tests.

Notes for developers:
- reinstall (`pip install -e .`) to get the `iteras` command and entry point;
- tests that patch framework globals patch them in iteras;
- framework loggers are `iteras.*`.

### Phase 1b (#843)
- Lazy scipy/cupyx helpers in `BaseTimeObj`.
- Storage API for all data objects and DataStore/DataSource:
  - npz, npy and HDF5, besides FITS;
  - `SPECULA_DATA_FORMAT` alias;
  - `fits_io` moved to iteras, with a reusable chunked GPU upload.
- h5py as a requirement.
- Each test runs with the `init()` settings of its own file.
- A check mode verifies that every object saved by the tests restores identically from
  all formats (`ITERAS_TEST_CHECK_OBJECT_FORMATS=1`).
- Removed duplicated GPU memory tracking code.
- After review:
  - Header class for non-FITS headers;
  - complex arrays in `load()`;
  - `Intmat.restore()` argument fix;
  - coverage tests (local patch coverage 99.6%).

### Phase 2 (branch `iteras-phase2`)
- **Moved to iteras:**
  - data objects: `IirFilterData`, `SsrFilterData`, `TimeHistory`;
  - the filters and integrators;
  - the generators;
  - `BaseOperation`, `BaseSlicer`, `BaseInserter`;
  - `RoundToMultiple`, `Demodulator`, `WindowedIntegration`;
  - lib: `fast_pinv`, `demodulate_signal`, `rebin`, `process_utils`, `interp2d`, `psd_to_signal`.
- **Generic `Matrix`**, with `Recmat` as a subclass. The matrix is saved under the name `matrix`,
  and old `recmat` files are still read. In SPECULA, `Matrix` files are looked up in `rec/`.
- **MVM and SoftLimiter** moved, with the new parameter names `matrix` / `matrix_list` and the
  old ones still accepted.
- **`VectorValuesDisplay`**, with `ModesDisplay` as a subclass.
- **Web frontend** moved, with the new commands `iteras_frontend_start` / `iteras_frontend_stop`
  (the specula commands still work).
- **Docs:**
  - API pages of the moved modules generated from the iteras modules (they had been empty since Phase 0);
  - iteras-only objects listed;
  - summary links fixed.
- **Simul:** defaults of the main parameters.
- **Tutorials:** step response and control stability run with iteras only, and are tested
  without specula.
- **FieldAnalyser / EfReplay:** results in any data object format (`data_format` argument).
- **Tests fixed:**
  - `to_xp` skip when the GPU is disabled;
  - display server queue race (tests re-enabled in CI);
  - tutorial test `PYTHONPATH`;
  - NFS-dependent calibration manager and time history tests.

## 6. Open decisions

- Separate repository (Phase 1) vs. publishing iteras as its own distribution from the
  SPECULA repository.
- Deprecation schedule for the `specula.*` alias modules (now: kept indefinitely).
- Whether `Intmat` should also derive from `Matrix`.

## 7. Smaller pending items

- `InfinitePhaseScreen` overrides `self.xp`, which is read-only for derived classes per the
  Extension API.
- Timing-dependent test: `test_trigger_puts_speed_report_after_one_second`.
- Pre-existing docstring formatting warnings in some iteras classes, visible now that their
  API pages have content (`BaseDataObj`, `DataPrint`, ...).
- The `iteras-allocator` branch (allocator work) needs a rebase onto `main` after Phase 2 is merged.
