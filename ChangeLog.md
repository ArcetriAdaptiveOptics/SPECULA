# SPECULA Changelog


## [Next version]

### New processing and data objects

- `IntensitySum`: sums a list of `Intensity` inputs (`in_i_list`) pixel by pixel into `out_i`. The output shape is taken from the first input; all inputs must have the same shape.

### Interface changes

- Added the `--async-displays` command-line flag (`async_displays` in `main_simul()` and `Simul`): all displays run in a separate process, so that slow drawing does not slow down the simulation. Data is copied to the CPU and sent through a queue, and updates are skipped while the display process is busy, except for displays with a history (`PlotDisplay`, `PlotVectorDisplay`, or any display with `skip_updates = False`), which never lose points. The display code, including `setup()` and `finalize()` of derived classes, runs in the display process. Not compatible with `DisplayRecorder`.
- `make_modal_base_from_ifs_fft()`: the `m2c` columns of the KL modes no longer apply a piston: `influence_functions.T @ m2c` now gives the modes of `kl_basis` (before, a few KL modes carried a piston of several percent of their peak, invisible to the WFS). `kl_basis` and the Zernike columns of `m2c` are unchanged; the KL columns of `m2c` change by a piston command.
- `ShSlopec` (#776): on GPU the trigger is captured in a CUDA graph, together with the slope corrections of `Slopec` (slope null, filtering, slopes map), about 9x faster per step. `Slopec`-derived classes now implement `compute_slopes()` instead of `trigger_code()` (overriding `trigger_code()` raises a `TypeError`): `Slopec.trigger_code()` calls it and then applies the slope corrections, which are no longer applied in `post_trigger()`. Classes derived from `ShSlopec` do not use the CUDA graph unless they call `build_stream()` in their `setup()`. The unused `ShSlopec.thr_mask_cube` output has been removed.
- Added `window_xy` ([x, y] screen pixels) to all displays except `DoublePhaseDisplay`, to place the window on screen with GUI backends that allow it (Tk, Qt, GTK); ignored on the others.
- Added `dark_frame_tag` to `DynamicDarkCalibrator`: dark frame file in `data_dir` loaded in `setup()` (an error is raised if it cannot be loaded).
- Added force limiting to `DM`: new `stiffness` (matrix, `stiffness_data` in YAML, requires `m2c`) and `max_force` parameters. If the forces exceed `max_force`, the highest-order modes are discarded (modes are assumed sorted by increasing spatial frequency), before the position `stroke` clipping. New outputs `out_forces` (forces of the applied command, empty without `stiffness`) and `out_force_nmodes` (number of modes kept).
- `AtmoEvolution`/`AtmoEvolutionUpDown` (#530): on GPU the trigger is captured in a CUDA graph, so their inputs must be updated in place by the producers (a reallocated input now raises an error). `airmass` is no longer an attribute, `scale_coeff` is now `scale_coef` (device array), `delta_time` is a scalar, and `last_position(s)`, `extra_delta_time(s)` and `last_effective_position` are device arrays.
- Added `thr_ratio_value` to `ShSlopec`: per-subaperture threshold, as a fraction of the brightest pixel of each subaperture (as in PASSATA). The code path existed but was unreachable and broken (it used the maximum flux over all subapertures).
- Added `pyr_max_side_ld` to `ModulatedPyramid` and its derived classes to cap the radial support of the pyramid surface in lambda/D units, forcing values outside the support radius to zero and enabling a central fifth pupil.
- Added `compute_single_im` (bool, default True) to `ImCalibrator` to optionally skip populating the `out_single_im` per-mode output (the output itself is always present, empty when disabled). When True (default, unchanged behavior) this costs an O(nmodes) Python loop on every `trigger_code()` call (not just push-pull events) plus roughly double the fixed memory (one extra Intmat per mode); set to False to skip that loop/memory when nothing downstream consumes `out_single_im` (only `out_intmat` is used elsewhere in this codebase) -- needed for large-nmodes, long calibrations.
- Removed `specula.lib.modal_pushpull_signal.modal_pushpull_signal()`, which duplicated the step ordering of `PushPullGenerator` and was only used by tests (it is kept in test\_generators.py as a reference implementation). Per-mode amplitudes are available from `modal_pushpull_amplitudes()`.
- `ModalAnalysis` now always computes the wavefront RMS: the `dorms` parameter is deprecated and ignored (a `FutureWarning` is issued if it is set). In list mode (`in_ef_list`) the per-input RMS is available from the new `rms_list` output, mirroring `out_modes_list`; `rms` is now only updated for the single input `in_ef` (previously, in list mode, it held the RMS of the last input only).
- MORFEO configurations: the LGS (and reference) tomographic reconstructors now use `ModalrecImplicitPolc` instead of `ModalrecExplicitPolc`, since the pseudo open-loop modes were not used. The `out_pseudo_ol_modes` output is not available anymore for these objects: use `ModalrecExplicitPolc` again if needed.
- `Modalrec`: the `nmodes` parameter, which was ignored, is deprecated (a `FutureWarning` is issued if it is set): the number of modes is set by the recmat, use `ncutmodes` to discard the last modes. Removed it from the ANDES configurations.
- Added `cache` to `ZernikeGenerator.getZernike()` (default True, unchanged behavior): if False, the polynomial is not cached.
- `TerminalInput` now keeps the input prompt on the last line of the terminal while log output scrolls above it (using `prompt_toolkit`, new dependency). `SpeculaInput` now reads input in a background thread instead of a separate process: `set_input_task()` is replaced by the thread-safe `put_input()`, which validates values immediately.

### Other

- `ModalrecExplicitPolc` and `ModalrecImplicitPolc` share the command inputs and the slopes update check in the new base class `BasePolcModalrec`.
- `compute_zern_ifunc()` (Zernike `IFunc` and `ModalAnalysis`) no longer keeps all full-frame Zernike polynomials in memory, and normalizes the modes in place (480 pixels, 1000 modes: peak GPU memory 4.0 -> 2.3 GB).
- Fixed `filt_modes` in `make_modal_base_from_ifs_fft()`, whose content was ignored: only their number was used, to drop the same number of the highest-order KL modes. They are now projected on the influence functions span and removed from the KL basis; modes outside the span, or duplicating piston, the Zernike modes or other `filt_modes`, are discarded with a warning.
- Fixed display grouping: displays can share a window again, each one in its own `subplot` (an error is raised only if the same subplot of a window is used twice). The window size is set by the first display of the window. Each simulation now starts with no windows, closing those of a previous simulation in the same process (with `--nsimul` or in a notebook, explicit window numbers raised an error).
- Fixed `DoublePhaseDisplay`, which always showed an error: its `trigger_code()` called `_update_display()` without the data.
- Fixed `AtmoPropagation` with `doFresnel` and `upwards` (#770): the field was conjugated before the propagation and after it, which amounts to propagating it backwards, so beam wander and scintillation came out mirrored with respect to the phase (the beam drifted towards -grad(phase) instead of +grad(phase)). The output phase and downwards propagation are unchanged; single realizations of upwards propagated fields change, their ensemble statistics do not. The docstring now states that in the Fresnel path the lowest layer is the pupil plane.
- Objects using a CUDA graph now raise an error if an input array is reallocated by its producer after the graph capture (it was silently read at the old address), a check previously done only by `AtmoEvolution`; derived classes whose graph does not read some inputs can exclude them overriding `graph_input_ptrs()`. `DynamicDarkCalibrator` now updates `out_subtracted_pixels` in place. Fixed `ExtSourcePyramid` modifying its `ext_source_coeff` input without CUDA graph (it appended the 4 face centers and zeroed the flux of the filtered points in the `ExtendedSource` output, so other objects reading it saw those changes): it now uses a local copy.
- Fixed `DynamicDarkCalibrator`: a dark frame loaded with `in_load` replaced the dark frame object, so the `out_darkframe` output kept the old one (it is now copied in place, with a shape check), and failed loads/saves raised an `AttributeError` while logging the error (also fixed in `DynamicPyrPupdataCalibrator` saves, now logged instead of printed). Fixed `PixelsPupDisplay` failing at the first update (`img` was not initialized).
- `BaseValue.restore()`: float arrays now follow the object precision (they kept the FITS dtype); other types, scalars and the host location are unchanged.
- `PhaseScreenCube`: fixed crash on GPU; the interpolator and its input ElectricField are built once instead of at every step (about 2x faster on CPU); added the `precision` parameter; raises a `ValueError` if the simulation starts before the first cube time (it silently used the last screen).
- `AtmoPropagation` with `doFresnel`: propagators are still computed in float64 but stored in the object precision, so the per-step FFTs and products are no longer in double (-35% time per step measured on a 1536x1536 padded case, same accuracy).
- `AtmoPropagation`: layers are interpolated directly into the temporary electric field, without allocating temporary arrays at each step (-36% time per step measured on GPU with 10 sources, 35 layers and a 480x480 pupil, bit-identical results).
- Added `specula.lib.affine_transform`: order-1 affine transform (as `ndimage.affine_transform`), with fast CPU paths for shifts/rot90/flips and cupyx on GPU.
- Fixed arrays not following the object precision (silent float64 computation with 32 bit precision) in `IFuncInv`, `ShSlopec` weights, `BaseOperation` (concat), `PowerLoss`, `PolyChromWFS`, `MirrorCommandsCombinator`, `MultirateComplementaryFilter`, `Lift` (the whole iteration ran in complex128) and `demodulate_signal()` (which was also forcing float32 in double precision). Fixed `SprintPyr` passing its internal command array as the `BaseValue` description.
- Added per-object timing of all trigger phases: NVTX ranges for Nsight Systems, and new `--trace-file`, `--trace-skip`, `--trace-sync` and `--trace-gpu-events` options to write them to a text file with a summary (see docs/profiling.rst). The tracer can also be used as a context manager or decorator to mark more sections of code; `show_in_profiler()` has been removed. Also fixed `--profile` failing.
- Fixed replays started at `start_time > 0` (e.g. FieldAnalyser/EfReplay with `start_time`), broken since the loop started at `t0` (#685): objects with iteration-based state (`TimeHistoryGenerator`, `VibrationGenerator`, `RandomGenerator` draws, `AtmoEvolution` screen shifts) started from the wrong state, so e.g. disturbances fed through a DM were misaligned with the replayed commands. `Simul.run` now pre-rolls from 0 to `t0-dt` only the objects that are not driven by the DataSource but feed the replayed part (plus replayed objects used as delayed inputs, e.g. `dm.out_layer:-1`), reproducing their state at `t0` exactly (`Simul.find_preroll_objects`, `LoopControl.preroll`). `t0` must be a multiple of the time step; pre-rolled objects on other MPI ranks are not supported. Added test\_replay\_preroll.py.
- Added `specula.lib.source_flux` with `phot_density_from_source_params()` (photons/s/m^2/nm from scalar source parameters, usable without instantiating a `Source`; `Source.phot_density()` now delegates to it) and `flux_per_pixel()` (simple scalar estimate of detected photo-electrons per pixel from magnitude, collecting area, bandwidth, integration time, throughput, QE and fraction of light on the pixel; no noise/extinction modeling). Added test\_source\_flux.py.
- `IFunc.inverse()` now computes the pseudoinverse via the smaller of the two Gram matrices (`specula.lib.fast_pinv`) instead of calling `xp.linalg.pinv` directly on the full influence-function matrix. Mathematically identical result (including in the rank-deficient case), but substantially faster for the typical case of many pixels and few modes -- measured 3x-8x on real KL/zonal influence-function bases, with the speedup growing with pixel count. Added test\_fast\_pinv.py.
- Fixed test\_im\_sh\_synim\_generator.py: on a machine with a GPU, the reference IM computed directly via `synim.interaction_matrix()` came back as a cupy array regardless of the test's own `target_device_idx`/`xp` (SynIM's backend is bound once at process start, see synim\_utils.py), crashing the comparison against the always-numpy generated IM with a cupy TypeError; also loosened test\_im\_generator\_no\_misreg/test\_im\_generator\_with\_misreg tolerances from 1e-10/1e-7 to 1e-6, since those paths go through `ImShSynimGenerator`'s float32 (`precision=1`) cast and cannot match a float64 reference to 1e-10 (CI failure was a real ~2.7e-8 relative mismatch, not a fluke).
- Implemented `InfinitePhaseScreen` cache for A/B extrusion matrices, saves about one minute in tests
- Removed duplicated calculation in `ModalAnalysis.trigger_code()`
- Fixed ExtSourcePyramid with cuda_stream_enable=True: the CUDA graphs kept reading the data from frame 0, but with FROM_PSF a coeff array is computed for every new PSF. Now a recapturing is performed if necessary.
- Fixed constructor type hints narrower than what the code accepts (#709):
- Fixed a bug in ModalAnalysis that was forcing a 64-bit computation even when SPECULA is running with 32 bit precision
- Fixed `DM` stroke clipping promoting the commands to 64-bit when SPECULA is running with 32 bit precision. Added tests on the `DM` output phase (slice, `idx_modes` and `m2c` paths) and on its precision. `DM` now raises a `ValueError` if the `m2c` rows do not match the influence function modes.
- Fixed remaining 64-bit promotions in `ModalAnalysis` phase unwrapping (Poisson right-hand side, Laplacian eigenvalues, pupil mask) and missing `precision` in the `out_modes_list` outputs. Removed the unreachable zero-padding branch of `ModalAnalysis.trigger_code()`, and the never-set `_doZeroPad` attribute (with its dead code) from `IFunc` and `IFuncInv`.
- `PushPullGenerator` no longer allocates the full `(n_steps, nmodes)` push-pull time history (mostly zeros, growing as nmodes^2 * ncycles * nsamples, e.g. ~40 GB for 5000 modes and 100 cycles): each step is now computed on the fly from the per-mode amplitudes and the pattern. The `time_hist` attribute has been removed (the sequence length is available as `nsteps`), and triggering past the end of the sequence raises an explicit `IndexError`. The amplitude computation has been factored out as `specula.lib.modal_pushpull_signal.modal_pushpull_amplitudes()`, and `modal_pushpull_signal()` has been removed (see Interface changes).
- Fixed an out-of-bounds read in the `Interp2D` bilinear CUDA kernel: on the last column it read the first element of the next row, and at the bottom-right corner one element past the end of the input array. The weight was zero, but NaN/Inf in that memory propagated to the output. Added GPU regression tests in test\_interp2d.py.
- Removed support for Python 3.8 and 3.9, minimum required version is now 3.10.

## [1.0.4] - 2026-08-19

### New processing and data objects

- Added IntValue, FloatValue and StringValue as specialized containers for scalars and strings, to be used in place of BaseValue where needed.
- Added DisplayRecorder processing object
- Added Phasescreen data object.
- Added Phase Extractor processing object.
- Added CLOSE gain optimizer processing object.
- Added RoundToMultiple processing object.
- Added EfReplay class (specula.ef_replay): replays a list of existing ElectricField/Layer outputs (e.g. an ElectricFieldCombinator or a DM's out_layer) exactly as they were in a past run, by targeting the existing object(s) directly with Simul.build_targeted_replay instead of synthesizing new off-axis sources like FieldAnalyser does. Complements FieldAnalyser for cases where a disturbance was injected downstream of AtmoPropagation (see FieldAnalyser's new "Limitation" tutorial section) and the exact original direction/sensor is what's needed. Added docs/tutorials/ef\_replay\_tutorial.rst and test\_ef\_replay.py

### Interface changes

- Removed simul\_params argument from IirFilter, Integrator and other related processing objects
- Outputs for SpeculaInput and derived objects like TerminalInput must be typed with :int, :float or :str
- Added "window" and "subplot" arguments to all displays to enable multi-plot windows
- Renamed MmsePistonUnwrapper to SoftLimiter and moved the module to specula.processing_objects.soft_limiter
- Added stroke thresholding for dm class
- Added open\_loop\_estimate parameter to OpticalGainEstimator.
- Enabled start and end time (start\_time and end\_time parameters) in FieldAnalyser.
- Added "out_window_id" output to all displays to support video recording
- Added "beam_center" for uplink beam in pixel. Used for Fresnel propagation to indicate if beam is not located in the center.
- Added pupil\_mask parameter to SprintShSynim, forwarded to BaseSprintEstimator as the WFS-side pupil (previously silently fell back to dm.mask, e.g. missing spider obscuration); added regression test in test\_sprint.py
- Enabled pyr_tlt_coeffs for the modulated_pyramid, allowing to correctly set different tilt coefficients for the pyramid faces
- Extracted FieldAnalyser's shared replay machinery (params loading, replay precision/downsampling checks, temp-simulation execution) into a new BaseReplayAnalyser base class, reused by EfReplay; pure refactor, no behavior change (mock patch targets for Simul/specula in test\_field\_analyser.py moved to specula.base\_replay\_analyser accordingly)
- Added "out_slopes_map" output to Slopec (and thus to all its subclasses, e.g. PyrSlopec, ShSlopec): a 2d remap of the slopes vector (shape (2, size\_x, size\_y) for a single subaperture, reusing the existing single\_mask/display\_map/get2d() machinery), useful to store slopes in DataStore with a (timesteps, 2, size\_x, size\_y) shape instead of a flat vector. Added "out_pixels_subap" (raw, pre-threshold pixel intensities of the 4 pyramid pupils, shape (4, size\_x, size\_y)) and "out_pixels_subap_sum" (their sum, shape (size\_x, size\_y), e.g. for scintillation analysis) outputs to PyrSlopec. Added PupData.local_display_map() helper. No changes needed to DataStore, which already saves whatever shape an output's get_value() returns.

### Other

- Fixed Simul.build\_targeted\_replay/FieldAnalyser silently dropping disturbances injected downstream of the replay target via ElectricFieldCombinator/PhaseScreenCube (SPECULA #696: e.g. a phase screen summed onto an AtmoPropagation source's output before the WFS), which could bias off-axis FieldAnalyser results with a spurious, direction-independent term. build\_targeted\_replay now raises ValueError by default (opt-out via on\_missing\_downstream\_consumers='warn'/'ignore') when such a silently-dropped ElectricField/Layer-producing consumer is detected; FieldAnalyser exposes the same on\_missing\_downstream\_consumers parameter (default 'error'). Added regression tests in test\_simul.py and test\_field\_analyser.py, and a new "Limitation" section in docs/tutorials/field\_analyser\_tutorial.rst
- Fixed RandomGenerator objects with no explicit `seed` not being reproducible across a replay (e.g. via Simul.build\_targeted\_replay/FieldAnalyser): the actually-resolved seed is now recorded in replay\_params.yml at the end of a run and re-injected on replay (Simul.inject\_recorded\_seeds), leaving fresh, non-replay runs unaffected (still ambient-random by default). Added RandomGenerator.get\_resolved\_seed() / BaseProcessingObj.get\_resolved\_seed() hook and DataSource random\_seeds parameter; added tests in test\_generators.py, test\_simul.py and test\_field\_analyser.py
- Fixed BaseOperation using stale/uninitialized input values when in\_value1 or in\_value2 had never been generated
- Fixed vecWeiPixRadT extraction in ShSlopec
- Fixed output\_names in PhaseScreenCube
- Fixed PhaseScreenCube crash on GPU due to np.searchsorted called on a cupy array
- Fixed start\_time bug in WindowedIntegration
- Fixed SprintShSynim's \_plot\_debug\_info passing GPU (cupy) arrays directly to matplotlib without cpuArray() conversion, crashing on GPU
- Corrected SprintShSynim's docstring/perturbation labels for enable\_wpup\_magn\_xy: params [4]/[5] are anamorphosis\_90/anamorphosis\_45 (functional in SynIM via compute\_im\_synim), not independent magn\_x/magn\_y as previously (incorrectly) documented as "not yet implemented"
- Added regression test (test\_sprint\_anamorphic\_magnification\_is\_functional) verifying enable\_wpup\_magn\_xy's anamorphosis\_90/anamorphosis\_45 parameters actually affect the computed nominal IM
- SPRINT logger lever changed to debug for intermediate steps
- Bumped synim requirement to 1.2.3 (was 1.1.3)
- Optimization of the compute\_ifs\_covmat function
- Added Fraunhofer far field propagation
- Fixed silent misparsing/confusing errors in split\_output() when an object, alias or output name contained a reserved '.', '-' or ':' character; added early validation of YAML section names in Simul
- Fixed A size in get_pyr_tlt, adding a round (rather than flooring by default) to avoid cases where the pyramid tilt mask (pyr_tlt) and the focal plane mask (fp_mask) could be of different sizes when using an odd number of pixels across the pupil
- Updated calculation of power loss such that reference PSF also uses Fresnel propagation
- Fixed ModulatedPyramid/ModulatedDoubleRoof.calc\_pyr\_geometry producing pupils smaller than the requested pup\_diam pixels whenever pup\_dist was large enough to trigger the fft\_res\_min increase (roughly pup\_dist > 1.73 \* pup\_diam with default pup\_margin): fft\_totsize grew with the bumped fft\_res, but toccd\_side (the internal CCD side the FFT-plane pupils are rebinned to via toccd()) stayed frozen at the value computed from the pre-bump, nominal fft\_res, shrinking the sub-pupils after rebinning. toccd\_side is now recomputed from the final fft\_res returned by calc\_geometry. Added regression test in test\_pyr.py

## [1.0.3] - 2026-05-18

### New processing and data objects:

- CiaoCiao WFS and slope computer
- Chromatic effects in atmospheric propagation
- Phasescreen cube processing object
- SpatioTempArray data object
- Interactive inputs, dynamic versions of pupil calibrators and dark calibrators for hardware-in-the-loop simulations
- Multi-rate modal reconstructor: selection of multiple reconstruction matrices depending on which inputs are valid at a given time
- Multi-rate complementary filter
- Separated modal reconstructor with explicit Pseudo-Open Loop algorithm into its own processing object
- PupilstopController: processing object for generation of pupilstop-like layers
- MMSE piston unwrapper processing object
- Added script to plot influence functions
- New parameters and interface changes:

### New parameters and interface changes

- Using the standard Python logging module instead of print(), verbose parameters removed, added --log-level command line argument
- Added optional downsampling in DataStore
- FieldAnalyzer support custom influence functions and optional displays
- Added band limit factor in AtmoPropagation
- Added layer height parameter in AtmoRandomPhase
- ElectrictFieldCombinator optionally accepts an input EF list instead of two separate EFs
- Added scaling factor to PhaseScreenCube
- Added computation of PSF profile and metrics to Psf and PsfCoronograph
- Added optional wavelength parameter (and tolerance) in ElectricField
- Removed value2\_is\_shorter parameter from BaseOperation, now automatically derived
- Removed unused parameters tag\_template from rec, subap and sn calibrators.
- Other updates and bugfixes:

### Other updates and bugfixes

- Support for python 3.14
- Changed phase unwrapping algorithm in ModalAnalysis, can run on GPU as well
- Fixed SH-like normalzation in pyramid slope computer
- Fixed #380 (wrong error message during initialization)
- Calculation precision handling
- Fixed SynIM dependency to version 1.1.3
- remove\_piston flag in IFunc inverse method
- Many other minor bugfixes

