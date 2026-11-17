# SPECULA Changelog

Entries are short; the PR linked to each one has the details.


## [Next version]

### New processing and data objects

- `GerchbergSaxton`: Gerchberg-Saxton phase retrieval, with the same inputs and outputs as `Lift` (#846)
- `NonlinearModalrec` and `NonlinearResponseCalibrator`: modal reconstructor with a per-mode saturation correction from calibrated response curves (#862)
- `Conv2dNetTrainer` and `Conv2dNetTester`: training and scoring of a CNN on 2D maps buffered during the simulation, with the new `Slopes2D`; needs torch (`pip install specula[nn]`) (#863)
- `DistributedExtSourcePyramid`: extended source points split over several GPUs (#874)
- `PeriodicRandomGenerator`: random value redrawn every `update_interval` seconds (#852)

### Interface changes

- The simulation framework (base classes, `Simul`, generic data and processing objects, math and control objects) is now the `iteras` package; `specula.*` imports, class names and parameter files are unchanged (#828, #843, #847)
- New Allocator (placement of the objects on nodes, processes and GPUs) and Accelerator (CUDA graphs of the simulation loop) (#851)
- `DataStore`/`DataSource`: new `npz` and `hdf5` data formats (#843)
- Accelerator: new `on_error` option; by default an object whose capture fails runs as Python code, with a warning (#917)
- `Intmat.generate_rec()` honours `cut_modes` (truncated SVD) (#873)
- `DataBuffer`: new `align` option (default `True`), emits only the steps in which all the inputs were updated (#852)
- `WaveGenerator`: new `duty_cycle` option for square waves (#852)
- `PolyChromSH`, `PolyCromPyramid`: new `n_devices`, to spread the wavelengths over several GPUs (#875)
- `toccd()`: new `out` argument (#885)
- Every input declared in `input_names()` must be created by the constructor (`optional=True` for unconnected ones); `sanity_check()` runs before `setup()` (#916)
- Accelerator on by default (`--no-accelerate` to turn it off) and the only CUDA graph mechanism: `build_stream()` and the object streams removed, classes opt in with `can_accelerate()`; new `internal_objects()` (#897)
- Accelerator: matrix products between objects with different precisions computed in one precision; change `Modalrec.recmat` by assignment, not in place (#919)
- New class attribute `host_data_objects`: data objects restored on the CPU; `ModalrecImplicitPolc` keeps its input matrices on the host (MORFEO GPU peak 18.0 -> 11.3 GB) (#904)
- New `--data-objects-on-device` option, to restore all data objects on the GPU (#910)
- `DM`: sparse product for sparse influence functions (e.g. M4); change them with the `DM.ifunc` setter, in-place changes are not seen (#901)
- New `Slopec` parameter `clamp_negative` (default `True`), now also used by `ZwfsSlopec`, `CurWfsSlopec` and `CiaoCiaoSlopec`; zero or negative normalizations give zero slopes (#890)
- `Slopes`: 2d display geometry in a `SlopesLayout` object, `single_mask`/`display_map` deprecated (#858)
- New `get_metadata_arrays()`/`set_metadata_arrays()` in data objects, saved by `DataStore` and restored by `DataSource` (stored slopes keep their 2d layout) (#858)
- `LaserLaunchTelescope`: new `absolute_coordinates` parameter, to share one launcher among WFSs with different rotations (#892)
- `ModulatedPyramid`: new optional input `in_misregistration` (pupil shifts and rotation) (#861)
- `DynamicIntegrator`, `DynamicIirFilter`: the `reset` input accepts a `BaseValue` and resets only on non-zero values (#860)
- `ScheduleGenerator`: new `period` option; `TimeHistoryGenerator`: new `loop` option (#860)
- `DistributedSH`: default `fov_ovs_coeff` is 2.0, as in `SH` (#866)
- Allocator: may split an object over several GPUs of a node (`split_objects`, default on), e.g. an `SH` into a `DistributedSH`; new class method `split_options()` (#866)
- `IirFilterData`: exact `max_stable_gain()`; `resonance_frequency()` returns the peak of the RTF (#850)
- `IirFilterData`: the analysis methods need an explicit plant (`delay`, `dm`/`nw`/`dw` or `unity_plant=True`); `get_zeros()`/`get_poles()` give NaN for unused entries (#849)
- `IirFilterData`: transfer functions and plots are discrete-time by default (`dt=True`) (#848)
- `IirFilterData.lpf_from_fc()` and `LowPassFilter` with `n_ord=1`: first-order Butterworth filter (#848)
- `PhaseUnwrapper`: new interval unwrapping mode (`interval_unwrap`, `err_1`, `err_2`) (#841)
- `SH`: exact integration of the PSF over the detector pixels, the pixel scale is exactly `sensor_pxscale`; `set_fov_res_to_turbpxsc` deprecated, `squaremask=False` not supported; builds on #836 by @salma2vec (#839)
- `SH` with LGS: closed-form Fourier coefficients of the convolution kernels, no kernel files saved (#908)
- `ModulatedPyramid`: physical model of the pyramid defects and new `pyr_roof_ld` roof defect; `fp_obs` is a diameter in lambda/D; new `default_mod_step()` (#912)
- `PolyCromPyramid`: parameters in lambda/D given at a reference wavelength (new `ref_wavelength_idx`) (#912)
- `ModulatedDoubleRoof`: defects follow the roof geometry, `pyr_max_side_ld` is applied, `pyr_tip_def_ld` raises an error (#912)

### Other

- Fixed `ShSlopec` `vecWeiPixRadT` given as a list, e.g. from YAML (#924)
- Fixed `ImShSynimGenerator` with `compute_rec=False` (#915)
- Async displays: the display process exits when the simulation dies (#914)
- Fixed `FocalPlaneFilter` `fp_obs`, now the obstruction diameter in lambda/D (#913)
- `PsfCoronagraph`: profiles in contrast units, new azimuthal std profile output (#909)
- Documentation: separate iteras section, builds without warnings (#898); functions and classes of `specula.lib` in the API page (#896)
- `AtmoPropagation`: lower GPU memory peak in `setup()` with the fused kernel (#895)
- Fixed logging typo in `calc_noise_cov_elong()` (#891)
- Fixed 2d display of `ZwfsSlopec` and `CurWfsSlopec` slopes (#858)
- New `display_lgs_geometry()`: plot of the LGS geometry seen by an SH (#893)
- Fixed `calc_noise_cov_elong()` subaperture positions (centres instead of corners) (#892)
- Fixed `make_mask` spiders: requested width, centred on the pupil (#879)
- Fixed `lgs_map_sh()` sampling: pixel centres for focal plane and subapertures (#878)
- `Interp2D`: new `outside` argument, `EFInterpolator` no longer copies edge values outside the input array (#881)
- Fixed bias of the edge extrapolation with a single valid neighbour (#877)
- Fixed `Interp2D` default grid, which magnified the field by N/(N-1) (#877)
- `AtmoPropagation`: all sources and layers in a single GPU kernel (MORFEO: 7.5 -> 1.1 ms per step) (#869)
- `Lift`: modes resampled over the valid pupil pixels only (#876)
- SPRINT: Gauss-Newton fit of all the parameters together, new `param_tolerance` (#880)
- CuPy 12 fixes (#887)
- Fixed `EFInterpolator.update_parameters()` on interpolators without interpolation (#861)
- Fixed `PeriodicRandomGenerator` drift with intervals that are not a whole number of steps (#860)
- MPI: GPU-direct transfers with `ITERAS_MPI_GPU_DIRECT=1` (needs a CUDA-aware MPI) (#866)
- Allocator: models the work repeated by each part of a split object; placement search twice as fast; split objects placed only in processes driving all their GPUs (#866)
- Fixed `DistributedSH` slices on other GPUs computing wrong images with MPI inputs (#866)
- Fixed `LowPassFilter` without `n_ord` and `amplif_fact` (#848)
- `IirFilterData`: complex zeros and poles; `num`/`den` of different widths aligned as in `IirFilter` (#848)
- Fixed sign error in far-field propagation and Fresnel propagation distance, added power-loss warning (#842)
- `ModulatedPyramid`: one modulation step at a time, pruned FFT (-80% GPU memory, 20-40% faster) (#886)
- GPU memory: memory cached by the cupy pools released after setup and first trigger (MORFEO 19 -> 10 GB) (#902), `IFunc`/`IFuncInv` restored without copies (#903), file uploads staged in a temporary pool (#905)
- `SH`: field of each subaperture row computed directly into the padded cube (#900)
- `PSF`: preallocated buffers and in-place FFT (#884)
- `compute_ifs_covmat()`: batched FFTs and tiled symmetric product (#889, #899)
- `PolyChromSH`, `PolyCromPyramid`: the instances use their own CUDA graphs (#882)
- `MVM`, `SsrFilter` and `MirrorCommandsCombinator` can be accelerated (#853)
- `rebin2d()`: faster on GPU (#883); FITS `has()` reads only the header (#894)
- `toccd()`: exact area weights, much faster; fixed non-square and non-contiguous arrays on GPU (#885)
- Fixed `BaseOperation` ignoring `value2_remap` when `value2` is shorter than `value1`; fixed the capture of `BaseSlicer` and `PSF` (#917)
- Fixed `PAPLCoronagraph` pupil stop, which was not applied (#872)
- `AtmoInfiniteEvolution`: bilinear interpolation of the fractional shift, positions in double precision (#854)
- Fixed `transferDataTo()` skipping scalar attributes (#852)
- Fixed `DistributedSH` rows not computed when `n_slices` does not divide `subap_on_diameter` (#859)
- Fixed display server startup on Windows and `LoopControl.niters()` rounding (#864)
- Removed unused code (#857, #870)
- Docs: science camera tutorial and static aberrations (#845)

### Upgrade notes

- Recalibrate configurations with an SH (interaction matrices, reconstructors, slope nulls): new pixel integration (#839), `Interp2D` grid (#877), LGS kernel sampling (#878)
- Objects resampling their input with `EFInterpolator` (SH, pyramid) give slopes larger by 1/N and shift their pupils by up to 1 px: check pyramid pupil data and hand-tuned `pup_shifts` (#877)
- Elongated LGS noise covariances, and the MMSE reconstructors using them, change (#878, #892)
- `ZwfsSlopec`, `CurWfsSlopec` and `CiaoCiaoSlopec` now clamp negative pixels: `clamp_negative: false` restores the previous results (#890)
- Pyramid results change with `pyr_tlt_coeff` other than 1, `pyr_tip_maya_ld`, edge/tip defects, `fp_obs`, and for `PolyCromPyramid` with more than one wavelength (#912); `FocalPlaneFilter` with `fp_obs` (#913)
- `DistributedSH` without `fov_ovs_coeff` changes (#866)
- First-order low-pass filters change (#848); pass the plant explicitly to the `IirFilterData` analysis methods (#849), and `dt=None` for the previous continuous-time functions (#848)
- `make_mask` spiders are thinner: use a larger `spider_width` (about 5 instead of 3) to keep a similar mask (#879)
- `PsfCoronagraph` integrated and std profiles were too small by the PSF peak (#909); `Lift` and SPRINT estimates change (#876, #880)
- Reconstructors computed with `cut_modes` change (#873)
- With connected objects in different precisions, the precision of some matrix products changes (#919)
- Runs using `BaseOperation` with `value2_remap` and a shorter `value2` (e.g. MORFEO `subtract_ref`) change (#917)
- `PAPLCoronagraph` results change, also with the default stop ratios (#872)
- `AtmoInfiniteEvolution` layers change for winds not along an axis (#854)
- `DataBuffer` drops the steps in which not all inputs were updated: `align: false` keeps them (#852)
- Custom processing objects: create every declared input in the constructor, not in `setup()` (#916); override `can_accelerate()` instead of calling `build_stream()` (#897)

### Other

- `config/` cleanup (#922).

## [1.1.0] - 2026-10-20

### New processing and data objects

- `GreyFilter`: neutral density filter for an `Intensity`, with fixed or time-varying transmission (#804)
- `AVC`: Adaptive Vibration Cancellation of narrowband vibrations (#820)
- `IntensitySum`: pixel-by-pixel sum of a list of `Intensity` (#797)
- `PhaseUnwrapper`: two-wavelength unwrapping of segment pistons (#727)

### Interface changes

- New `DM.modal_ifunc()`, used by all objects that need the DM modes; `Intmat.generate_rec_mmse()` lost its `m2c` argument (#816):
  - `RecCalibrator` (MMSE): turbulence prior on the DM input modes (was wrong with `start_mode`/`idx_modes`); raises an error if `dm` is not given
  - `SprintPyr`: fixed internal DM input with `start_mode`/`idx_modes`
  - `ImShSynimGenerator`, `SprintShSynim`: interaction matrix columns are the DM input commands
  - `platescale_coeff()` and `LinearCombination`: plate scale modes are always modes 2-4 of the full DM basis; `start_modes` removed/deprecated
- `ImCalibrator`, `RecCalibrator`: `first_mode` deprecated, use `first_mode` of `PushPullGenerator` (#829)
- Delays other than `0` and `-1` in output references raise an error (#813)
- `DM`: mode selection resolved once at init; `nmodes` is now the input command length (#799)
- New `--async-displays` option: displays in a separate process (#781)
- `make_modal_base_from_ifs_fft()`: KL columns of `m2c` without piston (#787)
- `ShSlopec`: trigger captured in a CUDA graph (9x faster); `Slopec` subclasses implement `compute_slopes()` instead of `trigger_code()` (#776)
- Displays: new `window_xy` parameter for the window position (#726)
- `DynamicDarkCalibrator`: new `dark_frame_tag` (#726)
- `DM`: force limiting with new `stiffness` and `max_force` parameters, outputs `out_forces` and `out_force_nmodes` (#754)
- `AtmoEvolution`/`AtmoEvolutionUpDown`: trigger captured in a CUDA graph, inputs must be updated in place (#757)
- `ShSlopec`: new `thr_ratio_value` (per-subaperture threshold) (#755)
- `ModulatedPyramid`: new `pyr_max_side_ld` (#720)
- `ImCalibrator`: new `compute_single_im`, to save time and memory with many modes (#723)
- Removed `modal_pushpull_signal()`, use `modal_pushpull_amplitudes()` (#751)
- `ModalAnalysis`: RMS always computed, `dorms` deprecated, new `rms_list` output (#753)
- `ModalAnalysis`: one matrix product for all inputs, new `out_modes_{i}` outputs, `unwrap_2d()` removed (#825)
- `FieldAnalyser`: a single `ModalAnalysis` for all sources (#825)
- MORFEO configurations: LGS reconstructors use `ModalrecImplicitPolc` (#793)
- `Modalrec`: `nmodes` deprecated, use `ncutmodes` (#793)
- `ZernikeGenerator.getZernike()`: new `cache` argument (#792)
- `Simul`: `xxx_object` parameters restored from the calibration subdirectory of their class; `Slopes` from `slopenulls/` (#822)
- `TerminalInput`: prompt kept on the last line (new dependency `prompt_toolkit`); `SpeculaInput.set_input_task()` replaced by `put_input()` (#736)
- New `specula_list_runs` command: lists TN folders and the parameters that change between them (#794)
- Creating objects before `specula.init()`, or calling it after importing SPECULA submodules, raises an error (#743)
- `Simul`: setup order follows all the inputs, delayed ones included (#821)
- `Lift`: output modal coefficients wrapped to [-lambda/2, +lambda/2); `gui_pos` ignored (#727)
- Displays: `window` must be a unique positive integer, assigned automatically if missing (#733)

### Other

- `ModalAnalysis`: fixed `ifunc_inv` with `nmodes`, faster setup; `check_output_names()` exact names take precedence over patterns (#825)
- Fixed the circular loop error message of `Simul.build_trigger_order()` (#812)
- Fixed `ModulatedDoubleRoof` PSF outputs, which were doubled (#803)
- New `BasePolcModalrec` base class (#793)
- `compute_zern_ifunc()`: less GPU memory (#792)
- Fixed `filt_modes` in `make_modal_base_from_ifs_fft()`, which was ignored (#787)
- Fixed display grouping in a window and `DoublePhaseDisplay` (#783)
- Fixed upwards Fresnel propagation in `AtmoPropagation`, which propagated backwards (#777)
- CUDA graphs: error if an input array is reallocated after the capture; fixed `ExtSourcePyramid` modifying its input (#776)
- Fixed `DynamicDarkCalibrator` loads and saves and `PixelsPupDisplay` (#726)
- Precision fixes: `BaseValue.restore()`, `PhaseScreenCube` (also GPU crash and speed), `AtmoPropagation` Fresnel propagators (#756)
- `AtmoPropagation`: no temporary arrays per step (-36% time) (#769)
- New `specula.lib.affine_transform` (#757)
- Fixed float64 computations with 32 bit precision in several objects (#755, #748, #753, #754)
- Per-object timing with NVTX ranges and new `--trace-*` options (see docs/profiling.rst); fixed `--profile` (#742)
- Fixed replays started at `start_time > 0` (#739)
- New `specula.lib.source_flux` (#738)
- `IFunc.inverse()`: faster pseudoinverse (#722)
- Fixed SynIM tests on GPU (#734)
- `InfinitePhaseScreen`: cache of the extrusion matrices (#746)
- `ModalAnalysis`: removed duplicated computation (#744)
- Fixed `ExtSourcePyramid` CUDA graphs with a new PSF at each frame (#737)
- Fixed constructor type hints narrower than the accepted types (#749)
- `PushPullGenerator`: steps computed on the fly instead of storing the full time history (#751)
- Fixed out-of-bounds read in the `Interp2D` CUDA kernel (#762)
- Removed support for Python 3.8 and 3.9 (#773)
- Fixed GPU memory leak of the CUDA graph memory pools (#826)
- CUDA graph temporaries in a private memory pool (#768); `AtmoEvolution` no longer needs its own stream (#788)
- Fixed GPU memory tracking per object (#745, #819)
- `SH`: faster trigger, interpolation in the CUDA graph, geometry fixes, LGS kernels recomputed only when needed (#759, #763, #764, #775)
- `ConvolutionKernel`: half-spectrum kernels, shared between SHs, faster processing (#765, #789, #782)
- New `load_fits_array()`: fast loading of large FITS arrays (#818)
- `compute_zonal_ifunc()`: faster thin plate spline (#761)
- `demodulate_signal()`: O(n) cumulated demodulation (#810)
- SPRINT: relative error independent of the probe amplitude (#811)
- MPI: sequential connection tags (#824); an exception in one rank aborts all (#729)
- Fixed `DataStore` race creating TN folders (#830)
- macOS and Windows fixes (#832, #831)
- Display fixes (#815, #774)
- Minor fixes (#747, #758, #750, #772)
- Docs: ELT segmented-pupil / Soft-Limiter tutorials (#791), MMSE petal tutorial fix (#795)

### Upgrade notes

- MMSE reconstructors computed with `start_mode` or `idx_modes` change (#816)
- `first_mode` of `ImCalibrator`/`RecCalibrator` and `start_modes` of `LinearCombination` must be removed (#829, #816); delays like `:-2` raise an error (#813)
- `DM.nmodes` meaning changed, too large `nmodes` raises an error (#799)
- `Slopec` subclasses: implement `compute_slopes()` (#776)
- Producers of `AtmoEvolution` inputs must update them in place (#757)
- `Slopes` objects are read from `slopenulls/` (#822)
- Call `specula.init()` before creating objects and importing submodules (#743)
- New dependency `prompt_toolkit` (#736); Python >= 3.10 (#773)
- Single realizations of upwards Fresnel propagated fields change (#777)

## [1.0.4] - 2026-08-19

### New processing and data objects

- `IntValue`, `FloatValue` and `StringValue` data objects (#639)
- `DisplayRecorder` processing object (#684)
- `Phasescreen` data object (#663)
- `PhaseExtractor` processing object (#687)
- CLOSE gain optimizer processing object (#690)
- `RoundToMultiple` processing object (#715)
- `EfReplay`: replays existing `ElectricField`/`Layer` outputs of a past run (#703)

### Interface changes

- Removed `simul_params` argument from `IirFilter`, `Integrator` and related objects (#660)
- `SpeculaInput` and `TerminalInput` outputs must be typed with `:int`, `:float` or `:str` (#639)
- Displays: new `window` and `subplot` arguments for multi-plot windows, new `out_window_id` output (#684)
- Renamed `MmsePistonUnwrapper` to `SoftLimiter` (#679)
- `DM`: stroke thresholding (#686)
- `OpticalGainEstimator`: new `open_loop_estimate` (#671)
- `FieldAnalyser`: new `start_time` and `end_time` (#685)
- New `beam_center` for off-centre uplink beams in Fresnel propagation (#691)
- `SprintShSynim`: new `pupil_mask` parameter (#697)
- `ModulatedPyramid`: `pyr_tlt_coeffs` for the pyramid faces (#710)
- New `BaseReplayAnalyser` base class of `FieldAnalyser` and `EfReplay` (#703)
- `Slopec`: new `out_slopes_map` output; `PyrSlopec`: new `out_pixels_subap` and `out_pixels_subap_sum` outputs (#718)

### Other

- `build_targeted_replay()`/`FieldAnalyser`: raise an error when disturbances injected downstream of the replay target would be dropped (#702)
- Fixed reproducibility of `RandomGenerator` without `seed` in replays (#701)
- Fixed `BaseOperation` with inputs never generated (#693)
- Fixed `vecWeiPixRadT` in `ShSlopec` (#672)
- Fixed `PhaseScreenCube` output names and GPU crash (#673, #694)
- Fixed `start_time` in `WindowedIntegration` (#676)
- `SprintShSynim`: fixed debug plots on GPU, corrected anamorphosis documentation (#699)
- SPRINT intermediate steps logged at debug level; synim >= 1.2.3 (#704)
- Faster `compute_ifs_covmat()` (#680)
- Fraunhofer far-field propagation (#691)
- Reserved characters '.', '-' and ':' rejected in object and output names (#698)
- Fixed pyramid tilt and focal plane mask sizes with an odd number of pupil pixels (#713)
- Power loss reference PSF with Fresnel propagation (#711)
- Fixed pyramid pupils smaller than `pup_diam` with large `pup_dist` (#719)

### Upgrade notes

- Rename `MmsePistonUnwrapper` to `SoftLimiter` (#679), remove `simul_params` from filters (#660), type the `SpeculaInput` outputs (#639)
- `FieldAnalyser` raises an error when it would drop downstream disturbances, `on_missing_downstream_consumers` to change it (#702)

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

