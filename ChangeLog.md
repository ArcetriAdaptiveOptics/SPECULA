# SPECULA Changelog


## [Next version]

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
- ...

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

