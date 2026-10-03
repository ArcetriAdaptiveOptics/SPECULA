.. _elt_petal_soft_limiter_closed_loop_tutorial:

Closed-Loop ELT SCAO with the Soft-Limiter
=============================================

This is Part 3 of the segmented-pupil tutorial series. It calibrates a
pyramid WFS against the shared pupil built in
:ref:`elt_segmented_dm_tutorial` (Part 1), wires it into a full SCAO
closed loop together with the MMSE petal reconstructor from
:ref:`elt_petal_mmse_reconstructor_tutorial` (Part 2), and runs that loop
with and without the *Soft-Limiter* to show its effect on the petal
fringe-jump ambiguity.

**What you'll learn:**

* Calibrating the pupil geometry of a pyramid WFS
  (:class:`PyrPupdataCalibrator`) and a classical interaction/reconstruction
  matrix pair, on the exact same pupil and DM products already built in
  Part 1 and Part 2
* Why a single, uniform integrator gain does not work across thousands of
  modes, and how a per-mode-group :class:`IirFilterData`/:class:`IirFilter`
  scheme addresses that
* Wiring the :class:`SoftLimiter` into the loop between the integrator and
  the DM, using the two :class:`Recmat` objects Part 2 produced
* Using the differential petal inverse from Part 1 as an independent,
  ground-truth diagnostic of the true petal state, separate from anything
  the loop itself estimates or corrects
* Comparing a Classical loop (Soft-Limiter leak gain forced to zero)
  against the Soft-Limiter loop on the same turbulence, and reading the
  fringe-order/Strehl diagnostics that show the difference

**Prerequisites:**

* :ref:`elt_segmented_dm_tutorial` and
  :ref:`elt_petal_mmse_reconstructor_tutorial` completed: the full,
  ELT-class run, not a shrunk-down one, so that the pupil,
  influence-function, modal-basis and MMSE-reconstructor products they
  produce are available by tag
* Basic familiarity with closed-loop AO control (integrators, interaction
  and reconstruction matrices) and, ideally, a look at
  :doc:`control_stability_analysis` for the IIR filter convention used
  below

.. warning::

   **This is not a quick tutorial to run either.** Both the interaction
   matrix calibration (a push-pull over 4000 modes) and the closed loop
   itself (30 s of simulated time, 240x240-pixel detector) are
   substantially more expensive than Part 1 or Part 2. A GPU is strongly
   recommended (pass its index to ``specula.init()`` as before), and even
   then, expect this to take considerably longer than the few minutes
   Part 2 needed. Run it as a background job.

The problem, briefly
-----------------------

Part 2 built an MMSE reconstructor that turns a KL-mode command vector
into an estimate of the current *relative* petal-piston state. That
estimate is exactly what the Soft-Limiter needs to do its job: a pyramid
WFS cannot tell a petal piston offset from that same offset plus an
integer number of sensing wavelengths, so a plain integrator has no way to
correct the integer part and will let it drift. The Soft-Limiter drains
that unobservable component from the accumulated command with a leak
gain, retaining the fractional (observable, correctable) part. See the
paper this tutorial series accompanies for the full derivation. This part
closes the loop and shows the effect in practice.

Step 1: Restoring the products of Part 1 and Part 2
--------------------------------------------------------

Nothing is regenerated here. Every object below is restored from the
:class:`CalibManager` by the tag it was saved under in Part 1 or Part 2:
the pupilstop (``ELT39_6petals``), the KL-mode DM influence functions
(``ELT39_KL4000``), the differential 5-petal inverse
(``ELT39_6petals_inv``), and the two MMSE reconstructor
:class:`Recmat` objects (``ELT39_modes_to_petals_mmse`` and
``ELT39_petals_to_modes``). A SPECULA simulation YAML file references
these purely by tag; there is no Python restoration code to write for
this part, only YAML.

Step 2: Calibrating the pyramid pupil geometry
----------------------------------------------------

Before a :class:`PyrSlopec` can turn pyramid frames into slopes, it needs
to know which detector pixels belong to which of the four pupils of the
pyramid. :class:`PyrPupdataCalibrator` derives that mapping from a single
frame with an oversized modulation radius (so the four pupils are cleanly
separated and easy to segment), independent of any DM or turbulence:

.. code-block:: yaml

    pyr_pupdata:
      class: 'PyrPupdataCalibrator'
      thr1: 0.1
      thr2: 0.25
      output_tag: 'ELT39_pupdata'
      overwrite: True
      inputs:
        in_i: 'pyramid.out_i'

    prop_override:
      inputs:
        common_layer_list: ['pupilstop']

    pyramid_override:
      mod_amp: 10.0

    remove: ['slopec', 'dm', 'pushpull']

combined with a base file that defines ``pupilstop``, ``prop`` and
``pyramid`` against the shared pupil (``pupilstop`` tag ``ELT39_6petals``,
same pupil-sampling parameters as Part 1: 400x400 pixels,
39 m/400 px pixel pitch). The ``_override``/``remove`` blocks here follow
the same convention used throughout this tutorial series and in
:ref:`scao_basic_tutorial`: layer on top of a shared base file rather than
duplicating it. This step is cheap: a single frame.

Step 3: Calibrating the classical interaction and reconstruction matrices
------------------------------------------------------------------------------

With the pupil geometry known, calibrate the standard (non-petal) part of
the loop: a push-pull interaction matrix over all 4000 KL modes, and its
pseudoinverse.

.. code-block:: yaml

    im_calibrator:
      class:     'ImCalibrator'
      nmodes:    4000
      im_tag:    'ELT39_KL4000_im'
      overwrite: True
      inputs:
        in_slopes:   'slopec.out_slopes'
        in_commands: 'pushpull.output'
      outputs: ['out_intmat']

    rec_calibrator:
      class:     'RecCalibrator'
      nmodes:    4000
      rec_tag:   'ELT39_KL4000_rec'
      overwrite: True
      inputs:
        in_intmat:   'im_calibrator.out_intmat'

    prop_override:
      inputs:
        common_layer_list: ['pupilstop', 'dm.out_layer']

This reuses the same ``dm``/``pushpull`` blocks already implied by the DM
defined in Part 1 (``ifunc_object: 'ELT39_KL4000'``, ``nmodes: 4000``): a
:class:`PushPullGenerator` steps through all 4000 modes one at a time
while :class:`ImCalibrator` records the corresponding slopes, the same
generic pattern as :ref:`scao_basic_tutorial`. As a sanity check once this
finishes, ``rec @ im`` should come out as a clean 4000x4000 identity
(diagonal at 1.0 to within numerical noise, everything else at machine
noise). If it doesn't, look at the WFS/DM sign convention before going any
further: a closed loop will amplify rather than mask a mistake here.

Step 4: The closed-loop simulation
---------------------------------------

Everything now comes together in one YAML file. Walking through it block
by block:

**Atmosphere.** A single 25 m outer-scale, ground-layer turbulence layer
at 1.0 arcsec seeing, 15 m/s wind:

.. code-block:: yaml

    atmo:
      class:                'AtmoEvolution'
      simul_params_ref:     'main'
      L0:                   25
      heights:              [0.]
      Cn2:                  [1.0]
      fov:                  0.0
      seed:                 1
      inputs:
        seeing: 'seeing.output'
        wind_speed: 'wind_speed.output'
        wind_direction: 'wind_direction.output'
      outputs: ['layer_list']

A single layer keeps the focus on the petal-piston problem rather than on
multi-layer atmospheric reconstruction, which is orthogonal to what this
tutorial demonstrates.

**WFS path.** A modulated pyramid (3 :math:`\lambda/D` modulation radius,
800 nm) observing an 8th-magnitude on-axis NGS (``wfs_source``, defined in
the same shared base file as ``pupilstop`` and ``prop``), feeding a noisy
CCD (photon and readout noise both enabled, unlike the noise-free
calibration steps) and the slope computer calibrated in Step 2:

.. code-block:: yaml

    pyramid:
      class:             'ModulatedPyramid'
      simul_params_ref:  'main'
      pup_diam:          100.0
      pup_dist:          120.0
      fov:               2.1
      mod_amp:           3.0
      output_resolution: 240
      wavelengthInNm:    800
      inputs:
        in_ef: 'prop.out_wfs_source_ef'
      outputs: ['out_i']

    detector:
      class:             'CCD'
      simul_params_ref:  'main'
      size:              [240, 240]
      dt:                0.002
      photon_noise:      true
      readout_noise:     true
      inputs:
        in_i: 'pyramid.out_i'
      outputs: ['out_pixels']

    slopec:
      class:             'PyrSlopec'
      pupdata_object:    'ELT39_pupdata'
      slopes_from_intensity: true
      inputs:
        in_pixels: 'detector.out_pixels'
      outputs: ['out_slopes']

**Modal reconstruction and temporal control.** :class:`Modalrec` turns
slopes into a 4000-mode residual using the reconstruction matrix from
Step 3. That residual then goes through a per-mode-group IIR filter,
*not* a single flat integrator:

.. code-block:: yaml

    modalrec:
      class:              'Modalrec'
      recmat_object:      'ELT39_KL4000_rec'
      nmodes:             4000
      inputs:
        in_slopes: 'slopec.out_slopes'
      outputs: ['out_modes']

    temporal_filter:
      class:            'IirFilterData'
      ordnum:           [3,    2,    2,    2,    2,    2]
      ordden:           [3,    2,    2,    2,    2,    2]
      num: [[0.39168, -1.3312, 1.024], [0.0, 0.9925, 0.0], [0.0, 0.7855, 0.0],
            [0.0, 0.5685, 0.0], [0.0, 0.5085, 0.0], [0.0, 0.4635, 0.0]]
      den: [[0.995, -1.995, 1.0], [-1.0, 1.0, 0.0], [-0.995, 1.0, 0.0],
            [-0.98, 1.0, 0.0], [-0.96, 1.0, 0.0], [-0.94, 1.0, 0.0]]
      n_modes:          [2, 98, 900, 1000, 1000, 1000]

    control:
      class:                'IirFilter'
      delay:                2
      iir_filter_data_ref:  'temporal_filter'
      inputs:
        delta_comm: 'modalrec.out_modes'
        in_ost: 'unwrapper.out_ost:-1'
      outputs: ['out_comm']

.. note::

   **Why not one flat gain for all 4000 modes.** It is not a matter of
   robustness or noise: at 8th magnitude the WFS SNR is good, and a flat
   gain would be robust enough on that count alone. The real reason is
   that the optical gain of the pyramid is not flat across modes -- how
   much slope signal a given wavefront error produces depends strongly on
   spatial frequency -- so a single loop gain over- or under-drives large
   parts of the basis. Leak on the higher-order bands, increasingly strong
   toward the highest band, also limits how much spatial aliasing of the
   turbulence those modes can accumulate. ``temporal_filter`` groups the
   4000 modes into six bands (sizes given by ``n_modes``), each with its
   own IIR pole via ``num``/``den``: band 2 (98 modes) is an exact pure
   integrator (``den`` = ``[-1, 1, 0]``, infinite DC gain); bands 3-6
   progressively trade gain for leak as mode order increases (gain 0.7855
   down to 0.4635, pole 0.995 down to 0.94), so band 6 is the most leaky
   and the least aggressive of the six, not the fastest. See
   :doc:`control_stability_analysis` for the transfer-function
   convention and how to derive/verify a per-band pole from a stability
   margin. This is a deliberately coarse scheme tuned for this pupil/DM
   configuration at 1.0 arcsec seeing, not a real per-mode optimization --
   treat it as a starting point to adapt.

.. note::

   **Band 1 (tip-tilt) is different.** It uses a two-pole/two-zero IIR
   filter, not a plain integrator: the same design used for the ANDES
   tip-tilt channel to reject the ELT M2 wind-shake (up to ~10 :math:`\mu`\ m
   RMS); see Agapito et al., "MORFEO control strategy", *J. Astron.
   Telesc. Instrum. Syst.* 12(3), 039001 (2026), Appendix A, for the
   derivation of these coefficients. The atmosphere used in this tutorial is a
   single turbulence layer with no injected wind-shake, so a plain
   integrator on tip-tilt would perform just as well here; the filter is
   kept to mirror the real ANDES architecture, useful if you later add
   M1/M2 dynamics of your own.

.. note::

   The ``in_ost`` input on ``control`` closes a second loop: it lets the
   Soft-Limiter leak correction feed back directly into the accumulated
   state of the integrator (rather than only being subtracted once,
   downstream, on the command sent to the DM), which is what keeps that
   correction from simply being re-added by the integrator on the next
   step.

**The Soft-Limiter.** Sits between the integrator and the DM, consuming
the two Part 2 :class:`Recmat` objects:

.. code-block:: yaml

    unwrapper:
      class: 'SoftLimiter'
      recmat_list_object: ['ELT39_modes_to_petals_mmse', 'ELT39_petals_to_modes']
      gain: 0.1
      inputs:
        in_comm: 'control.out_comm'
      outputs: ['out_comm', 'out_ost']

    dm:
      class:             'DM'
      simul_params_ref:  'main'
      ifunc_object:      'ELT39_KL4000'
      nmodes:            4000
      height:            0
      inputs:
        in_command: 'unwrapper.out_comm'
      outputs: ['out_layer']

``recmat_list_object`` must list the MMSE reconstructor first and the
interaction matrix second: :class:`SoftLimiter` uses the first to project
the current command onto a petal-piston estimate, and the second to
project the resulting leak correction back into mode space. On every step
it computes ``est = recmat @ in_comm``, ``delta = intmat @ (gain *
est)``, and outputs ``out_comm = in_comm - delta`` (what reaches the DM)
together with ``out_ost = delta`` (fed back into ``control`` above).
Setting ``gain: 0.0`` disables the leak entirely without changing anything
else in the loop, which is how the Classical comparison run in Step 5 is
built.

**Ground-truth petal diagnostic.** Independent of anything the loop
itself estimates, and using the differential 5-petal inverse built in
Part 1 specifically to be immune to global piston contamination:

.. code-block:: yaml

    petal_analysis:
      class: 'ModalAnalysis'
      ifunc_inv_object: 'ELT39_6petals_inv'
      nmodes: 5
      inputs:
        in_ef: 'prop.out_wfs_source_ef'
      outputs: ['out_modes']

This reads the true residual electric field directly, not the noisy,
delayed estimate the loop itself produces, and is what the analysis in
Step 6 below is computed from. Using anything other than the corrected,
piston-differential inverse here would silently reintroduce the bug
documented in the warning in Part 1, so keep that in mind if you build a
similar diagnostic for a different configuration.

**PSF and telemetry.** A :class:`PSF` object provides the Strehl ratio at
2200 nm, and :class:`DataStore` saves the two time series needed for the
analysis below:

.. code-block:: yaml

    psf:
      class:             'PSF'
      simul_params_ref:  'main'
      wavelengthInNm:    2200
      nd:                3
      start_time:        0.05
      inputs:
        in_ef: 'prop.out_wfs_source_ef'
      outputs: ['out_psf', 'out_sr']

    data_store:
      class:             'DataStore'
      store_dir:         '<your output directory>'
      inputs:
        input_list: ['petal-petal_analysis.out_modes', 'sr-psf.out_sr']

Step 5: Running Classical vs Soft-Limiter
------------------------------------------

Run the same 30 s, single-seed closed loop twice, with one override file
each. Each file also redirects ``data_store`` to its own directory --
without that, both runs would write their timestamped output folder under
the same ``store_dir``, and the analysis in Step 6 below would not find
separate ``output_classical``/``output_softlimiter`` trees to compare:

.. code-block:: yaml

    # override_classical.yml
    unwrapper_override:
      gain: 0.0

    data_store_override:
      store_dir: 'output_classical'

.. code-block:: yaml

    # override_softlimiter.yml
    unwrapper_override:
      gain: 0.1

    data_store_override:
      store_dir: 'output_softlimiter'

.. code-block:: bash

    python -m specula.scripts.specula_main params_closed_loop.yml override_classical.yml --target <gpu_idx>
    python -m specula.scripts.specula_main params_closed_loop.yml override_softlimiter.yml --target <gpu_idx>

Everything else (atmosphere seed, WFS noise, control gains) is identical
between the two runs; the leak gain is the only thing that changes.

Step 6: Reading the result
------------------------------

The petal-piston ambiguity is naturally expressed in units of the WFS
sensing wavelength: define the fringe order :math:`k = \mathrm{round}(p /
\lambda_{\rm wfs})` for the estimate :math:`p` of each petal from
``petal_analysis``. A loop that is tracking well stays at a small,
constant :math:`k`; a jump in :math:`k` from one step to the next is
exactly the ambiguity the Soft-Limiter is meant to keep in check.

A small script computes this, plus the Strehl ratio statistics, directly
from the ``petal.fits``/``sr.fits`` telemetry saved by ``data_store``:

.. code-block:: python

    import numpy as np
    from astropy.io import fits

    lambda_wfs_nm = 800.0
    time_step = 0.002       # matches params_closed_loop.yml
    skip_transient_s = 1.0  # discard the initial convergence transient

    def analyze(output_dir):
        petal = fits.getdata(f'{output_dir}/petal.fits')  # (n_steps, 5), nm
        sr = fits.getdata(f'{output_dir}/sr.fits')         # (n_steps,)
        n_skip = int(skip_transient_s / time_step)

        k = np.round(petal / lambda_wfs_nm)
        jumps = np.diff(k, axis=0) != 0
        duration_s = petal.shape[0] * time_step
        jump_rate = jumps.sum() / duration_s  # all 5 petals combined

        sr_steady = sr[n_skip:]
        print(f'{output_dir}:')
        print(f'  max|k|          = {np.max(np.abs(k)):.0f}')
        print(f'  jump rate       = {jump_rate:.1f} jumps/s (all petals)')
        print(f'  Strehl mean/min = {sr_steady.mean():.3f} / {sr_steady.min():.3f}')

    analyze('output_classical/<timestamp>')
    analyze('output_softlimiter/<timestamp>')

Running this on the two telemetry sets from Step 5
(:math:`\lambda_{\rm wfs}` = 800 nm, 30 s at a 500 Hz frame rate):

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Metric
     - Classical (leak gain 0)
     - Soft-Limiter (nominal gain)
   * - Max :math:`|k|` reached (any petal)
     - 3
     - 2
   * - Total fringe jumps, all 5 petals, per second
     - highest of the two runs, by a wide margin
     - clearly reduced, roughly 2/3 of the Classical rate
   * - Strehl ratio, mean (excluding start-up transient)
     - low, and with large run-to-run scatter
     - roughly double the Classical mean, and much steadier
   * - Strehl ratio, worst moments
     - repeatedly collapses to near zero
     - never collapses, stays well clear of zero throughout

The pattern is the qualitative one the paper describes: Classical spends
much of the run at higher fringe orders and pays for it with an unstable,
frequently-collapsing Strehl ratio, while the Soft-Limiter leak keeps the
ambiguity confined to a smaller range and the image quality both higher
and far steadier. Treat the specific numbers above as what the generic
pupil, seed and seeing used here produce, not as a reproduction of the
ANDES-specific figures in the paper (real, as-designed pupil and DM).

Two configuration choices matter more than they might look if adapting
this to your own case: the spider width from Part 1 (thin spiders
understate how strongly petal jumps couple into the WFS signal, making
Classical look better than it should), and the per-mode-group temporal
filter of Step 4 (a flat gain never converges cleanly at this scale,
making *both* runs look equally bad).

Summary
----------

Starting from the products of Part 1 and Part 2, this tutorial:

* calibrated the pupil geometry of a pyramid WFS and a classical
  interaction/reconstruction matrix pair on the shared ELT-class pupil
* built a full closed loop with a per-mode-group temporal filter, driven
  entirely by public parameters and the products of Parts 1-2
* wired the :class:`SoftLimiter` between the integrator and the DM using
  the MMSE petal reconstructor from Part 2
* used the differential petal inverse from Part 1 as an independent
  ground-truth diagnostic
* compared a Classical loop against the Soft-Limiter loop on identical
  turbulence, and showed the Soft-Limiter confining the fringe-jump
  ambiguity to a smaller range with a correspondingly higher and steadier
  Strehl ratio

This completes the three-part series: an ELT-class segmented pupil and
modal basis built from scratch, an MMSE petal reconstructor built and
validated on top of it, and a full closed-loop demonstration of the effect
of the Soft-Limiter, all without depending on any project-specific
calibration archive.
