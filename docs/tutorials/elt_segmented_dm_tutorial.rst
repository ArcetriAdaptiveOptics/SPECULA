.. _elt_segmented_dm_tutorial:

Building an ELT-Class Segmented Pupil and Modal Basis
========================================================

This tutorial shows how to build, entirely from public parameters and the
calibration tools built into SPECULA, the pupil mask and modal basis for an
Extremely Large Telescope (ELT)-class system whose primary mirror is
supported by several independent structures ("petals"). It is a
self-contained, reusable prerequisite: any tutorial or simulation that
needs an ELT-class segmented pupil and DM (for example the
:ref:`segmented_pupil_soft_limiter_tutorial`) can build on the products
generated here.

**What you'll learn:**

* Why a segmented pupil needs its own influence-function/modal-basis
  generation step, and how to build one from scratch
* Computing a shared pupil mask once and reusing it consistently across
  every influence-function/modal-basis product (and why that matters)
* Generating petal influence functions with :func:`compute_petal_ifunc`
* Generating a zonal DM influence-function set and a KL modal basis with
  :func:`compute_zonal_ifunc` and :func:`make_modal_base_from_ifs_fft`
* Saving everything with the :class:`CalibManager`, ready to be referenced
  by tag from a simulation YAML file

**Prerequisites:**

* SPECULA installed and working (see :doc:`../installation`)
* Basic understanding of adaptive optics concepts (influence functions,
  modal bases, wavefront sensing)
* Python familiarity
* Time and a real machine (see the warning below before you start)

.. warning::

   **This is not a quick tutorial to run.** "ELT-class" means a large
   pupil and thousands of actuators, and there is no small/fast version of
   this that still means anything (see the
   :ref:`note on modal bandwidth <elt_modal_bandwidth_note>` below). On a
   single CPU core,
   for the configuration used here (400x400 pixels, 90-actuator grid, 4000
   modes), expect the whole pipeline to take on the order of **an hour**,
   with :func:`compute_zonal_ifunc` (Step 3) responsible for most of it.
   Run it as a background job, and save the result (as done here) so you
   only pay this cost once.

   A GPU index passed to ``specula.init()`` (instead of ``-1``) cuts this
   down substantially. Every function here accepts an ``xp`` module, so
   treat "an hour" as the pessimistic, CPU-only baseline.

   To just confirm the code runs on your machine before committing to the
   full run, see :ref:`elt_dm_smoke_test`, a deliberately tiny,
   non-representative configuration for that purpose only.

Everything used here (pupil mask, influence functions, modal basis) is
generated in Python from public, generic parameters; nothing is loaded
from an external, project-specific calibration archive.

Why not just use an existing pupil?
------------------------------------

If you already have the real, as-designed ELT pupil and M4 influence
functions for your project, **skip this tutorial**. Everything generated
here is only a didactic stand-in for those two products, and Parts 2-3
just need *a* pupil mask, petal influence functions, a DM
influence-function set and a modal basis, however you obtained them.
Point Parts 2-3 at your own products (saved under matching
:class:`CalibManager` tags) and carry on from there.

If you don't have them yet, the generic generators used here are a
reasonable stand-in, with three known limits. None of them matter for the
petal-piston control problem this series addresses, but they do mean this
pupil/DM should not be mistaken for the real thing:

* "Segmented" here means six large **petal** sectors, one per independent
  support structure. The mask does not reproduce the individual hexagonal
  M1 segments and their gaps within each petal: irrelevant to
  petal-piston control, but visibly different from the real pupil.
* The spider width (see the Design choices table below) matches the real
  ELT spider thickness *on average*, but is uniform across all six arms
  here; on the real telescope at least one arm is thicker than the others.
* The generated M4-like influence functions sit on a plain circular
  actuator grid with edge slaving. The real M4 geometry instead repeats
  per petal, with dedicated rows of actuators running along each petal
  boundary, a layout this tutorial does not impose.

Design choices
---------------

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Parameter
     - Value
     - Notes
   * - Telescope diameter
     - 39 m
     - ELT-class
   * - Pupil sampling
     - 400 x 400 pixels
     - matches the resolution used in ELT-class SCAO studies
   * - Central obstruction
     - 28%
     - representative of an ELT-class M2/M5 obstruction
   * - Number of petals
     - 6
     - one per independent primary-mirror support structure
   * - Spider
     - 3 pixels wide
     - matches the real ELT spider thickness at this sampling, see note below
   * - Actuator-to-actuator coupling
     - none (``do_mech_coupling=False``)
     - see note below: M4 is not a stacked-actuator DM
   * - Edge actuator handling
     - linear (piston+tip+tilt) slaving
     - smoother edge extrapolation than plain weighted-average slaving
   * - Modal basis size
     - generate several thousand, keep 4000
     - a choice made for this tutorial, see note below

.. note::

   **Why no mechanical coupling.** ``do_mech_coupling`` models the
   nearest/next-nearest-neighbor print-through of a stacked-piezo DM. An
   M4-type mirror has internal metrology that actively imposes the
   commanded displacement on each actuator, eliminating that coupling.

.. note::

   **Why a 3-pixel spider.** The real ELT spiders are roughly 310 mm
   thick; at this sampling (39 m / 400 px ≈ 0.0975 m/pixel) that is close
   to 3 pixels. This is a uniform average, though; see the caveats above
   for how the real spider differs.

.. _elt_modal_bandwidth_note:

.. note::

   **Why 4000 modes, and why that count cannot be shrunk down.** No fixed
   rule sets the number itself: use anywhere up to the full generated
   count, and change ``n_modes_to_use`` freely for a different case. What
   does matter is having *enough*: the petal-piston pattern Part 2 and
   Part 3 need to reconstruct is a sharp, sector-wise step, and
   representing a step in any smooth modal basis (KL, Zernike, ...) takes
   many modes, the same way a Fourier series needs many harmonics to
   approximate one. A basis generated from a handful of actuators (as in
   the smoke test below) is missing essentially all of that bandwidth, so
   any reconstruction accuracy it produces is not a scaled-down preview of
   the full-size result -- it is a different, much worse regime.

Step 1: A shared pupil mask
-----------------------------

The single most important design decision in this tutorial is to compute
the pupil mask **once**, and pass that exact array to every subsequent
influence-function generator. If instead each generator were left to build
its own mask independently, even a tiny difference in how each one
rasterizes the aperture edge would leave the petal influence functions and
the DM (zonal/KL) influence functions defined over subtly different pixel
sets. Any later step that mixes information from both bases (e.g. a
petal-to-mode reconstruction matrix) would then silently be misaligned.
That kind of bug is very difficult to track down after the fact, because
every individual piece still looks reasonable in isolation.

We get the shared mask directly from :func:`compute_petal_ifunc`, since
that call also gives us the petal influence functions we need:

.. code-block:: python

    import specula
    specula.init(-1)  # -1 selects the CPU; use a GPU index if you have one

    import numpy as np
    import matplotlib.pyplot as plt

    from specula.lib.compute_petal_ifunc import compute_petal_ifunc
    from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
    from specula.lib.modal_base_generator import make_modal_base_from_ifs_fft
    from specula.data_objects.ifunc import IFunc
    from specula.data_objects.m2c import M2C
    from specula.data_objects.pupilstop import Pupilstop
    from specula.data_objects.simul_params import SimulParams
    from specula.calib_manager import CalibManager
    from specula import cpuArray

    # --- Physical configuration ---
    telescope_diameter = 39.0    # meters, ELT-class
    obsratio = 0.28              # central obstruction, ELT-class
    n_petals = 6                 # one per M4-like support structure
    angle_offset = 0.0           # degrees

    # --- Resolution / actuator grid -- see the warning above for the cost ---
    pixel_pupil = 400
    n_act = 90

    dtype = specula.xp.float32

    petal_ifunc, pupil_mask, _ = compute_petal_ifunc(
        pixel_pupil, n_petals, xp=specula.xp, dtype=dtype,
        angle_offset=angle_offset, obsratio=obsratio, diaratio=1.0,
        mask=None, spider=True, spider_width=3,
        add_tilts=False, special_last_petal=False)

    print(f'Petal influence functions: {petal_ifunc.shape} '
          f'(6 petals x {petal_ifunc.shape[1]} valid pixels)')
    print(f'Valid pixels in shared mask: {int(specula.xp.sum(pupil_mask))} '
          f'/ {pixel_pupil**2}')

    plt.figure(figsize=(5, 5))
    plt.imshow(cpuArray(pupil_mask), cmap='gray')
    plt.title(f'Shared pupil mask ({n_petals} petals, {pixel_pupil}px)')
    plt.colorbar(label='mask value')
    plt.show()

This step takes well under a second. With a 28% central obstruction,
expect somewhat more than two thirds of the 400x400 grid to come out as
valid pixels.

``pupil_mask`` is the array we will now pass, unchanged, to every other step.

Step 2: Saving the Pupilstop and petal basis
-----------------------------------------------

Before moving on, save the pupil as a :class:`Pupilstop` object: this is
what a SPECULA simulation YAML file references (via ``pupilstop_object``)
to define the telescope aperture.

.. code-block:: python

    import os

    calib = CalibManager('./calib_elt_segmented_dm_tutorial')

    # CalibManager resolves tags to paths but does not create directories;
    # a YAML-driven calibrator (as in Part 3) does this for you, but a
    # direct .save() call like the ones below does not.
    for subdir in ['pupilstop', 'ifunc', 'm2c', 'rec']:
        os.makedirs(calib.root_subdir(subdir), exist_ok=True)

    simul_params = SimulParams(pixel_pupil=pixel_pupil,
                                pixel_pitch=telescope_diameter / pixel_pupil)

    pupilstop_obj = Pupilstop(simul_params, input_mask=cpuArray(pupil_mask))
    pupilstop_filename = calib.filename('pupilstop', 'ELT39_6petals')
    pupilstop_obj.save(pupilstop_filename, overwrite=True)
    print(f'Saved: {pupilstop_filename}')

    petal_ifunc_obj = IFunc(ifunc=petal_ifunc, mask=pupil_mask)
    petal_ifunc_filename = calib.filename('ifunc', 'ELT39_6petals')
    petal_ifunc_obj.save(petal_ifunc_filename, overwrite=True)
    print(f'Saved: {petal_ifunc_filename}')

Using :class:`CalibManager` (rather than hand-built paths) means these
products can be referenced later purely by tag, the same way a simulation
YAML file resolves ``ifunc_object`` or ``pupilstop_object`` tags under a
shared ``root_dir``; see :ref:`calibration_manager` if this is new to you.

Optional: ground-truth and alternative petal bases
------------------------------------------------------

The two subsections below are not needed to reach Part 2: they build extra
products on top of the petal basis above, for diagnostics and for sensors
that cannot use it directly. Skip ahead to Step 3 on a first read if you
only need the main pipeline.

A differential petal inverse, for ground-truth diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A closed-loop simulation typically wants a ground-truth readout of the
current petal-piston state directly from the electric field (e.g. via
:class:`ModalAnalysis`), independent of the MMSE reconstructor built in
:ref:`elt_petal_mmse_reconstructor_tutorial`. That needs an inverse of the
petal basis, and it's easy to get wrong in a way that still looks
plausible at first glance:

.. code-block:: python

    # 5 rows only (petal 6 stays the reference), NOT all 6:
    petal_ifunc_5_obj = IFunc(ifunc=petal_ifunc[:n_petals - 1, :], mask=pupil_mask)
    petal_inv_obj = petal_ifunc_5_obj.inverse()   # remove_piston=True by default
    petal_inv_filename = calib.filename('ifunc', 'ELT39_6petals_inv')
    petal_inv_obj.save(petal_inv_filename, overwrite=True)
    print(f'Saved: {petal_inv_filename}')

.. warning::

   **The obvious-looking alternative is wrong.** Inverting all *6* raw
   petal rows at once and keeping only the first 5 columns (e.g. via
   ``nmodes=5`` in :class:`ModalAnalysis`) gives each petal's *absolute*
   mean phase, not its *differential* piston relative to petal 6, so it
   tracks any true global piston almost one-for-one (measured: a 2000 nm
   global offset leaked out at slope 0.99). Dropping the sixth row
   *before* inverting, as done above, avoids this: the default
   ``remove_piston=True`` of :func:`IFunc.inverse` centers each of the 5
   remaining rows, making them orthogonal to the global-piston pattern
   (the same 2000 nm offset then leaks out at slope ~0.01).

Theta modes: a smooth alternative to the petal basis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every petal mode above is a disjoint, piecewise-constant sector: ideal for
the influence-function machinery used throughout this tutorial, but not a
shape every sensor can work with. A focal-plane modal sensor restricted to
smooth, globally-defined modes (e.g. LIFT) cannot sense a discontinuous
sector directly, but it can sense a smooth phase ramp. The same
petal-piston information can be recovered from six such ramps instead, one
referenced to each spider -- informally a "garage-ramp" mode, for its
one-sided linear slope:

.. code-block:: python

    # Same (y, x, theta) convention as compute_petal_ifunc, but a
    # continuous ramp referenced to each spider instead of a disjoint
    # sector indicator.
    center = pixel_pupil / 2
    y, x = specula.xp.mgrid[:pixel_pupil, :pixel_pupil]
    y = y - center
    x = x - center
    theta_full = specula.xp.arctan2(y, x) + specula.xp.radians(angle_offset)
    petal_angle = 2 * specula.xp.pi / n_petals
    idx = specula.xp.where(pupil_mask)

    def wrap_pi(angle):
        return (angle + specula.xp.pi) % (2 * specula.xp.pi) - specula.xp.pi

    theta_modes = specula.xp.stack([
        wrap_pi(theta_full - k * petal_angle - specula.xp.radians(angle_offset))[idx]
        for k in range(n_petals)
    ]).astype(dtype)
    print(f'Theta modes: {theta_modes.shape} (6 ramps x {theta_modes.shape[1]} valid pixels)')

Differencing two ramps referenced at *adjacent* spiders (0 and 1) gives a
clean, two-level step function -- but note which sector it isolates:

.. code-block:: python

    diff = np.round(cpuArray(theta_modes[0] - theta_modes[1]), 4)
    levels = np.unique(diff)
    print(f'theta[0] - theta[1] takes {len(levels)} distinct values: {levels}')
    assert len(levels) == 2 and abs(abs(levels[1] - levels[0]) - 2 * np.pi) < 1e-3

    antipodal_petal = (0 + n_petals // 2) % n_petals  # = 3, not 0
    low_region = (diff == levels.min()).astype(np.float32)
    assert np.allclose(low_region, cpuArray(petal_ifunc[antipodal_petal]))
    print(f'Confirmed: theta[0] - theta[1] exactly reproduces petal_ifunc[{antipodal_petal}].')

::

    Theta modes: (6, 111398) (6 ramps x 111398 valid pixels)
    theta[0] - theta[1] takes 2 distinct values: [-5.236   1.0472]
    Confirmed: theta[0] - theta[1] exactly reproduces petal_ifunc[3].

Each ramp's single discontinuity sits *opposite* its own reference spider,
not on it, so differencing two adjacently-referenced ramps isolates the
sector **antipodal** to that pair, not the one between them -- a real trap
if you assume otherwise. To isolate petal :math:`p` directly, reference
the pair ``(p + n_petals // 2) % n_petals`` and the next index instead.
None of this is specific to :math:`p=0`: the same ``n_petals // 2`` shift
applies for any pair of adjacent reference spiders.

.. note::

   **Why theta needs all 6 modes, while the petal basis only needs 5.**
   Both bases are full rank as raw vectors (6 independent rows each), and
   they share exactly the same 5-dimensional space of relative
   petal-piston information: reconstructing the 5 relative-petal targets
   (``petal_ifunc[:5] - petal_ifunc[5]``) from all 6 theta modes is exact
   to machine precision. Where they differ is the 6th, non-shared
   dimension. For petals it is the global-piston pattern: irrelevant to a
   closed loop, and conveniently isolated in whichever single row you
   call the reference, so dropping it (as Step 2 above does, inverting
   only 5 of the 6 petal rows) costs nothing.

   For theta the 6th dimension is something else: it cannot represent a
   constant (global-piston) pattern at all (reconstructing one from all 6
   ramps leaves ~100% relative error), and it is the *same* leftover
   pattern in every one of the 6 ramps -- after removing the shared
   5-dimensional component, each row's residual points in exactly the
   same direction, to machine precision, rather than being isolated in
   any single row. That is what makes it non-droppable: discarding one
   ramp (e.g. keeping only ``theta_modes[:5]``) does not cleanly remove
   that extra direction, it also removes 2 of the 5 relative-petal
   degrees of freedom along with
   it (measured: ~0.34 relative reconstruction error on exactly 2 of the
   5 targets, versus ~1e-15 using all 6). So 5 modes suffice for petals
   because their discardable direction is both irrelevant *and* cleanly
   separable; theta keeps all 6 because its extra direction is equally
   irrelevant but not separable from any single row.

Saving this basis follows the same pattern as the petal basis above:

.. code-block:: python

    theta_ifunc_obj = IFunc(ifunc=theta_modes, mask=pupil_mask)
    theta_ifunc_filename = calib.filename('ifunc', 'ELT39_6theta')
    theta_ifunc_obj.save(theta_ifunc_filename, overwrite=True)
    print(f'Saved: {theta_ifunc_filename}')

This is only the forward basis. Turning it into an estimator for a real
focal-plane sensor needs a dedicated reconstructor built around that
sensor's own response to each ramp, which is outside the scope of this
series (built instead around a pyramid WFS and the MMSE/Soft-Limiter
architecture of :ref:`elt_petal_mmse_reconstructor_tutorial` and
:ref:`elt_petal_soft_limiter_closed_loop_tutorial`); it is saved here only
as a documented starting point for that separate exercise.

Which basis to use depends on what does the sensing, not on the
disturbance itself. Under pure atmospheric turbulence, this series'
architecture already estimates petal state indirectly from the
accumulated KL-mode command of the main pyramid-WFS loop -- a smooth,
continuous quantity with no modal-bandwidth restriction, so the disjoint
petal basis is perfectly adequate. :ref:`elt_petal_soft_limiter_closed_loop_tutorial`
demonstrates exactly that, with no theta modes involved at all. A separate,
dedicated sensor with a genuinely limited, smooth modal bandwidth -- a
focal-plane sensor such as LIFT, often added specifically because the main
WFS loop is comparatively blind to petal piston under Low Wind Effect --
cannot represent the disjoint petal shapes directly, and is where a basis
like these theta modes becomes the natural choice instead.

Step 3: Zonal DM influence functions, on the *same* mask
------------------------------------------------------------

Now we generate the deformable-mirror influence functions: a much larger
set, one per actuator, used as the basis for the KL modal decomposition in
Step 4. Note that ``mask=pupil_mask`` reuses the exact array from Step 1,
rather than letting :func:`compute_zonal_ifunc` build its own. **This is
the expensive step; see the warning at the top of this page.**

.. code-block:: python

    zonal_ifunc, pupil_mask_check, _, _ = compute_zonal_ifunc(
        pixel_pupil, n_act, xp=specula.xp, dtype=dtype,
        circ_geom=True, angle_offset=angle_offset,
        do_mech_coupling=False,
        do_slaving=True, slaving_thr=0.1, linear_slaving=True,
        obsratio=obsratio, diaratio=1.0, mask=pupil_mask)

    assert np.array_equal(cpuArray(pupil_mask_check), cpuArray(pupil_mask)), \
        "mask mismatch -- the zonal ifunc is not on the same pixel grid!"

    print(f'Zonal influence functions: {zonal_ifunc.shape[0]} valid actuators '
          f'(after slaving) x {zonal_ifunc.shape[1]} pixels')

We leave mechanical coupling disabled (see the design-choices note above)
and enable ``do_slaving`` with ``linear_slaving=True``: each weakly-coupled
edge actuator is extrapolated from a local piston+tip+tilt plane fit to
nearby masters, rather than a flat weighted average. The explicit
``assert`` is the check that would catch the shared-mask problem described
above, had one crept in.

This is the slow step (most of the "about an hour" from the warning at the
top). With ``n_act=90``, expect several thousand valid actuators after
slaving, a small fraction of them slaved at the edge and the rest masters.

Step 4: A KL modal basis, generating more modes than we need
------------------------------------------------------------------

:func:`make_modal_base_from_ifs_fft` turns the zonal influence functions
into a Karhunen-Loeve-like modal basis, ranked by the turbulence power they
capture:

.. code-block:: python

    kl_basis, m2c, singular_values = make_modal_base_from_ifs_fft(
        pupil_mask=pupil_mask, diameter=telescope_diameter,
        influence_functions=zonal_ifunc,
        r0=0.15, L0=25.0,          # only used to *weight* the basis
        zern_modes=3, oversampling=2,
        if_max_condition_number=None,
        xp=specula.xp, dtype=dtype)

    print(f'KL basis: {kl_basis.shape[0]} modes x {kl_basis.shape[1]} pixels')

``r0``/``L0`` here only shape which spatial frequencies the basis
prioritizes; they are independent of whatever atmosphere you eventually
simulate in closed loop.

This is comparatively cheap next to Step 3 (a few minutes rather than most
of an hour) and produces one mode fewer than the actuator count (the
global piston is not a controllable DM mode).

You can use anywhere up to the full count of generated modes, keeping as
many as you need without re-running Step 3:

.. code-block:: python

    n_modes_total = kl_basis.shape[0]              # before truncation
    n_modes_to_use = min(4000, n_modes_total)      # see note on this choice above
    kl_basis = kl_basis[:n_modes_to_use]
    m2c = m2c[:, :n_modes_to_use]
    print(f'Using the first {n_modes_to_use} of {n_modes_total} generated modes')

With ``n_act=90`` there are more than 4000 modes available, so this line
does real work, keeping the first 4000 and dropping the rest.

Saving the modal basis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    m2c_obj = M2C(m2c=m2c)
    m2c_filename = calib.filename('m2c', 'ELT39_KL4000')
    m2c_obj.save(m2c_filename, overwrite=True)
    print(f'Saved: {m2c_filename}')

    zonal_ifunc_obj = IFunc(ifunc=zonal_ifunc, mask=pupil_mask)
    zonal_ifunc_filename = calib.filename('ifunc', 'ELT39_zonal')
    zonal_ifunc_obj.save(zonal_ifunc_filename, overwrite=True)
    print(f'Saved: {zonal_ifunc_filename}')

    # Also save the forward KL basis itself as an IFunc (not just its
    # inverse): building a petal<->KL reconstructor in the next tutorial
    # needs the forward basis to compute a turbulence covariance matrix in
    # mode space.
    kl_ifunc_obj = IFunc(ifunc=cpuArray(kl_basis), mask=cpuArray(pupil_mask))
    kl_ifunc_filename = calib.filename('ifunc', 'ELT39_KL4000')
    kl_ifunc_obj.save(kl_ifunc_filename, overwrite=True)
    print(f'Saved: {kl_ifunc_filename}')

    # IFunc.inverse() computes the pseudo-inverse and hands back a
    # ready-made IFuncInv.
    ifunc_inv_obj = kl_ifunc_obj.inverse()
    ifunc_inv_filename = calib.filename('ifunc', 'ELT39_KL4000_inv')
    ifunc_inv_obj.save(ifunc_inv_filename, overwrite=True)
    print(f'Saved: {ifunc_inv_filename}')

This last step, a pseudo-inverse of a (4000 modes, many pixels) matrix,
adds a further few minutes. ``IFunc.inverse()`` computes it via the smaller
of the two Gram matrices (:func:`specula.lib.fast_pinv.fast_pinv`) rather
than an SVD of the full rectangular matrix, which is what makes this
tractable at this scale.

Sanity checks and visualization
----------------------------------

Worth running every time you regenerate a basis, since a silent mistake
here (e.g. the shared-mask issue above) would otherwise only surface much
later, deep inside a closed-loop simulation:

.. code-block:: python

    # Mode RMS should be well-behaved (non-zero, no NaNs) for every mode
    rms = np.sqrt(np.mean(cpuArray(kl_basis)**2, axis=1))
    print(f'Mode RMS: min={rms.min():.3g}, max={rms.max():.3g}, '
          f'any NaN: {np.any(np.isnan(rms))}')

    # Singular value spectrum: should decay smoothly, no discontinuities
    plt.figure(figsize=(8, 5))
    plt.semilogy(cpuArray(singular_values['S1']), 'o-', label='IF covariance')
    plt.semilogy(cpuArray(singular_values['S2']), 'o-', label='Turbulence covariance')
    plt.xlabel('Mode number')
    plt.ylabel('Singular value')
    plt.legend()
    plt.grid(True)
    plt.title('Singular value spectrum')
    plt.show()

    # A handful of KL modes, reshaped onto the pupil, for a visual check
    kl_np = cpuArray(kl_basis)
    mask_np = cpuArray(pupil_mask)
    idx_mask = np.where(mask_np)
    n_show = min(9, kl_np.shape[0])
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for i, ax in enumerate(axes.flat[:n_show]):
        mode_img = np.zeros(mask_np.shape)
        mode_img[idx_mask] = kl_np[i]
        ax.imshow(mode_img, cmap='viridis')
        ax.set_title(f'Mode {i + 1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

.. _elt_dm_smoke_test:

Smoke-testing the pipeline before committing to the full run
------------------------------------------------------------------

If you only want to check that the code above runs on your machine (no
typos, no missing dependencies, no shape mismatches) before spending close
to an hour on it, shrink the two size parameters drastically:

.. code-block:: python

    pixel_pupil = 40
    n_act = 9

This finishes in a few seconds. **Do not draw any conclusion from the
numbers it produces.** Valid-pixel counts, actuator counts, mode counts,
and (if you carry it into :ref:`elt_petal_mmse_reconstructor_tutorial`)
any reconstruction accuracy are all specific to this toy size and do not
scale down meaningfully from the full configuration. Its only job is to
confirm the pipeline executes; once it does, switch back to
``pixel_pupil = 400`` and ``n_act = 90`` and let the full run complete.

Summary and what's next
--------------------------

At this point you have, generated entirely from public parameters and
saved through the :class:`CalibManager`:

* a shared pupil mask (400x400 pixels, 39 m, 28% obstruction, 6 petals),
  saved as a :class:`Pupilstop`
* petal influence functions (:class:`IFunc`, 6 petals) on that mask, and a
  properly differential 5-petal :class:`IFuncInv` for ground-truth
  diagnostics
* a 6-mode "theta" ramp basis (:class:`IFunc`) carrying the same
  petal-piston information in a smooth, globally-defined form, for sensors
  that cannot work with the disjoint petal sectors directly
* a zonal DM influence-function set (:class:`IFunc`, several thousand
  actuators) on the *same* mask
* a 4000-mode KL modal basis, saved both as an :class:`M2C` and directly as
  an :class:`IFunc` (the forward basis), plus the corresponding
  :class:`IFuncInv` (its pseudo-inverse), all on the same shared mask

These products are the starting point for
:ref:`elt_petal_mmse_reconstructor_tutorial`, which uses the shared mask to
build a self-consistent reconstructor from KL-mode commands to petal-piston
estimates, and, further down the line, for the closed-loop simulation in
:ref:`elt_petal_soft_limiter_closed_loop_tutorial`.
