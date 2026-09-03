.. _ef_replay_tutorial:

EfReplay Tutorial: Exact Replay of Existing Electric Field / Layer Outputs
===========================================================================

This tutorial demonstrates how to use SPECULA's ``EfReplay`` to recompute one
or more **existing** ``ElectricField``/``Layer`` outputs from a past
simulation run exactly, without re-running the whole simulation.

``EfReplay`` complements :ref:`field_analyser_tutorial`: where ``FieldAnalyser``
synthesizes **new** off-axis sources and attaches them to an
``AtmoPropagation`` object, ``EfReplay`` instead targets object(s) that already
exist in the original ``params.yml`` — e.g. an ``ElectricFieldCombinator`` that
sums a propagated source with a disturbance before a WFS — and replays exactly
what was computed there, including anything feeding into it.

**Goals:**

- Understand when to use ``EfReplay`` instead of ``FieldAnalyser``

- Learn how to specify which existing outputs to replay

- Recompute those outputs from saved DM commands, bypassing the WFS chain
  where possible

**Prerequisites:**

- You have already run a simulation and have a data directory with results
  (see :ref:`scao_basic_tutorial` for running a simulation)

- The output directory contains ``params.yml`` and the necessary replay data
  (see :ref:`field_analyser_tutorial`, Step 1, for what to save)

When to Use ``EfReplay`` Instead of ``FieldAnalyser``
------------------------------------------------------

``FieldAnalyser`` reconstructs off-axis phase/PSF by attaching **new** field
points directly to your ``AtmoPropagation`` object. It only replays what feeds
*into* that object (atmosphere and captured DM commands) — anything summed
onto a source's electric field **downstream** of the propagation object (for
example a disturbance injected via ``ElectricFieldCombinator``, such as a
``PhaseScreenCube``) is invisible to it, because it was never one of that
object's inputs. See the "Limitation" section of :ref:`field_analyser_tutorial`
for the full explanation of why this can bias off-axis results.

``EfReplay`` does not have this limitation, but it also cannot do what
``FieldAnalyser`` does: it only reproduces outputs of objects that **already
exist** in ``params.yml``, for the exact same directions/lines of sight used
in the original simulation — it does not synthesize anything new. Use it when
you want the exact field/layer an existing sensor or combinator saw, not a
new off-axis point:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Need
     - Use
     - Why
   * - PSF/phase at a **new** off-axis point
     - ``FieldAnalyser``
     - Synthesizes a new ``Source`` attached to ``AtmoPropagation``
   * - Exact field as seen by an **existing** WFS/combinator
     - ``EfReplay``
     - Targets the existing object directly; includes anything feeding it,
       e.g. an injected disturbance
   * - Both, in the same field point
     - Neither alone
     - An injected disturbance has no direction/height information to
       re-project onto a new point (no general solution)

Step 1: Configuring Your Simulation
--------------------------------------

The same guidance as :ref:`field_analyser_tutorial` Step 1 applies: save DM
commands (and any other signal you may want to target directly) in your
``DataStore``, so replay does not need to re-run the WFS chain. No additional
configuration is required specifically for ``EfReplay`` — it works directly
from ``params.yml`` and ``replay_params.yml``, just like ``FieldAnalyser``.

Step 2: Using EfReplay in Python
------------------------------------

.. code-block:: python

    import os
    import glob
    import specula
    specula.init(0)

    from specula.ef_replay import EfReplay

    # Find the latest data directory (assuming output is in ./data)
    data_dirs = sorted(glob.glob("data/2*"))
    latest_data_dir = data_dirs[-1]

    # Replay the exact field seen by an ElectricFieldCombinator that sums
    # an AtmoPropagation source with an injected disturbance, plus the
    # disturbance-cancelling DM's own surface, in one call
    replay = EfReplay(
        data_dir="data",
        tracking_number=os.path.basename(latest_data_dir),
        output_refs=[
            'ef_combinator13.out_ef',   # what the WFS actually saw
            'dm_foc_lift.out_layer',    # the injected calibration disturbance
        ],
        start_time=0.0,
    )

    results = replay.compute_replay(force_recompute=True)

    ef_combinator_field = results['ef_combinator13.out_ef']['data']
    injected_disturbance = results['dm_foc_lift.out_layer']['data']

``output_refs`` is a list of ``object_name.output_name`` strings, exactly like
a ``DataStore`` ``input_list`` entry without the filename prefix. Every unique
object name referenced this way becomes a target for
``Simul.build_targeted_replay``, which pulls in everything it depends on
recursively (atmosphere, captured DM commands, and any other object needed to
reproduce that output) — this is what makes the injected disturbance visible
in the example above: targeting ``ef_combinator13`` directly (instead of only
``prop``) also pulls in whatever feeds that combinator.

**Result format:** ``compute_replay()`` returns a dict keyed by each entry in
``output_refs``, each value a dict with ``'data'`` (the FITS primary array)
and ``'times'`` (the saved time vector, or ``None`` if not available).

Step 3: Caveats
------------------

- **Same direction only.** ``EfReplay`` cannot answer "what would this
  disturbance look like at a *different* field point" — it replays the
  object exactly as configured in ``params.yml``, which is tied to whatever
  direction it was originally wired to.
- **Non-deterministic upstream objects.** If reproducing the requested
  output requires reconstructing an object with unseeded randomness (e.g. a
  ``RandomGenerator`` used for WFS-arm jitter with no explicit ``seed``, or
  detector photon/readout noise), the replayed values will not be bit-exact
  with the original run, since that randomness was not recorded. Objects
  whose command was already captured by the ``DataStore`` (e.g. a DM's
  input) are unaffected, since the replay uses the recorded command
  directly rather than regenerating it.
- **Missing objects raise clearly.** If any name in ``output_refs`` is not
  present in ``params.yml``, ``EfReplay`` raises ``KeyError`` immediately
  rather than silently producing partial results.

.. seealso::

   - :ref:`field_analyser_tutorial` for off-axis PSF/phase at new field points,
     and for the "Limitation" section this tutorial complements
   - `build_targeted_replay <../specula/simul.py>`_ for the underlying replay
     mechanism shared by both classes
