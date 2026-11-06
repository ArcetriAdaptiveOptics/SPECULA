.. _accelerator:

Accelerating objects (CUDA graphs)
==================================

In a simulation of moderate size, the time per step is often set by the host,
not by the GPU: each processing object runs Python code that launches its
kernels one by one, and some objects wait for the GPU at each step. The accelerator
removes most of this host work. It is part of the iteras framework, so it works
for any plugin.

It is on by default. ``--no-accelerate`` on the command line turns it off
(also if the parameter files have an ``Accelerator`` section)::

    specula --no-accelerate params.yml

``--accelerate`` is accepted, and does nothing. From Python,
``Simul(..., accelerate=False)`` turns it off. The options are given with a
section whose class is ``Accelerator`` in the parameter files:

.. code-block:: yaml

    accelerator:
      class: 'Accelerator'
      exclude: ['psf']            # objects not accelerated
      report_file: 'accelerator.txt'   # which objects were accelerated, and why not the others

It works with the :doc:`allocator <allocator>` and with :ref:`MPI
<running_with_mpi>`: each process accelerates the objects it runs, on their devices.

What it does
------------

On each GPU, with the accelerator:

* the ``trigger_code()`` of each accelerated object is captured in a CUDA graph at
  its second trigger (``capture_after`` option), and the graph is launched at
  the following triggers instead of running the Python code. Each object has
  its own graph, so an object that does not trigger at some step is not
  launched, as when the accelerator is off;
* the GPU work goes to the default stream of the device, in trigger order:
  the graphs, and the kernels launched by the code that still runs in Python.
  The GPU work therefore runs in the right order without the host waiting for
  it: ``post_trigger()`` does not wait for the graph of the object. The host
  waits only where it reads results (copies to the host for MPI, or objects
  that read GPU values). Launching the graphs of a trigger level on separate
  streams, to run them in parallel, gave no measurable gain (SCAO, SOUL, ANDES,
  MORFEO), and was removed;
* ``check_ready()``, ``prepare_trigger()`` and ``post_trigger()`` still run as
  Python code at each step. So the values computed on the host at each step,
  like the time of the generators or the time step of the atmosphere, are up
  to date: the generators write the time in a GPU array in
  ``prepare_trigger()``, which their graph reads.

The objects keep their parameters, inputs and outputs: an accelerated simulation is
the same simulation, and its results agree with those of the normal run to
rounding (different kernels are used for the matrix products, see below). The
GPU work of the objects is not changed: the accelerator removes host time, not GPU
time. Simulations whose time per step is set by the GPU gain little.

Which objects are accelerated
-----------------------------

An object is accelerated if it runs on a GPU, is not in ``exclude``, its class does
not override ``trigger()``, and its ``can_accelerate()`` method returns ``True``
when the graph is about to be captured (see :doc:`extension_api`). This method
declares that the graph will do the same as ``trigger_code()`` at all the
following steps:

* the classes that declare it after checking their ``trigger_code()``:
  ``AtmoEvolution``, ``AtmoPropagation`` (without Fresnel propagation),
  ``ModulatedPyramid`` and its derived classes (with ``cuda_stream_enable``),
  ``SH``, ``ShSlopec``, ``AVC``, ``WaveGenerator`` (the only generator), ``IirFilter`` and its derived classes
  (``Integrator``, ``LowPassFilter``, ``DynamicIirFilter``,
  ``DynamicIntegrator``: a new gain captures the graph again), ``BaseSlicer``,
  ``LinearCombination``, ``PyrSlopec`` (and ``DoubleRoofSlopec``),
  ``Modalrec``, ``ModalAnalysis``, ``PSF``, ``PsfCoronagraph``; ``BaseOperation`` once its delayed
  inputs have been produced; ``DM`` without force limitation; ``CCD`` when it
  reads out at every step, after its start time;
* the internal objects of the containers (see ``internal_objects()``): the
  slices of ``DistributedSH`` and ``DistributedExtSourcePyramid``, the
  instances of ``PolyChromSH`` and ``PolyCromPyramid``. They are listed in the
  report as ``<container>[i]``, and excluded with their container;
* the others are not accelerated, for example the other generators
  (``TimeHistoryGenerator``, ``ScheduleGenerator``, ``RandomGenerator``,
  ``PushPullGenerator``...) and ``ModalrecImplicitPolc``, which choose on the
  host what to compute at each step; ``DataStore`` (it copies its inputs to
  the host, without waiting for the GPU: they are stored at the next step); and
  the objects on the CPU.

A derived class inherits ``can_accelerate()``. If it redefines
``trigger_code()``, the declaration of its base class does not cover the new
code: it is not accelerated unless it declares ``can_accelerate()`` itself
(the accelerator checks this, and reports it). A derived class that changes
code called by ``trigger_code()`` (like ``compute_slopes()`` of the slope
computers) must also declare it again.

Objects that read GPU values on the host at each step make the host wait for
all the GPU work queued before. ``PSF`` logs the Strehl ratio of each step one
step later, from a copy made without waiting.

At the end of the run, the accelerator logs a table of the objects with their
status (``accelerated``, or why not), also written to ``report_file`` if given.

The accelerator is the only CUDA graph mechanism: with the accelerator off
(``--no-accelerate``), or for an object that is not accelerated (on the CPU,
excluded, or whose class does not declare that it can be), ``trigger_code()``
runs as Python code.

The states of an object, with the accelerator on::

    built
      |
      v
    warm-up: trigger_code() as Python code, for "capture_after" triggers
      |
      v
    waiting: can_accelerate() asked at each trigger;      not True: Python code
      |                                                    (retried next time)
      | True: capture trigger_code() in a CUDA graph ---- fails: AcceleratorError
      v
    accelerated: graph launched at each trigger --- array reallocated: AcceleratorError
      |                                         ^
      | invalidate_graph()                      | captured again at the next trigger
      +-----------------------------------------+

In the code, ``BaseProcessingObj.trigger()`` runs the step in one of two
ways: through the accelerator (``GraphRunner.trigger()``), or as Python code.

Errors
------

``can_accelerate()`` declares that ``trigger_code()`` can be captured and
replayed. If that is not true, the simulation stops with an
``AcceleratorError`` that names the object: it is a bug of the class, to be
fixed there (meanwhile, the object can be listed in ``exclude``):

* CuPy refuses, with an exception, the calls that cannot be part of a graph:
  cuBLAS, cuRAND and the copies between host and GPU. If the capture of an
  object fails, the simulation stops. A ``CCD`` with photon, excess,
  clock-induced charge or charge transfer noise, for example, says in
  ``can_accelerate()`` that it cannot be accelerated (its random numbers come
  from cuRAND or need copies from the host), and runs as Python code.
* The graph reads and writes the arrays the object had when it was captured.
  Before each launch, the accelerator checks that the arrays of the object, of
  the data objects it holds and of its inputs are still the same (same memory
  address). An array that was reallocated, for example by ``prepare_trigger()``,
  would make the graph use stale or freed memory: the simulation stops, naming
  the array. An object that replaces arrays on purpose (like the dynamic filters
  when their gain changes) calls ``invalidate_graph()``, and its graph is
  captured again. The check takes a few microseconds per object;
  ``check_pointers: false`` disables it.

What is not checked: a Python value computed in ``trigger_code()`` from
something that changes at each step (the time, a counter, a value read from
an input on the host) is frozen in the graph at its capture value. This is
what ``can_accelerate()`` declares not to happen.

Matrix products
---------------

CuPy does not allow cuBLAS during a graph capture, so ``a @ b`` cannot be
captured. ``specula.lib.graph_matmul.matmul(a, b, xp)`` is ``a @ b`` (cuBLAS)
outside a capture; inside one it uses kernels that can be captured, chosen by
the memory layout of the matrix: ``matvec()`` (one warp per row of the matrix)
and ``vecmat()`` (partial sums over chunks of rows, added in a fixed order).
Both are deterministic and, on the matrices of MORFEO and SOUL (influence
functions, mode-to-command and reconstruction matrices), as fast as cuBLAS,
without copying the transposed matrix. ``Modalrec``, ``DM``, ``ModalAnalysis``
and the slope computers use it, so their results with the accelerator on and off
differ by rounding.

With the allocator
------------------

Unless the accelerator is turned off, the allocator profiles the objects
accelerated, as they will run: their host time is the time to launch their graph.
These costs are kept in the profile file apart from those
measured with the accelerator off, and an allocation computed with the accelerator is stale
without it (and the reverse). Accelerating does not constrain the placement: each
object is accelerated on the device where it is placed.

Options
-------

See :class:`iteras.data_objects.accelerator.Accelerator`.

``exclude``
    Names of objects not to accelerate.
``capture_after``
    Triggers of each object before its graph is captured (default 1): the first
    triggers allocate the buffers and the FFT plans the graph reuses.
``check_pointers``
    Check, before each launch, that the arrays of the object have not been
    reallocated (default ``true``).
``report_file``
    File where the table of the accelerated objects is written at the end.

Measured gains and accuracy
---------------------------

Time per step on a computer with two NVIDIA RTX PRO 5000 Blackwell GPUs, for
the configurations in ``config/`` (full runs, median after the first 20% of
the steps), without and with the accelerator, on one GPU and with the placement of
the :doc:`allocator <allocator>` (one process):

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 26

   * - Simulation
     - normal (ms)
     - accelerated
     - allocated
     - allocated, accelerated
     - accelerated objects
   * - SCAO SH (``scao``)
     - 2.06
     - 1.12
     - 2.25
     - 1.22
     - 11 of 13
   * - SCAO pyramid (``scao``)
     - 2.77
     - 1.16
     - 2.87
     - 1.18
     - 11 of 13
   * - space8m
     - 2.94
     - 1.20
     - 2.82
     - 1.34
     - 9 of 12
   * - SOUL
     - 3.23
     - 1.58
     - 3.48
     - 1.31
     - 13 of 14
   * - ANDES
     - 33.4
     - 32.9
     - 34.0
     - 31.5
     - 14 of 19
   * - MORFEO (full)
     - 64.1
     - 62.3
     - 41.9
     - 37.2
     - 53 of 63

The small simulations, whose time is set by the host, run 1.8 to 2.5 times
faster, close to the time of their GPU work; the large ones, set by
the GPUs, gain little from the accelerator alone (MORFEO gains from the
placement on two GPUs). Detector noise is the main host work left in the small
simulations: a ``CCD`` with noise is not accelerated (space8m runs at 0.70 ms
per step accelerated with the noise off, 1.20 ms with it). The "allocated"
columns use the placement chosen by the allocator for each mode; without the
accelerator, it is slower than one GPU on the small simulations (see
:doc:`allocator`).

Accuracy: with the noise of the detectors off, the outputs of the accelerated runs
(slopes, commands, residual fields, Strehl ratios over 100 steps) agree with
those of the normal run within 1e-5 of their range (they differ by rounding:
other kernels for the matrix products, and the order of some sums). With noise,
a rounding difference eventually changes a random sample (a Poisson count of a
detector pixel), after which the noise realization, and the run, differ: the
statistics agree (for example the mean Strehl ratio of SOUL, 0.8960 over 18700
steps, in all the modes). Compare accelerated and normal runs with the noise off.

Writing objects that can be accelerated
---------------------------------------

See ``can_accelerate()`` in :doc:`extension_api`. In short, ``trigger_code()`` must
only launch GPU work on arrays that are updated in place, with no Python value
that changes from one step to the next: values that change (time, inputs read
on the host, counters) are written to GPU arrays in ``prepare_trigger()``, which
runs at each step, and ``trigger_code()`` reads them from there. Use
``graph_matmul.matmul()`` for matrix products, and no random numbers from
cuRAND (``cupy.random.default_rng()`` keeps its state on the GPU and can be
captured, except for the distributions that copy data from the host, like
``poisson``).
