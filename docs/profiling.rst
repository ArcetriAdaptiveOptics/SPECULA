.. _profiling:

Profiling and timing
====================

SPECULA can report where time is spent during a simulation, per object and per phase.
Two complementary mechanisms are available:

- **NVTX ranges**, always emitted when NVTX is available through CuPy, and visible in the
  timeline of NVIDIA Nsight Systems together with the CUDA kernels.
- **A trace file**, enabled with ``--trace-file``, with one line per object and phase, plus a
  summary table sorted by total time. It needs no external tool and also works on CPU.

Both are implemented in :mod:`specula.tracing` and are driven by the simulation loop.

Traced phases
-------------

At every iteration, each processing object goes through these phases, in this order:

``inputs``
    Gathering of inputs, including MPI receives and copies between devices
    (:meth:`~specula.base_processing_obj.BaseProcessingObj.checkInputTimes`).

``prepare_trigger``
    :meth:`~specula.base_processing_obj.BaseProcessingObj.prepare_trigger`, only if the
    object's inputs have been refreshed.

``trigger``
    :meth:`~specula.base_processing_obj.BaseProcessingObj.trigger`: the actual computation
    (``trigger_code()``, or the launch of its CUDA graph).

``post_trigger``
    :meth:`~specula.base_processing_obj.BaseProcessingObj.post_trigger`.

``send_outputs``
    Sending of outputs to other MPI ranks. Only traced for objects with remote outputs.

In addition, the ``setup`` phase of each object is traced once, before the loop starts.

Phases are marked where the simulation loop calls them, so methods overridden in derived
classes are fully included.

NVTX ranges and Nsight Systems
------------------------------

Each phase is marked with an NVTX range named ``<object name>.<phase>``, for example
``slopec.trigger``, where the object name is the one used in the YAML file.
Each phase has its own color. No command line option is needed:

.. code-block:: bash

    nsys profile -t cuda,nvtx -o my_run specula params.yml

Open ``my_run.nsys-rep`` in Nsight Systems: the NVTX row shows the phases of each object on the
host timeline, and the CUDA rows show the kernels they launch.

If CuPy is not installed, or NVTX is not available, ranges are silently disabled.
To mark additional sections of code in a processing object, use
:func:`specula.show_in_profiler` as a decorator or context manager:

.. code-block:: python

    from specula import show_in_profiler

    with show_in_profiler('my_object.interpolation'):
        ...

Trace file
----------

.. code-block:: bash

    specula params.yml --trace-file run.tsv

This writes two files:

``run.tsv``
    One tab-separated line per object and phase, with columns:

    ================  ====================================================================
    ``iter``          Loop iteration (``-1`` for the setup phase)
    ``t_sim_s``       Simulated time, in seconds
    ``object``        Object name, as in the YAML file
    ``class``         Object class
    ``phase``         Phase name (see above)
    ``start_us``      Start time in microseconds, relative to the start of the trace
    ``dur_us``        Duration in microseconds
    ================  ====================================================================

``run.summary.txt``
    All (object, phase) pairs sorted by total time, with number of calls, total, mean and
    maximum time, and percentage of the loop time. The same summary is logged at the end of
    the run. For example:

    .. code-block:: text

        Iterations: 10, time in loop iterations: 150.2 ms (15021.9 us/iteration), device sync: False, ...
        Loop time outside traced phases (loop overhead, speed report, ...): 3.6 ms (2.4%)

        object          class            phase             count   total_ms    mean_us     max_us  %loop
        psf             PSF              trigger              10      61.85     6185.1     7156.1   41.2
        sh              SH               prepare_trigger      10      30.85     3085.2     5059.4   20.5
        sh              SH               trigger              10      25.11     2510.5     2637.0   16.7

    The "loop time outside traced phases" line is the time spent in the loop itself, outside
    any object: a large value points to overhead in the loop or in the objects' bookkeeping.

With MPI, each rank writes its own files, with the rank number added to the name
(``run.rank0.tsv``, ``run.rank0.summary.txt``, ...).

The trace file is easy to analyze with pandas:

.. code-block:: python

    import pandas as pd

    df = pd.read_csv('run.tsv', sep='\t', comment='#')
    loop = df[df['iter'] >= 0]
    print(loop.groupby(['object', 'phase']).dur_us.describe())

Durations are inclusive: if a phase calls another object's phase (for example an object that
triggers an internal DM), the nested time is also counted in the enclosing phase.

Options
~~~~~~~

The following options modify the trace file. They have no effect without ``--trace-file``.

``--trace-skip N``
    Do not record the first ``N`` loop iterations. The first iterations include one-time costs,
    such as FFT plan creation and compilation of fused kernels, that are not representative of
    the steady state. The setup phase is always recorded.

``--trace-sync``
    Synchronize the GPU at the end of each phase, so that durations include the execution of the
    GPU work launched in that phase (see below). Host and GPU no longer overlap, so the
    simulation runs slower than normal.

``--trace-gpu-events``
    Measure the GPU time of each ``trigger`` with CUDA events, without synchronizing the GPU.
    Results are written as an additional ``trigger_gpu`` phase, a few iterations later, when the
    events have completed. Only objects running on a GPU get ``trigger_gpu`` lines.

Typical usage:

.. code-block:: bash

    specula params.yml --trace-file run.tsv --trace-skip 10 --trace-gpu-events

Host time and GPU time
----------------------

GPU work is asynchronous: a ``trigger`` returns as soon as its kernels have been queued. Without
options, trace durations are therefore **host** times, the time spent by Python to prepare and
launch the work. Which option to use depends on the question:

======================  =====================================  =======================================
Option                  Measures                               Cost
======================  =====================================  =======================================
(none)                  Host time of each phase                Negligible
``--trace-gpu-events``  Host time, plus time taken on the GPU  Two CUDA events per GPU trigger;
                        stream by each ``trigger``             reading them rarely stalls the host
``--trace-sync``        Host time including completion of      A device synchronization per phase:
                        all GPU work of each phase             no overlap, slower simulation
======================  =====================================  =======================================

Comparing ``trigger`` and ``trigger_gpu`` for the same object is a quick way to see what limits it:

- ``trigger_gpu`` close to ``trigger``: the object is limited by Python and kernel launch overhead.
  Reducing the number of CuPy calls (fusing operations, avoiding temporaries) helps more than
  faster kernels.
- ``trigger_gpu`` much larger than ``trigger``: the object is limited by GPU execution.

Some caveats for ``trigger_gpu``:

- Events are recorded on the stream where the object's work runs: its own stream if it uses a
  CUDA graph, the current stream otherwise. Work that an object sends to other streams is not
  included.
- The measured time is that of the stream segment between the two events, so it also includes
  time the GPU spends waiting for the host to launch the next kernel.
- Objects running concurrently on different streams are measured independently: their times can
  add up to more than the wall time.

Python API
----------

The same options are available as arguments of :func:`specula.main_simul`:

.. code-block:: python

    import specula

    specula.main_simul(['params.yml'], trace_file='run.tsv', trace_skip=10,
                       trace_gpu_events=True)

When using :class:`specula.simul.Simul` directly, open and close the tracer around the run:

.. code-block:: python

    import specula
    specula.init(0)

    from specula.simul import Simul
    from specula.tracing import tracer

    tracer.open('run.tsv', skip=10, gpu_events=True)
    try:
        Simul('params.yml').run()
    finally:
        summary = tracer.close()   # also writes run.summary.txt
    print(summary)

Python profiler
---------------

For a function-level view of the host code, ``--profile`` runs the simulation under Python's
``cProfile`` and prints the functions with the highest cumulative time at the end of the run.
It adds a significant overhead to every Python call, so its timings are best used to compare
functions with each other, not as absolute values.
