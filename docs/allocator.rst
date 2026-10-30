.. _allocator:

Automatic allocation
====================

The allocator places the objects of a simulation on the available resources
(nodes, MPI processes, GPUs and CPUs), choosing the placement with the lowest
estimated time per step that fits in the memory of each device. It is part of
the iteras framework, so it works for any plugin.

It is enabled by adding a section whose class is ``Architecture`` to the
parameter files, for example in a separate file given after the others::

    specula params.yml architecture.yml

Describing the resources
------------------------

On a single computer, the GPUs and the CPU can be detected at run time:

.. code-block:: yaml

    architecture:
      class: 'Architecture'
      nodes: 'local'
      memory_source: 'probe'      # memory free right now (the GPUs may be shared)

On a cluster, the nodes are described explicitly. The memory is assumed to be
fully available (``memory_source: 'declared'``, the default), as on nodes
granted by slurm. For example, two Leonardo Booster nodes:

.. code-block:: yaml

    architecture:
      class: 'Architecture'
      nodes:
        - name: 'booster'
          count: 2
          cpu: {cores: 32, memory_gb: 512}
          gpus: {count: 4, memory_gb: 64, model: 'A100'}
          links: {gpu_gpu_gb_s: 100, host_gpu_gb_s: 12}
      inter_node: {bandwidth_gb_s: 10, latency_us: 5}
      processes: 'per_gpu'

With ``memory_source: 'slurm'``, the number of nodes, cores, memory and GPUs
are read from the slurm environment of the job, and the rest from the first
node type.

The main options (see :class:`iteras.data_objects.architecture.Architecture`
for all of them):

``mode``
    ``'apply'`` (default): allocate, then run the simulation. ``'plan'``: only
    write the allocation and its report (also ``--allocate`` on the command line).
``processes``
    ``'auto'`` (default) compares one process per GPU with one process per node
    driving all its GPUs; ``'per_gpu'``; ``'single'`` (one process per node:
    no MPI with a single node).
``override_placement``
    Objects with ``target_rank`` or ``target_device_idx`` in the parameter files
    are not moved, unless this is ``true`` (objects split over several devices
    are never moved, see below).
``pin_ranks_per_node``
    How many ranks per node the ``target_rank`` of the parameter files assume
    (rank r is on node r // pin_ranks_per_node). By default, as in each process
    layout considered.
``allocation_file``
    Where the allocation is written (default ``allocation.yml``), with a report
    (``allocation_report.txt``) and the measured costs (``allocation_profile.json``).
``memory_margin``
    Fraction of each device kept free (default 0.1).
``device_tolerance``
    The placement of all the objects on one GPU is kept unless the placement
    found by the search, on more devices, is estimated faster by more than this
    fraction (default 0.05), since the estimate of the copies between devices is
    the least reliable part of the model.

How it works
------------

1. **Profiling.** The costs of the objects are measured by running the
   simulation itself for a few steps on the local GPU with most free memory
   (or on the CPU if there is none): host time of each phase, GPU time, memory,
   arrays shared between objects, size of each output, host time to copy each
   output to another GPU and to the CPU (as the framework copies an input from
   another device at each step), size of the data objects referenced with
   ``_ref``. The warm-up goes past the largest ``start_time``
   parameter, and the measurement covers two periods of the slowest object
   (largest ``dt``). If the objects do not fit on the GPU, they are measured in
   chunks, the rest of the simulation running on the CPU. The CPU is measured
   only if requested (``profile_cpu: true``), or for a few objects when the GPUs
   are too small for the whole simulation. The costs are kept in the profile
   file and reused.
2. **Estimation.** The time per step of a placement is estimated by simulating
   a few steps of the loop: each process runs its objects in trigger order, the
   GPU work of each device runs in parallel with the host, data is copied between
   devices and sent between processes as the framework does it: a copy between
   two GPUs is ordered on the GPUs (the host does not wait for it), a copy to the
   CPU waits for the data.
3. **Search.** Objects that must stay together (an object and the processing
   objects it references, the users of a data object that some object modifies)
   are placed as a unit. An initial placement (memory-aware HEFT) is improved by
   moving and swapping objects, for each process layout. With several identical
   nodes, the search is made on the first k nodes for each k, and the fastest
   placement is kept, with the fewest nodes if others are within
   ``node_tolerance`` (5%): the report lists the time per step for each number
   of nodes. Likewise, all the objects on one GPU are preferred to a
   placement on more devices that is not faster by more than
   ``device_tolerance`` (5%).

Objects of classes that must run on the CPU or in the main process (see
:doc:`extension_api`) are placed accordingly. Objects of classes whose
constructor does not accept ``target_device_idx`` stay on the default device of
their process. A data object used by objects on several devices is duplicated
on each of them.

Some objects split their work over several devices, like ``DistributedSH``,
which runs ``n_slices`` parts of a Shack-Hartmann sensor on the GPU given by
``target_device_idx`` and on the following ones (see ``devices_used()`` in
:doc:`extension_api`). They are measured on a single device (with
``n_slices: 1``), and the estimate divides their memory and GPU time among their
devices, with the copies of their inputs and the gathering of the results. When
such an object is placed by hand (``target_rank`` and ``target_device_idx``), it
stays there, even with ``override_placement``: for example, with the
``DistributedSH`` objects of ``config/leonardo/overrides/ov_MORFEO_leonardo_ds4.yml``
(one rank per node, so ``pin_ranks_per_node: 1``), the allocator places the
other objects around them.

With the accelerator
--------------------

With ``--accelerate`` or an ``Accelerator`` section (see :doc:`accelerator`), the objects are
profiled as they run accelerated: the host time of an accelerated object is the time to
launch its CUDA graph, and it does not wait for the GPU in ``post_trigger()``.
These costs are kept apart from those measured without accelerator, and an
allocation computed with the accelerator is stale without it, and the reverse.

Using the allocation
--------------------

The allocation file contains ``<object>_override`` sections with
``target_rank`` and ``target_device_idx``, so it can also be given as an
additional parameter file without the ``Architecture`` section. With the
``Architecture`` section, later runs reuse it; if the simulation, the
placements in the parameter files or the architecture have changed, the run
stops with an error that names the difference. Use ``--reallocate`` (or
``recompute: true``) to compute a new one.

If the allocation needs several processes, the simulation must be launched
with MPI and the number of ranks given in the report::

    mpirun -n 8 specula --mpi params.yml architecture.yml

Planning for another computer
-----------------------------

The costs depend on the GPU model. The profile file keeps them per model (the
name reported by CUDA, like ``NVIDIA A100-SXM-64GB``), and a ``model`` given in
the architecture matches a profiled model whose name contains it (``A100``).
For a model that was never profiled, the GPU times of a profiled one are
divided by ``relative_speed`` (for example 0.5 for a GPU twice as slow): useful
for a first plan, but only an assumption.

To measure the GPUs of a cluster, run the allocator there in plan mode, on one
node, with the profile file of the other computer or without it: when the GPU of
the node is one of the models of the architecture and the profile has no costs
for it, the objects are measured on it. For example, in a slurm job on one node::

    specula --allocate params.yml architecture.yml

The plan for any number of nodes can then be computed from that profile file,
anywhere.

Limitations
-----------

* The trigger order is not changed (``schedule: 'fixed'``).
* GPUs of a model that was not profiled use the times of the profiled one,
  divided by ``relative_speed``.
* Transfers between processes go through the host, as in the framework.
* The MPI transfers are modeled as described in :ref:`running with MPI
  <running_with_mpi>`: on links where the receiver cannot pull the data by
  itself (``single_copy: false``, the default between nodes), the data of a
  large message moves only while the sender is inside MPI, and the estimate is
  less reliable. With TCP between two processes of the same computer, MORFEO ran
  about 60% slower than with single-copy shared memory.
* Small, host-bound simulations: without the :doc:`accelerator <accelerator>`,
  the placements on two GPUs of SCAO, space8m and SOUL ran 4-14% slower than
  all the objects on one GPU, although the copy times are measured and
  ``device_tolerance`` prefers one GPU for small estimated gains: the host time
  of these runs is not modeled accurately enough. With the accelerator, the
  placements run as fast as one GPU. On MORFEO, the placement on two GPUs runs
  1.66 times faster than one GPU (1.72 with the accelerator).
* Accuracy: on a computer with two GPUs, the estimates of MORFEO for ten
  placements, with one or two processes, were within 5% of the measured times,
  except for the placements chosen by the allocator itself, which were
  optimistic by up to 10% (the search favors the placements where the model
  errs on the fast side), and for the old buffered MPI sends (15%).
