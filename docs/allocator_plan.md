# Plan: automatic allocation and scheduling of processing objects (iteras `Allocator`)

Status: second revision (2026-10-28). Phase A implemented on branch `iteras-allocator`,
together with parts of phases B and C (local search, process layouts, MPI broadcast,
allocation reuse, lazy CPU profiling). Section 11 lists where the implementation
differs from this plan.

The `Allocator` places the objects of a simulation on the available
computational resources (nodes, MPI processes, GPUs, CPUs) and, optionally,
chooses when each object runs. It is an iteras core feature: it only uses the
simulation graph, measured object sizes and timings, and a description of the
hardware, so SPECULA and any other plugin get it for free.

Use cases, in order of priority:

1. HPC: several nodes granted by slurm, several GPUs per node, memory fully available.
2. Local server with several GPUs, shared with other users (memory partly in use).
3. One GPU that is too small: some objects go to the CPU, choosing those that
   cost least in speed.

Decisions taken in review:

* Placements written in the YAML are respected by default; optionally the
  allocator supersedes them.
* Profiling runs on the GPU when one is available locally; a CPU profiling run
  happens only when requested (or when needed, see section 3).
* A saved allocation is reused unless a recomputation is requested; a saved
  allocation that no longer matches the simulation or the architecture is an error.
* Data objects used on several devices are duplicated.
* Object costs come from measurements only.
* Classes clearly meant to run on the CPU only get a class flag (section 6).
* The execution order may change well beyond the current trigger levels,
  including overlap of consecutive time steps, as long as results stay
  identical and deadlocks are impossible by construction (section 4).
* Verifying a new order by running the first steps twice is acceptable.

## 1. What already exists

| Item | Where | Use for the allocator |
|---|---|---|
| Per-object `target_rank`, `target_device_idx` | `Simul.build_objects()` | The placement output is these two parameters for every object |
| Overrides (`--overrides`, `*_override` sections, extra YAML files) | `Simul.combine_params()`, `apply_overrides()` | The allocation is applied as an override, saved and reused |
| Graph: inputs/outputs, delays, trigger levels | `build_trigger_order()`, `iterate_inputs()`, `split_output()` | Topology, intra-step (delay 0) and inter-step (delay -1) dependencies |
| Same device: input is a reference to the producer's output, no copy | `_InputItem.get()` | Zero communication cost, but a single buffer per output |
| Different device, same process: `copyTo()` / `transferDataTo()` | `_InputItem.get()` | Device-to-device copy, plus a cloned buffer on the receiver |
| Different process: `Ibsend` of `cpuArray(...)` + blocking `Recv` | `BaseProcessingObj.send_outputs()`, `_InputItem.receive_new_value()` | Always through host memory: D2H + MPI + H2D. Messages queue at the receiver |
| MPI send buffer: fixed, about 40 MB | `iteras.__init__` (`MPI.Attach_buffer`) | A full buffer makes `Ibsend` fail, not wait: limits how far a producer may run ahead |
| Readiness: trigger if some input has `generation_time >= current_time` | `BaseProcessingObj.checkInputTimes()` | Works unchanged if an object is run at a shifted time step |
| `*_object` parameters: restored once per consumer, on its device | `Simul.build_objects()` | Already duplicated |
| `*_ref` parameters: one shared instance | `Simul.build_objects()` | Duplicated per device, or a co-location constraint (section 5) |
| GPU memory per object, up to the first trigger | `startMemUsageCount()`, `gpu_mem_report()`, `LoopControl` | Memory probe |
| Per-object, per-phase timing, GPU events | `iteras.tracing` | Speed probe |
| Objects with a CUDA graph have their own stream | `BaseProcessingObj` | Starting point for concurrent execution on one GPU |
| Pre-roll objects must stay on one rank | `Simul._check_preroll_is_local()` | Constraint |

### Current execution model

* `build_trigger_order()` assigns each object a level: its longest distance
  from the sources over the delay-0 connections. A producer of a delayed output
  is moved after the readers of that output (it overwrites the value they read).
* Each process walks the levels in order, with only its own objects. For each
  level it runs `check_ready` (gathers inputs, blocking on MPI receives) for all
  its objects of that level, then `trigger`, then `post_trigger` + `send_outputs`.
* Processes synchronize only through messages, so a process with no pending
  inputs (e.g. one hosting only the atmosphere evolution) already runs ahead
  into the next steps, up to the size of the MPI buffer. Within a process,
  everything is sequential (apart from GPU asynchrony).

## 2. Architecture description (user input)

A new section, recognized by class (like `SimParams`), that can be in the main
YAML, in an extra YAML file, or given as an override. Its presence triggers the
allocation.

```yaml
architecture:
  class: 'Architecture'
  memory_source: 'declared'      # declared | probe | slurm
  memory_margin: 0.1             # fraction of each device kept free
  objective: 'speed'             # speed | min_resources | fit
  mode: 'apply'                  # apply | plan (write allocation and report, do not run)
  override_placement: false      # true: ignore target_rank/target_device_idx in the YAML
  allocation_file: 'allocation.yml'
  recompute: false               # true: ignore a saved allocation (also --reallocate)
  profile_cpu: false             # true: also measure every object on the CPU
  schedule: 'fixed'              # fixed | reorder | pipeline | streams (section 4)
  max_stages: 2                  # pipeline: maximum time-step offset between objects
  nodes:
    - name: 'gpu_node'
      count: 4                   # 4 identical nodes
      cpu: {cores: 32, memory_gb: 256}
      gpus: {count: 4, memory_gb: 80, model: 'A100'}
      intra_node: {gpu_gpu_gbps: 300, host_gpu_gbps: 25}   # NVLink / PCIe
  inter_node: {bandwidth_gbps: 25, latency_us: 2}
  processes:
    per_gpu: 1                   # default policy: one MPI rank per GPU
    cpu_ranks_per_node: 1        # extra ranks for CPU-only objects, if useful
```

* `declared`: the given memory is assumed fully available (HPC case).
* `probe`: each process reads the free memory of its devices
  (`cudaMemGetInfo`, `/proc/meminfo`); with MPI the values are all-gathered.
* `slurm`: node count, GPUs and memory read from `SLURM_*` and
  `CUDA_VISIBLE_DEVICES`.
* Default bandwidths come from a small table by link type (order of magnitude
  is enough). Relative speeds of GPU models are measured (section 3).
* With `override_placement: false`, objects with `target_rank` or
  `target_device_idx` in the YAML are pinned; with `true` the allocator places
  them freely and reports the user placement it ignored.

## 3. Profiling: measured memory and speed of each object

For every processing object *v* the allocator needs `mem_gpu(v)`, `t_gpu(v)`
and, when the CPU is a candidate, `mem_cpu(v)`, `t_cpu(v)`; for every connection
*e* the bytes per step `b(e)`; for every `_ref` data object its size, its
consumers, and whether any object modifies it during the run.

### 3.1 Profile cache

Results are stored in `<root_dir>/allocator_profiles/`, keyed by class name,
hash of the object parameters and of the shapes of its inputs, precision, and
device model. Profiling only measures what the cache does not have.

### 3.2 GPU profiling (default when a GPU is available)

Profiling needs memory and time, not correct physics, so an object can be
measured with *frozen inputs*: a sample of each input, of the right type and
shape, fed again at every step.

1. Build the objects on the local GPU in build order, inside the existing memory
   accounting. When the free memory would go below the margin, close the current
   *chunk*: release its objects and continue with a new chunk.
2. Run each chunk for a few steps (e.g. 2 warm-up + 5 measured) with the tracer
   in GPU-event mode:
   * memory: `startMemUsageCount` up to the first trigger (FFT plans, cuBLAS
     workspaces and CUDA graphs included), plus the pool peak during the steps;
   * time: host time per phase and GPU time of the trigger;
   * `b(e)`: size of each output after the first trigger.
3. Inputs coming from outside the chunk are frozen: boundary values are captured
   on the host when their producer runs and replayed into the next chunks.
   Delayed inputs whose producer is not built yet start from the producer's
   initial output; a feedback loop split across chunks is profiled open-loop,
   which is enough for memory and timing.
4. `_ref` data objects are checksummed before and after the steps: a changed
   checksum marks the object as *mutable*.
5. An object that does not fit the local GPU alone is marked "> free memory",
   its GPU memory is taken from its CPU memory (measured on the CPU with frozen
   inputs), and the report flags it as estimated.
6. The profile is per GPU model. On a heterogeneous cluster a short profiling
   job on each node type fills the cache (`--profile` can run alone, e.g. as the
   first step of a slurm job).

### 3.3 CPU profiling, only when requested or needed

* `profile_cpu: true` (or no GPU available): every object is also measured on
  the CPU, with the same chunk and frozen-input mechanism.
* Otherwise CPU timings are measured lazily: if the GPU-only allocation is
  infeasible, or limited by GPU memory, the solver picks the candidate objects
  for the CPU (largest memory per unit of GPU time saved), measures only those
  on the CPU, and solves again.

## 4. Execution order

This is where large gains can hide: placement alone cannot help when
processes wait for each other, or when work that nothing in the current step
needs delays work that something needs. The allocator therefore decides, for
each object, not only where it runs, but at which time step relative to the
others and in which position.

### 4.1 What fixes the results

Think of the run as a set of *instances* `(v, k)`: object *v* at time step *k*.
Results are identical, bit for bit on the same devices, for every execution
that respects these dependencies:

* **flow**: for a delay-0 connection u → v, `(u, k)` before `(v, k)`; for a
  delayed connection, `(u, k-1)` before `(v, k)`;
* **buffer reuse**: an output has `n` buffers (today `n = 1`), so the producer
  must not write step `k+n` into a buffer that a reader still has to read for
  step `k`. With one buffer, this is the current rule "reader of a delayed output
  before its producer". More buffers (section 4.4) remove most of these constraints;
* **hidden couplings**: objects sharing a mutable `_ref` data object, and an
  object with the processing objects it references (`dm_ref`, `sensor_dict_ref`,
  ...): their instances keep the original relative order;
* **barrier objects**: objects that interact with the outside world or modify
  shared state (display server, interactive and terminal input, stepping mode)
  stay at their original place relative to all other objects of the same step,
  and are never run ahead.

Changing delays (turning a delay-0 connection into a delayed one) changes the
physics: never applied, at most suggested in the report.

### 4.2 Deadlock freedom, by construction

Each process runs a sequence of instances; a receive blocks; a send does not
block but needs room in the MPI buffer.

* **Ordering.** The run cannot deadlock if the graph made of the dependencies
  of 4.1, plus "next instance on the same process", has no cycle. This holds
  when every process follows its instances in the order of **one global
  sequence** that respects the dependencies: a blocking receive then always waits
  for an instance earlier in that sequence. Conversely, any deadlock-free
  set of per-process sequences is compatible with some global sequence, so this
  representation loses no possible schedule. The two-process example
  (A: x1, y1; B: x2, y2; connections y1 → x2 and y2 → x1) is valid
  locally on each process, but has no global sequence, and does deadlock.
* **Buffer space.** When a producer runs ahead of its consumer on another process,
  messages accumulate. The allocator computes the bytes in flight on each process
  (`b(e)` times the step offset between the two ends) and sizes the MPI attached
  buffer accordingly (today it is a fixed 40 MB); `max_stages` bounds it.
* **Static check.** Before every run with a computed or user-given placement, the
  allocator builds this graph for a few consecutive steps (enough, because the
  schedule repeats every step) and verifies that it has no cycle and that the buffers
  are large enough. A failure is an error naming the cycle, before anything runs.
* **Runtime diagnostic.** A receive waiting more than a configurable time logs the
  object, tag and rank it waits for, to make any remaining problem easy to find.

### 4.3 Scheduling levels (`schedule:`)

* **`fixed`** (default): the current order. The allocator only chooses devices and
  processes. Its estimator simulates the real loop, including the natural
  running ahead of processes allowed by the MPI buffer.
* **`reorder`**: a new order within each step: one global sequence per step
  respecting 4.1, critical path first (objects that feed other processes, and
  objects on the feedback loop, before work that nothing in this step needs).
  It is expressed as level indices, one object per level where needed.
* **`pipeline`**: consecutive steps overlap (software pipelining). Each object gets
  a *stage* `s(v)` (0 ≤ s ≤ `max_stages`) and a position `o(v)`: at loop
  iteration `j` a process runs the instance `(v, j - s(v))` of each of its objects,
  in increasing `o(v)`. The global sequence is (iteration, position), so 4.2 holds;
  a dependency `(u, k-d)` → `(v, k)` is satisfied if `s(u) - d < s(v)`, or
  `s(u) - d = s(v)` and `o(u) < o(v)`.
* **`streams`**: within one process, independent objects on the same GPU run
  concurrently on separate CUDA streams, with CUDA events for the dependencies of
  4.1; the host order becomes only the launch order. Can be combined with the
  previous levels.

What the `pipeline` level gains, with typical AO examples:

* objects outside the feedback loop: atmosphere evolution, propagation and PSF
  of science sources, DataStore, displays. Today, on the same process as the loop,
  they add their time to every step; with a later stage they run while the
  process would otherwise wait for the next loop input;
* open-loop graphs (calibrations, pure feed-forward simulations): the time per
  step drops to that of the busiest device;
* feedback loops are not shortened: with one delayed connection in a loop, the
  loop latency is a hard limit (see 4.5).

### 4.4 Mechanisms needed in the core for `pipeline`

* `LoopControl` passes `t - s(v)·dt` to each object's `check_ready`; readiness by
  `generation_time` is unchanged. Prologue and epilogue: at iteration `j` only the
  instances with `0 ≤ j - s(v) < n_steps` run; the loop does `max(s)` extra iterations.
* Output buffers: for a same-process connection whose ends are on different
  stages, the connection keeps `n` copies of the value (a ring, `n` = stage
  distance + 1), so the producer never overwrites a value not yet read. This is in
  `connections.py`, not in the objects; its memory is in the cost model.
  Connections between processes need nothing: MPI messages already queue.
* MPI buffer size computed by the allocator (4.2).
* Barrier objects, pre-roll objects and replay sources stay at stage 0.

### 4.5 Objective and lower bounds

With overlapping steps, the right measure is the **steady-state time per step**
(the *initiation interval* of modulo scheduling), not the duration of one
isolated step. Two lower bounds are reported with every result, so the user sees
what limits the simulation:

* *resource bound*: the busiest device's total time per step;
* *loop bound*: for each feedback cycle, the sum of its object and transfer times
  divided by the number of delayed connections in it; the maximum over cycles.

The estimator simulates several consecutive steps (e.g. 10) of the chosen
schedule and returns the steady-state time per step; for `fixed` it measures the
natural overlap already allowed by MPI.

### 4.6 Verification

For `reorder`, `pipeline` and `streams`, `plan` mode runs a few steps with the
original order and with the new schedule, same placement, recording all outputs
with their time stamps, and compares them bit for bit. Any difference means a
hidden coupling not covered by 4.1: the schedule is rejected (falling back to the
previous level) and the report names the first object whose outputs differ.

### 4.7 Numerical note

Moving an object between CPU and GPU (unlike reordering) can change results at
the floating-point level, and objects that draw random numbers at construction
from the global generator (e.g. `ModulatedPyramid`, `ModulatedDoubleRoof`) get
different numbers on `numpy` and `cupy`. The allocation file is part of the run
record and is saved with the outputs.

## 5. Model

* Resources: a tree node → process (rank) → device (each process has its GPUs
  and one CPU device, `target_device_idx = -1`). Each device has a memory
  capacity `M_d` after the margin and the per-GPU CUDA context overhead.
* Memory used on device *d*:
  * the objects placed on *d*;
  * one copy of each `_ref` data object with a consumer on *d* (duplication);
  * cloned input buffers for inputs from another device, and the output rings of 4.4;
  * MPI staging and attached buffers (host memory of that rank).
* Co-location constraints (same process and device):
  * an object and the processing objects it references;
  * consumers of a mutable `_ref` data object;
  * pre-roll objects (same process, as today).
* Placement constraints from class flags: `cpu_only` objects on a CPU device,
  `main_process_only` objects on rank 0.
* Communication cost by placement of the two ends: same device 0; same process,
  different GPU `b/BW_gpu_gpu`; same node, different process
  `2 b/BW_host_gpu + b/BW_shm + latency`; different node adds `b/BW_net`.
* Time per step `T(x, schedule)`: discrete-event simulation of several steps,
  with measured host and GPU times; per process, host work is sequential; each
  GPU stream is sequential.

## 6. Algorithm

Hard constraints: device memory, pinned objects (unless `override_placement`),
co-location groups, class flags. Objective (default `speed`): minimize
`T(x, schedule)`; ties broken by communicated bytes, then by number of processes.

Class flags (added in Phase A):

* `cpu_only = True`: `DataStore`, `DataPrint`, `DisplayServer`, `InteractiveInput`,
  `TerminalInput`, and the display classes (`BaseDisplay` and subclasses);
* `main_process_only = True`: `DisplayServer`, `InteractiveInput`, `TerminalInput`;
* `barrier = True` (section 4.1): `DisplayServer` (it temporarily modifies data
  objects), `InteractiveInput`, `TerminalInput`.

Steps:

1. **Coarsening.** Contract the co-location groups, then chains where a
   connection costs more to transfer than both ends cost to compute.
2. **Initial placement: memory-aware HEFT** (Heterogeneous Earliest Finish Time,
   Topcuoglu et al. 2002): groups in order of their longest path to the end of
   the step; each goes to the device with the earliest estimated finish among
   those with room for it (objects, duplicated data objects, buffers). The CPU of
   each process is a candidate once its timings are known (section 3.3). If
   nothing fits anywhere: stop with a report (object, memory needed, largest
   free space).
3. **Scheduling** at the requested level (section 4.3), on the current placement:
   * `reorder`: list scheduling of one step with critical-path priorities;
   * `pipeline`: iterative modulo scheduling (Rau 1994): try a time per step
     starting from the larger of the two lower bounds of 4.5, assign stages and
     positions in priority order, increase the target if no legal schedule exists;
   * `streams`: assign independent objects of a process to streams, greedily.
4. **Refinement: local search** on `T(x, schedule)`: move one group, swap two
   groups, move a connected subgraph; memory first, then re-schedule (the
   scheduler is fast) and re-estimate. Deterministic (sorted iteration, fixed
   seed for optional annealing).
5. **Process packing.** Start with one rank per GPU used, plus CPU objects on the
   rank of the same node that exchanges most data with them. Merge ranks (e.g. one
   process driving two GPUs, no MPI) when `T` does not get worse.
6. **Uncoarsening.** Expand the groups and run step 4 again on single objects.
7. **Checks**: deadlock and buffer check (4.2), verification of the schedule (4.6).

Other objectives:
* `min_resources`: repeat for 1, 2, 4, ... GPUs/nodes and return the smallest
  configuration within X% of the best `T` (the report prints the table, useful to
  size a slurm request);
* `fit`: memory only, minimize communicated bytes (pure graph partitioning).

## 7. Workflow and output

* `allocation.yml`: an override file with `target_rank` / `target_device_idx`
  for every object, and the schedule (stages, positions, streams) when not
  `fixed`, plus a header with the hashes of the graph, parameters and architecture.
* Reuse: if `allocation.yml` exists it is used, unless `recompute: true` or
  `--reallocate`. If its hashes do not match, the run stops with an error naming
  what changed.
* `mode: plan` (or `iteras --allocate params.yml arch.yml`): profiles if needed,
  writes `allocation.yml` and a report: memory per device, estimated time per step
  and its two lower bounds, critical path or limiting cycle, bytes between
  processes, MPI buffer size, number of ranks to launch, schedule verification,
  and the simulation diagram colored by device.
* `mode: apply`: with MPI the number of ranks is fixed by the launcher; rank 0
  computes or loads the allocation and broadcasts it. If it needs more ranks
  than launched: error with the right `mpirun -n` value.
* The placement is applied before `build_objects()` through the normal override
  path; the schedule is passed to `LoopControl`.

## 8. Changes needed in iteras

* `iteras/data_objects/architecture.py`: `Architecture` (parsing, validation,
  slurm/probe readers).
* `iteras/allocator/`: `graph.py` (model, instance dependencies of 4.1),
  `profiler.py` (chunks, frozen inputs, cache), `cost.py` (memory and multi-step
  estimator), `solver.py` (coarsening, HEFT, local search, packing),
  `schedule.py` (reorder, modulo scheduling, streams, deadlock and buffer check),
  `report.py`.
* `Simul`: allocator hook before `build_objects()`; one instance of a `_ref` data
  object per device with a consumer, none on processes that do not need it;
  schedule passed to `LoopControl`; `--allocate`, `--profile`, `--reallocate`.
* `LoopControl`: per-object stage (time shift), prologue/epilogue, positions.
* `connections.py`: output rings for same-process connections across stages.
* `iteras.__init__`: MPI attached buffer size configurable.
* `BaseProcessingObj`: class flags `cpu_only`, `main_process_only`, `barrier`
  (default `False`); stream assignment for `streams`.
* Runtime diagnostic for long MPI receives.
* Later: CUDA-aware MPI or NCCL between ranks; only a new link type in the cost model.

## 9. Phases

* **A** – `Architecture` (declared memory), class flags, graph model, GPU profiling
  with chunks and frozen inputs, memory-aware HEFT with the `fixed`-order
  estimator, `plan` mode writing the override file and report. Tests on synthetic
  graphs and on fake architectures with tiny memories (CPU-only CI).
* **B** – `_ref` duplication in `Simul`, `apply` mode with MPI, allocation reuse,
  profile cache, lazy CPU profiling, deadlock/buffer check and MPI buffer sizing,
  receive diagnostic.
* **C** – local search, coarsening/uncoarsening, process packing, `min_resources`,
  `slurm` and `probe` memory sources, `override_placement`.
* **D** – `reorder` with verification.
* **E** – `pipeline`: stages in `LoopControl`, output rings, modulo scheduling,
  lower bounds in the report.
* **F** – `streams`.
* **G** – validation of the estimator against traced runs (SCAO on a small GPU
  with CPU spill; MCAO on the local 2-GPU server; multi-node run on HPC), tuning
  of the default bandwidth table. Partial validation already at the end of each
  phase.

## 10. Open questions

1. `pipeline` makes objects run at different time steps within one loop
   iteration. Objects that look at the simulation time for other purposes than
   readiness (e.g. a time-history generator, a DataStore `start_time`) still see
   their own shifted time, so they are fine; but anything that reads another
   object's state directly (beyond `_ref`, already a hidden coupling) would not be.
   Do you know of such cases in SPECULA?
2. `max_stages` default: 2 seems enough to hide the open-loop branches; more
   stages cost buffer memory. OK as a start?

## 11. Implementation notes (Phase A)

Differences from the plan above, and details decided while implementing:

* **Profiling without frozen inputs.** When the objects do not fit on the profiling
  GPU, the chunk being measured runs on the GPU and all the other objects run on the
  CPU, in the same simulation. All inputs are then real (feedback loops included),
  no new framework object is needed, and the CPU costs of the other objects are
  measured at the same time. Chunks are found by bisection on out-of-memory errors.
* **Two profiling runs.** One with a device synchronization after each phase (the
  time of each object alone, host and GPU), one without (the real step time, used to
  calibrate the loop overhead, and to find the objects that wait for the GPU work
  queued before them: without synchronization, their host time is much larger).
* **Memory.** `gpu_bytes_used` misses arrays allocated outside `__init__`/`setup`
  (e.g. objects restored from files), so the arrays reachable from each object are
  also walked. Arrays reachable from several objects (class-level caches, e.g. the
  SH kernels in MORFEO) are counted once per device that hosts any of them; arrays
  of the referenced data objects are counted with those.
* **Profile file.** Next to the allocation file (`allocation_profile.json`), not in
  the calibration directory, which may be shared or read-only. Entries are indexed by
  a key that depends on the object parameters, on the objects upstream of it and on
  the referenced data objects, but not on paths (`root_dir`, `store_dir`), run length,
  or placement: profiles measured on one computer can be used to plan for another.
* **Warm-up and measured steps.** The warm-up goes past the largest `start_time`
  parameter of any object, and at least two periods of the slowest object (largest
  `dt`) are measured (in MORFEO: the PSF starts at 0.05 s, the reference WFS run every
  5 steps).
* **Classes without `target_device_idx`.** `DataSource`, `DataBuffer` and the displays
  do not accept it: they stay on the default device of their process, and only their
  rank is chosen. `DataStore` now accepts it (as the Extension API requires), so that it
  can run on the CPU.
* **Data objects.** `Simul.ref_on_device()` duplicates a `_ref` data object on the
  device of each consumer. The allocation gives data objects a list of ranks and a
  device per rank (`target_rank: [0, 2]`, `target_device_idx: {0: 1, 2: 0}`), so they
  are built only where they are used.
* **Allocation file.** A valid override file, with the hashes needed for the stale
  check in a comment line.
* **CPU candidates.** When the GPUs are too small, objects are moved (in the cost
  model) from a GPU to the CPU one at a time, each time the one that frees most GPU
  memory per unit of GPU time, counting the data objects only it uses, until the rest
  fits with 20% to spare; only those are measured on the CPU, in one synchronized run.
  On MORFEO, `prop` (13 s per step on the CPU, 5 ms on the GPU) is never measured there.
* **Nodes to launch.** Only the nodes that host objects are part of the allocation;
  identical nodes are renumbered so that the used ones come first.
* **Where objects wait for the GPU.** Objects with a CUDA graph wait for it in
  `post_trigger` (framework code); others are found by comparing the two profiling
  runs phase by phase. The estimator applies each wait in its phase, so that GPUs
  driven by one process overlap.
* **MPI progress.** The framework sent outputs with buffered sends (Ibsend) whose
  data moved only when the sender called MPI again: two MORFEO ranks waited for each
  other up to 60 ms per step. Now (iteras.mpi_progress) outputs are sent with Isend,
  the host copies kept until completion, so that the receiver pulls the data when
  the transport allows it (single-copy shared memory, likely RDMA reads with UCX
  over InfiniBand), and a process with sends in flight polls them while it waits for
  its GPU. On the local server the two-rank allocation went from 74 to 42 ms per step,
  with bit-identical results. The cost model follows the same rules (single_copy per
  link, polling waits). Over TCP (no single-copy) the gain is small: the data of a
  large message is pushed in fragments, each needing the sender to call MPI; more
  progress points (or a progress thread) would be needed there.
* **Framework bugs found by the experiments** (fixed on this branch): data objects
  restored from a tag failed on the ranks that do not build them; data objects
  connected as inputs were "sent" to other ranks; `transferDataTo()` did not copy
  0-dimensional arrays in place (and silently skipped the copy between GPU arrays).
* **Objects split over several devices** (2026-10-29). Class hooks
  `devices_used(params)` and `single_device_params(params)`; `DistributedSH` uses
  `n_slices` devices and is profiled with `n_slices: 1`. Their placement by hand is
  kept even with `override_placement`; `pin_ranks_per_node` maps hand-written ranks to
  nodes; one process per node is considered on several nodes too. On the local
  server, MORFEO with 2-slice LGS sensors: allocator 42.7 ms (estimate 37.5),
  simple layout 45.8 ms (estimate 43.7); plain-SH allocation 41.4 ms. DistributedSH
  differs from SH at rounding level (3.6e-7), amplified by the closed loop.
* **Node count.** With identical nodes the search runs on the first k nodes for each
  k and keeps the fastest, preferring fewer nodes within `node_tolerance` (5%).
