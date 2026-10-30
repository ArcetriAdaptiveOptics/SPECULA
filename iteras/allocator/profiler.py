'''
Measured costs of the objects of a simulation.

The simulation itself is run for a few steps, with all the objects on one
GPU (or, if they do not fit, with a part of them on the GPU and the rest on
the CPU), and the tracer and the memory accounting of iteras record:

* per object and per step: the host time of each phase, the GPU time of the
  trigger, whether the object waits for the GPU work queued before it;
* the memory of each object, the arrays shared by several objects, the size
  of each output and of the data objects referenced with "_ref".

Two runs are made: one synchronizing the GPU after each phase (the time of
each object alone) and one without (the real step time, and which objects
wait for the GPU).
'''
import gc
import os
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass, field, asdict

import yaml

import iteras
from iteras import np
from iteras import registry
from iteras.base_time_obj import BaseTimeObj
from iteras.display.base_display import BaseDisplay
from iteras.display.display_recorder import DisplayRecorder
from iteras.processing_objects.data_store import DataStore
from iteras.log import get_iteras_logger

logger = get_iteras_logger(__name__)

PROFILE_VERSION = 4
HOST_PHASES = ('inputs', 'prepare_trigger', 'trigger', 'post_trigger', 'send_outputs')


@dataclass
class Cost:
    '''Cost of one object on one kind of device. Times are ns per step, memory in bytes.'''
    h_in: float = 0.0       # gathering inputs (no copies: measured with local inputs)
    h_prep: float = 0.0     # prepare_trigger
    h_trig: float = 0.0     # host part of trigger
    h_post: float = 0.0     # post_trigger
    h_send: float = 0.0     # sending outputs (only for remote outputs)
    g: float = 0.0          # GPU time of trigger
    active: float = 1.0     # fraction of the steps in which the object is triggered
    syncs: str = ''         # phase ('trigger' or 'post') in which the host waits for the
                            # GPU work of the object (and all the work queued before it)
    graph: bool = False     # runs a CUDA graph: waits in post_trigger through mpi_progress,
                            # which keeps the MPI sends of the process moving
    mem: float = 0.0        # memory not shared with other objects
    outputs: dict = field(default_factory=dict)    # output name -> bytes (all its arrays)
    mpi_outputs: dict = field(default_factory=dict)    # output name -> bytes sent by MPI
    copy_gpu: dict = field(default_factory=dict)    # output name -> host ns to copy it to another GPU
    copy_cpu: dict = field(default_factory=dict)    # output name -> host ns to copy it to the CPU
    measured: bool = True   # False if derived from another device kind

    @property
    def host(self):
        return self.h_in + self.h_prep + self.h_trig + self.h_post

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class OutOfDeviceMemory(Exception):
    pass


def _is_oom(exc):
    cp = iteras.cp
    if cp is not None and isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    return 'out of memory' in str(exc).lower() or 'out_of_memory' in str(exc).lower()


# ---------------------------------------------------------------------------
# Arrays reachable from an object

def array_blocks(obj, stop):
    '''
    Memory blocks of the arrays reachable from *obj*: {(kind, pointer): bytes},
    where kind is 'g' (GPU) or 'h' (host). The walk does not enter the objects
    whose id is in *stop* (the other simulation objects), nor the inputs.
    '''
    cp = iteras.cp
    roots = registry.plugin_roots()
    blocks = {}
    visited = set()
    todo = [obj]
    while todo:
        o = todo.pop()
        if id(o) in visited:
            continue
        visited.add(id(o))
        if cp is not None and isinstance(o, cp.ndarray):
            mem = o.data.mem
            blocks[('g', mem.ptr)] = mem.size
        elif isinstance(o, np.ndarray):
            base = o
            while isinstance(base.base, np.ndarray):
                base = base.base
            blocks[('h', base.__array_interface__['data'][0])] = base.nbytes
        elif isinstance(o, dict):
            todo.extend(o.values())
        elif isinstance(o, (list, tuple, set, frozenset)):
            todo.extend(o)
        elif isinstance(o, BaseTimeObj):
            if o is not obj and id(o) in stop:
                continue
            todo.extend(v for k, v in vars(o).items() if k not in ('inputs', 'local_inputs'))
        elif type(o).__module__.partition('.')[0] in roots and hasattr(o, '__dict__') \
                and not isinstance(o, type):
            todo.extend(vars(o).values())
    return blocks


def array_checksum(obj):
    '''Cheap checksum of the numeric arrays reachable from *obj*'''
    cp = iteras.cp
    sums = []
    todo = [obj]
    visited = set()
    roots = registry.plugin_roots()
    while todo:
        o = todo.pop()
        if id(o) in visited:
            continue
        visited.add(id(o))
        is_cp = cp is not None and isinstance(o, cp.ndarray)
        if is_cp or isinstance(o, np.ndarray):
            if o.dtype.kind in 'biufc' and o.size:
                xp = cp if is_cp else np
                v = o.astype(xp.complex128 if o.dtype.kind == 'c' else xp.float64)
                s = complex(xp.sum(v)) + complex(xp.sum(v * v.conj()))
                sums.append((o.shape, round(s.real, 6), round(s.imag, 6)))
        elif isinstance(o, dict):
            todo.extend(o.values())
        elif isinstance(o, (list, tuple)):
            todo.extend(o)
        elif hasattr(o, '__dict__') and not isinstance(o, type) and \
                (isinstance(o, BaseTimeObj) or type(o).__module__.partition('.')[0] in roots):
            todo.extend(vars(o).values())
    return sorted(sums, key=str)


# ---------------------------------------------------------------------------
# Profiling runs

def _profile_params(params, graph, devices, workdir, steps, warmup):
    '''Parameters of a profiling run: objects on *devices*, short run, no displays'''
    p = deepcopy(params)
    main = p[graph.main_key]
    main['total_time'] = (warmup + steps) * graph.main_params['time_step'] * (1 + 1e-9)
    if 'display_server' in main:
        main['display_server'] = False
    for name, klass in graph.classes.items():
        if name not in p:
            continue
        if issubclass(klass, (BaseDisplay, DisplayRecorder)):
            del p[name]
            continue
        p[name].pop('target_rank', None)
        if graph.spread.get(name, 1) > 1:
            # Measured on one device; the estimator divides its work among its devices
            p[name] = dict(klass.single_device_params(p[name]))
        if name in devices:
            if graph.device_settable[name]:
                p[name]['target_device_idx'] = devices[name]
            else:
                p[name].pop('target_device_idx', None)
        elif name in graph.data_objs:
            consumers = graph.ref_consumers.get(name)
            if consumers:
                p[name]['target_device_idx'] = devices[consumers[0]]
        if issubclass(klass, DataStore):
            p[name]['store_dir'] = os.path.join(workdir, 'store')
    return p


def _parse_trace(filename):
    '''{(object, phase): total ns} over the recorded iterations, and per-object trigger counts'''
    totals = {}
    triggers = {}
    with open(filename) as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if line.startswith('#') or len(fields) != 7 or fields[0] == 'iter':
                continue
            it, _, name, _, phase, _, dur = fields
            if int(it) < 0:
                continue
            totals[(name, phase)] = totals.get((name, phase), 0.0) + float(dur) * 1e3
            if phase == 'trigger':
                triggers[name] = triggers.get(name, 0) + 1
    return totals, triggers


def run_once(simul_cls, params, graph, devices, workdir, steps, warmup, sync, tag):
    '''
    Run the simulation for warmup + steps steps, with the processing objects on
    *devices* ({name: device index}). Return the Simul object, the trace totals
    and the loop time per step.
    '''
    from iteras.tracing import tracer
    os.makedirs(workdir, exist_ok=True)
    p = _profile_params(params, graph, devices, workdir, steps, warmup)
    yml = os.path.join(workdir, f'profile_{tag}.yml')
    with open(yml, 'w') as f:
        yaml.safe_dump(p, f, sort_keys=False)
    trace = os.path.join(workdir, f'profile_{tag}.tsv')

    saved = None
    if tracer.recording:
        saved = dict(tracer.__dict__)
        tracer._file = None
    # A profiling run is a single-process run, even inside an MPI job
    # (only rank 0 profiles, while the other ranks wait for the allocation)
    mpi = iteras.process_comm, iteras.process_rank
    iteras.process_comm, iteras.process_rank = None, None
    sim = simul_cls(yml)
    loop_ns = n_iter = 0
    tracer.open(trace, sync=sync, skip=warmup, gpu_events=iteras.cp is not None)
    try:
        sim.run()
    finally:
        loop_ns, n_iter = tracer._loop_ns, tracer._n_iter
        tracer.close()
        if saved is not None:
            tracer.__dict__.update(saved)
        iteras.process_comm, iteras.process_rank = mpi
    totals, triggers = _parse_trace(trace)
    return sim, totals, triggers, loop_ns / max(n_iter, 1), max(n_iter, 1)


def _transfer_times(obj, dev, repeat=15):
    '''
    Host time (ns) to copy each output of *obj*, on GPU *dev*, to another GPU
    (if there is one) and to the CPU, as InputValue.get() copies an input from
    another device at each step.
    '''
    from iteras.base_data_obj import BaseDataObj
    cp = iteras.cp
    ngpu = cp.cuda.runtime.getDeviceCount()
    to_gpu, to_cpu = {}, {}
    targets = [(to_cpu, -1)] + ([(to_gpu, (dev + 1) % ngpu)] if ngpu > 1 else [])
    for name, value in obj.outputs.items():
        items = value if isinstance(value, list) else [value]
        if not items or not all(isinstance(i, BaseDataObj) for i in items):
            continue
        for table, target in targets:
            try:
                with cp.cuda.Device(target if target >= 0 else dev):
                    clones = [i.copyTo(target) for i in items]
                    times = []
                    for n in range(repeat + 2):
                        t0 = time.perf_counter()
                        for i, clone in zip(items, clones):
                            i.transferDataTo(clone)
                        if n >= 2:
                            times.append(time.perf_counter() - t0)
                    cp.cuda.Device(dev).synchronize()
                    cp.cuda.Device().synchronize()
                table[name] = float(np.median(times) * 1e9)
            except Exception as e:
                logger.debug(f'Allocator: copy of {obj.name}.{name} to device {target} not measured: {e}')
            finally:
                clones = None
    return to_gpu, to_cpu


def _mpi_bytes(value, default):
    '''Bytes sent by MPI for an output: the get_value() of each data object'''
    total = 0
    for item in value if isinstance(value, list) else [value]:
        try:
            v = item.get_value()
        except Exception:
            return default
        total += getattr(v, 'nbytes', 8)
    return float(total)


def _release(sim):
    '''Drop the objects of a profiling run (the caller must drop *sim* too) and free their memory'''
    sim.objs.clear()
    loop = getattr(sim, 'loop', None)
    if loop is not None:
        loop.trigger_lists.clear()
        loop.setup_order = []
    _free_memory()


def _free_memory():
    gc.collect()
    cp = iteras.cp
    if cp is not None:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def _make_simul_cls(on_built):
    from iteras.simul import Simul

    class _ProfilingSimul(Simul):
        def __init__(self, yml):
            super().__init__(yml, speed_report=False, log_level='warning')

        def on_objects_built(self):
            on_built(self)

    return _ProfilingSimul


@dataclass
class RunResult:
    '''Costs measured in one profiling configuration'''
    costs: dict             # name -> (kind, Cost)
    shared: list            # [{'kind', 'owners': [names], 'bytes'}]
    refs: dict              # data object name -> {'bytes', 'mutable'}
    step_ns: float          # loop time per step without synchronization
    devices: dict           # name -> device index used


def _run_or_oom(*args, **kwargs):
    '''run_once(), raising OutOfDeviceMemory if the GPU memory is not enough'''
    oom = None
    try:
        return run_once(*args, **kwargs)
    except Exception as e:
        if not _is_oom(e):
            raise
        oom = str(e)
    # The traceback, which kept the objects of the failed run alive, is gone:
    # release their memory before the next attempt
    _free_memory()
    raise OutOfDeviceMemory(oom)


def profile_run(params, graph, devices, workdir, steps=5, warmup=3, gpu_model='', async_run=True):
    '''
    Measure the objects placed on *devices*. Raises OutOfDeviceMemory if they
    do not fit. Without *async_run*, the second run (step time and objects
    waiting for the GPU) is skipped: enough to measure objects on the CPU.
    '''
    checksums = {}

    def on_built(sim):
        for d in graph.data_objs:
            if d in sim.objs and graph.ref_consumers.get(d):
                checksums[d] = array_checksum(sim.objs[d])

    sim, totals, triggers, _, n_iter = _run_or_oom(_make_simul_cls(on_built), params, graph,
                                                   devices, workdir, steps, warmup, sync=True,
                                                   tag='sync')

    # Devices actually used (objects that do not accept target_device_idx use the default one)
    devices = {n: sim.objs[n].target_device_idx for n in graph.procs if n in sim.objs}
    try:
        costs, shared, refs = _measure(sim, graph, devices, totals, triggers, n_iter,
                                       checksums, gpu_model)
    finally:
        _release(sim)
        del sim

    if not async_run:
        return RunResult(costs, shared, refs, None, dict(devices))
    # Second run, without synchronization: real step time, and objects waiting for the GPU
    sim2, totals2, _, step_ns, n_iter2 = _run_or_oom(_make_simul_cls(lambda s: None), params,
                                                     graph, devices, workdir, steps, warmup,
                                                     sync=False, tag='async')
    _release(sim2)
    del sim2
    for name, (kind, c) in costs.items():
        if kind == 'cpu' or c.syncs:
            continue
        # Without synchronization, the host time of a phase that waits for the GPU
        # work of the object includes that work (and, since the stream runs in order,
        # all the work queued before it). Only the first object waiting in a trigger
        # level can be detected this way: objects with a CUDA graph are known above.
        threshold = max(0.5 * c.g, 0.2e6)
        a_trig = (totals2.get((name, 'prepare_trigger'), 0.0) + totals2.get((name, 'trigger'), 0.0)) / n_iter2
        a_post = totals2.get((name, 'post_trigger'), 0.0) / n_iter2
        if a_trig - (c.h_prep + c.h_trig) > threshold:
            c.syncs = 'trigger'
        elif a_post - c.h_post > threshold:
            c.syncs = 'post'
    return RunResult(costs, shared, refs, step_ns, dict(devices))


def _measure(sim, graph, devices, totals, triggers, n_iter, checksums, gpu_model):
    stop = {id(o) for o in sim.objs.values()}
    ref_blocks = {d: array_blocks(sim.objs[d], set()) for d in graph.data_objs
                  if d in sim.objs and graph.ref_consumers.get(d)}
    costs = {}
    blocks = {}
    for name in graph.procs:
        obj = sim.objs.get(name)
        if obj is None:         # not built (e.g. a display removed for profiling)
            continue
        dev = devices.get(name, -1)
        kind = f'gpu:{gpu_model}' if dev >= 0 else 'cpu'
        ph = {phase: totals.get((name, phase), 0.0) / n_iter for phase in HOST_PHASES}
        g = totals.get((name, 'trigger_gpu'), 0.0) / n_iter if dev >= 0 else 0.0
        c = Cost(h_in=ph['inputs'], h_prep=ph['prepare_trigger'],
                 h_trig=max(ph['trigger'] - g, 0.0), h_post=ph['post_trigger'],
                 h_send=ph['send_outputs'], g=g, active=triggers.get(name, 0) / n_iter)
        for out_name, value in obj.outputs.items():
            c.outputs[out_name] = float(sum(array_blocks(value, stop).values()))
            c.mpi_outputs[out_name] = _mpi_bytes(value, c.outputs[out_name])
        b = array_blocks(obj, stop)
        # Arrays of the referenced data objects are counted with them
        for d in graph.data_refs[name]:
            for k in ref_blocks.get(d, {}):
                b.pop(k, None)
        prefix = 'g' if dev >= 0 else 'h'
        blocks[name] = {k: v for k, v in b.items() if k[0] == prefix}
        c._used = obj.gpu_bytes_used if dev >= 0 else 0
        if dev >= 0:
            c.copy_gpu, c.copy_cpu = _transfer_times(obj, dev)
        if dev >= 0 and getattr(obj, 'cuda_graph', None) is not None:
            c.graph = True
            c.syncs = 'post'        # BaseProcessingObj.post_trigger() waits for the graph
        costs[name] = (kind, c)

    owners = {}
    for name, b in blocks.items():
        for k, size in b.items():
            owners.setdefault(k, (size, []))[1].append(name)
    shared_by_set = {}
    for k, (size, names) in owners.items():
        if len(names) > 1:
            key = tuple(sorted(names))
            shared_by_set[key] = shared_by_set.get(key, 0) + size
    for name, (kind, c) in costs.items():
        private = sum(size for k, size in blocks[name].items() if len(owners[k][1]) == 1)
        shared_reach = sum(size for k, size in blocks[name].items() if len(owners[k][1]) > 1)
        # gpu_bytes_used also counts referenced arrays allocated while the object was built
        ref_reach = sum(size for d in graph.data_refs[name] for k, size in ref_blocks.get(d, {}).items()
                        if k[0] == 'g')
        c.mem = float(max(c._used - shared_reach - ref_reach, private, 0))
        del c._used
    shared = [{'kind': costs[names[0]][0], 'owners': list(names), 'bytes': float(size)}
              for names, size in shared_by_set.items()]

    refs = {}
    for d in graph.data_objs:
        if d in sim.objs and graph.ref_consumers.get(d):
            refs[d] = {'bytes': float(sum(ref_blocks[d].values())),
                       'mutable': array_checksum(sim.objs[d]) != checksums.get(d)}
    return costs, shared, refs


# ---------------------------------------------------------------------------
# Profile file

class Profile:
    '''
    Measured costs, kept in a JSON file between runs. Entries are indexed by
    the object key (see SimGraph.keys), the device kind and the precision.
    '''

    def __init__(self, path=None):
        self.path = path
        self.entries = {}
        self.refs = {}
        self.shared = {}
        self.calibration = {}
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if data.get('version') == PROFILE_VERSION:
                self.entries = data['entries']
                self.refs = data['refs']
                self.shared = data['shared']
                self.calibration = data.get('calibration', {})

    def save(self):
        if not self.path:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'version': PROFILE_VERSION, 'entries': self.entries, 'refs': self.refs,
                       'shared': self.shared, 'calibration': self.calibration}, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _key(obj_key, kind, precision):
        return f'{obj_key}|{kind}|p{precision}'

    def add_run(self, graph, result, precision):
        for name, (kind, c) in result.costs.items():
            d = asdict(c)
            d['name'] = name
            self.entries[self._key(graph.keys[name], kind, precision)] = d
        for d, info in result.refs.items():
            self.refs[f'{graph.keys[d]}|p{precision}'] = info
        for s in result.shared:
            owner_keys = sorted(graph.keys[n] for n in s['owners'])
            self.shared['|'.join(owner_keys) + f'|{s["kind"]}|p{precision}'] = s['bytes']

    def cost(self, graph, name, kind, precision):
        d = self.entries.get(self._key(graph.keys[name], kind, precision))
        if d is None:
            return None
        d = {k: v for k, v in d.items() if k != 'name'}
        return Cost.from_dict(d)

    def kinds(self, graph, name, precision):
        prefix = f'{graph.keys[name]}|'
        suffix = f'|p{precision}'
        return [k[len(prefix):-len(suffix)] for k in self.entries
                if k.startswith(prefix) and k.endswith(suffix)]

    def ref(self, graph, name, precision):
        return self.refs.get(f'{graph.keys[name]}|p{precision}')

    def shared_blocks(self, graph, kind, precision):
        '''[(owner names, bytes)] of the arrays shared between objects on *kind* devices'''
        by_key = {}
        for name in graph.procs:
            by_key.setdefault(graph.keys[name], []).append(name)
        out = []
        suffix = f'|{kind}|p{precision}'
        for k, size in self.shared.items():
            if not k.endswith(suffix):
                continue
            keys = k[:-len(suffix)].split('|')
            if all(key in by_key for key in keys):
                names = sorted({n for key in keys for n in by_key[key]})
                out.append((names, size))
        return out


def split_chunk(chunk, groups):
    '''Split a list of objects in two halves, keeping the members of each group together'''
    group_of = {}
    for g in groups:
        for n in g:
            group_of[n] = tuple(g)
    units = []
    seen = set()
    for n in chunk:
        g = group_of.get(n, (n,))
        if g not in seen:
            seen.add(g)
            units.append([m for m in chunk if m in g] or [n])
    half = math.ceil(len(units) / 2)
    return [m for u in units[:half] for m in u], [m for u in units[half:] for m in u]
