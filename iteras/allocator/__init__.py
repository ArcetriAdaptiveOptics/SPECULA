'''
Automatic placement of the objects of a simulation on nodes, processes,
GPUs and CPUs.

A section of the parameter file whose class is Architecture (see
iteras.data_objects.architecture) describes the resources. Simul then calls
Allocator.allocate() before building the objects:

1. the costs of the objects (time per step, memory, size of the outputs) are
   read from the profile file, or measured by running the simulation for a
   few steps on a local GPU (see profiler);
2. the solver searches the placement with the lowest estimated time per
   step that fits in memory (see solver and cost);
3. the placement is written to the allocation file as overrides
   (target_rank, target_device_idx), with a report, and applied to the
   parameters. Later runs read it back, unless a new allocation is requested;
   if the simulation or the architecture changed, they stop with an error.
'''
import os
import json
import datetime
import importlib.util

import yaml

import iteras
from iteras.log import get_iteras_logger
from iteras.data_objects.architecture import Architecture
from .model import SimGraph, _hash
from .machine import Machine, GiB
from .profiler import Profile, profile_run, OutOfDeviceMemory, split_chunk
from .cost import Estimator, Proc
from .solver import Solver, Layout, Infeasible, Solution, build_groups

META_PREFIX = '# meta: '


class StaleAllocation(ValueError):
    pass


class Allocator:
    '''
    Parameters
    ----------
    simul : iteras.simul.Simul
    arch_pars : dict
        Parameters of the Architecture section (with or without "class").
    plan_only : bool
        Only compute and write the allocation (also set by mode: plan).
    recompute : bool
        Ignore a saved allocation (also set by recompute: true).
    accelerate : bool
        The objects run accelerated (see iteras.accelerator): they are profiled this way,
        and their costs are kept apart from those measured without accelerator.
    '''

    def __init__(self, simul, arch_pars, plan_only=False, recompute=False, accelerate=False):
        self.simul = simul
        self.logger = get_iteras_logger(__name__)
        self.arch = Architecture(**{k: v for k, v in arch_pars.items() if k != 'class'})
        self.plan_only = plan_only or self.arch.mode == 'plan'
        self.recompute = recompute or self.arch.recompute
        self.precision = iteras.global_precision
        self.accelerate = accelerate
        self.allocation = None
        self.report = ''

    @property
    def cost_tag(self):
        '''Precision and accelerate mode of the costs, part of their key in the profile file'''
        return f'{self.precision}a' if self.accelerate else self.precision

    # -- entry point -----------------------------------------------------

    def allocate(self, params):
        # The Accelerator section stays in *params* for the profiling runs, not in the graph
        accelerator = set(self.simul.accelerator_keys(params))
        graph = SimGraph({k: v for k, v in params.items() if k not in accelerator}, self.simul)
        comm = iteras.process_comm
        if comm is not None and comm.Get_size() > 1:
            allocation = None
            error = None
            if iteras.process_rank == 0:
                try:
                    allocation = self._load_or_compute(graph, params, n_ranks=comm.Get_size())
                except Exception as e:
                    error = e
            allocation, failed = comm.bcast((allocation, error is not None), root=0)
            if error is not None:
                raise error
            if failed:
                raise RuntimeError('The allocation failed on rank 0')
        else:
            allocation = self._load_or_compute(graph, params, n_ranks=1)
        self.allocation = allocation
        apply_allocation(allocation, params)
        return allocation

    def _load_or_compute(self, graph, params, n_ranks):
        path = self.arch.allocation_file
        meta = self._meta(graph)
        if os.path.exists(path) and not self.recompute:
            allocation = read_allocation(path)
            check_meta(allocation['meta'], meta, path)
            self.logger.info(f'Using the allocation in {path}')
        else:
            allocation, self.report = self.compute(graph, params)
            allocation['meta'] = meta
            write_allocation(path, allocation, self.report)
            self.logger.info(f'Allocation written to {path}:\n{self.report}')
        if not self.plan_only and allocation['n_ranks'] != n_ranks and \
                not (allocation['n_ranks'] == 1 and n_ranks == 1):
            raise ValueError(f'The allocation in {path} needs {allocation["n_ranks"]} MPI ranks, '
                             f'but {n_ranks} were launched: use mpirun -n {allocation["n_ranks"]} '
                             '(and --mpi)')
        return allocation

    def _kept_pins(self, graph):
        '''
        Objects placed in the parameter files that stay there: all of them, or
        with override_placement only those that split their work over several
        devices (their placement on a node is chosen by hand).
        '''
        return {name: pin for name, pin in graph.pinned.items()
                if not self.arch.override_placement or graph.spread[name] > 1}

    def _meta(self, graph):
        return {'signature': graph.signature(),
                'pins': {name: list(pin) for name, pin in sorted(self._kept_pins(graph).items())},
                'architecture': self.arch.as_dict(),
                'precision': self.precision,
                'accelerate': self.accelerate}

    # -- computation -----------------------------------------------------

    def compute(self, graph, params):
        machine = Machine.from_architecture(self.arch)
        self.logger.info(f'Allocator: resources\n{machine.describe()}')
        profile = Profile(self.arch.profile_file)
        self._ensure_profile(graph, params, profile, machine)
        try:
            machine_used, est, solution, by_nodes = self._solve_nodes(graph, machine, profile)
        except Infeasible as e:
            if iteras.cp is None or self._has_cpu_costs(graph, profile):
                raise
            self.logger.info(f'Allocator: {e}. Measuring candidate objects on the CPU')
            self._profile_cpu_candidates(graph, params, profile, machine,
                                         self.estimator(graph, machine, profile))
            machine_used, est, solution, by_nodes = self._solve_nodes(graph, machine, profile)
        allocation = make_allocation(graph, machine_used, solution)
        report = make_report(graph, machine_used, est, solution, profile, self, allocation, by_nodes)
        return allocation, report

    def _solve_nodes(self, graph, machine, profile):
        '''
        Best placement on *machine*. With several identical nodes, the placement is
        searched on the first k nodes for each k, and the fastest one is kept (the
        one with fewest nodes, if within node_tolerance of the fastest): the search
        on all the nodes alone may spread the objects more than useful.
        Return (machine used, estimator, solution, [(k, time per step)]).
        '''
        n = len(machine.nodes)
        if n == 1 or not machine.identical_nodes():
            est = self.estimator(graph, machine, profile)
            return machine, est, self._solve(graph, machine, est, profile), []
        counts = list(range(1, n + 1)) if n <= 8 else \
            sorted({2 ** i for i in range(n.bit_length()) if 2 ** i <= n} | {n})
        results = []
        for k in counts:
            sub = machine.first_nodes(k)
            est = self.estimator(graph, sub, profile)
            try:
                sol = self._solve(graph, sub, est, profile)
            except (Infeasible, ValueError) as e:      # does not fit, or pinned to a further node
                self.logger.info(f'Allocator: {k} node(s): {e}')
                if k == n and not results:
                    raise
                continue
            self.logger.info(f'Allocator: {k} node(s): {sol.evaluation.step_ns / 1e6:.2f} ms per step')
            results.append((k, sub, est, sol))
        if not results:
            raise Infeasible('No number of nodes fits in memory')
        fastest = min(r[3].evaluation.step_ns for r in results)
        limit = (1 + self.arch.node_tolerance) * fastest
        k, sub, est, sol = min((r for r in results if r[3].evaluation.step_ns <= limit),
                               key=lambda r: r[0])
        return sub, est, sol, [(r[0], r[3].evaluation.step_ns) for r in results]

    def estimator(self, graph, machine, profile):
        '''Estimator built from the measured costs, for the slot kinds of *machine*'''
        costs, refs, shared = self._tables(graph, machine, profile)
        calib = profile.calibration.get(self._calib_key(graph))
        est = Estimator(graph, machine, costs, refs, shared)
        if calib:
            predicted = self._predict_profile_run(graph, machine, profile, calib)
            est.overhead_ns = max(calib['step_ns'] - predicted, 0.0)
            est.calibration = (predicted, calib['step_ns'])
        return est

    def _tables(self, graph, machine, profile):
        '''Costs, referenced data sizes and shared arrays for the slot kinds of *machine*'''
        costs = {}
        kinds = {s.kind: s for s in machine.slots}
        for name in graph.procs:
            measured = profile.kinds(graph, name, self.cost_tag)
            gpu_ref = next((k for k in measured if k.startswith('gpu:')), None)
            for kind, slot in kinds.items():
                c = profile.cost(graph, name, kind, self.cost_tag)
                if c is None and slot.is_gpu:
                    # The same model, named differently (e.g. 'A100' and 'NVIDIA A100-SXM-64GB')
                    same = next((k for k in measured if k.startswith('gpu:')
                                 and same_gpu_model(k[4:], slot.model)), None)
                    if same is not None:
                        c = profile.cost(graph, name, same, self.cost_tag)
                if c is None and slot.is_gpu and gpu_ref is not None:
                    # GPU model not profiled: scale the GPU time of the profiled one
                    c = profile.cost(graph, name, gpu_ref, self.cost_tag)
                    c.g /= slot.speed
                    c.measured = False
                if c is not None:
                    costs[(name, kind)] = c
        refs = {}
        for d in graph.data_objs:
            info = profile.ref(graph, d, self.cost_tag)
            if info is not None:
                refs[d] = info['bytes']
        shared = {}
        for kind, slot in kinds.items():
            src = kind
            if slot.is_gpu and not profile.shared_blocks(graph, kind, self.cost_tag):
                src = next((k for k in self._gpu_kinds(graph, profile)), kind)
            shared[kind] = profile.shared_blocks(graph, src, self.cost_tag)
        return costs, refs, shared

    def _gpu_kinds(self, graph, profile):
        kinds = set()
        for name in graph.procs:
            kinds.update(k for k in profile.kinds(graph, name, self.cost_tag) if k.startswith('gpu:'))
        return sorted(kinds)

    def _calib_key(self, graph):
        return _hash([graph.keys[n] for n in graph.procs] + [self.cost_tag])

    def _predict_profile_run(self, graph, machine, profile, calib):
        '''Estimated step time of the profiling run (all objects in one process, one GPU)'''
        kind = calib['kind']
        model = kind[4:] if kind.startswith('gpu:') else ''
        m = Machine([{'name': 'profile', 'cpu': {'cores': 1, 'memory_gb': 1e6},
                      'gpus': {'count': 1, 'memory_gb': 1e6, 'model': model}}],
                    machine.inter_node, 0, 0)
        m.nodes[0].links = machine.nodes[0].links
        e = Estimator(graph, m, *self._tables(graph, m, profile))
        p = Proc(0, 0)
        placement = {n: (p, 0 if dev >= 0 else 1) for n, dev in calib['devices'].items()
                     if n in graph.position}
        return e.step_time(placement)[0]

    def _solve(self, graph, machine, est, profile):
        mutable = [d for d in graph.data_objs
                   if (profile.ref(graph, d, self.cost_tag) or {}).get('mutable')]
        if mutable:
            self.logger.info(f'Allocator: data objects modified during the run, their users stay '
                             f'together: {mutable}')
        groups = build_groups(graph, mutable)
        policies = self._policies(machine)
        best = None
        for policy in policies:
            layout = Layout(machine, policy)
            pins = self._pins(graph, machine, layout)
            solver = Solver(graph, machine, est, groups, pins, search_time=self.arch.search_time,
                            log=self.logger.info, default_device=self._default_device())
            try:
                sol = solver.solve(layout)
            except Infeasible as e:
                self.logger.info(f'Allocator: layout {policy}: {e}')
                if len(policies) == 1:
                    raise
                continue
            self.logger.info(f'Allocator: layout {policy}: {sol.evaluation.step_ns / 1e6:.2f} ms '
                             f'per step ({solver.n_evals} evaluations)')
            if best is None or sol.evaluation.key() < best.evaluation.key():
                best = sol
        if best is None:
            raise Infeasible('No layout fits in memory')
        return self._prefer_one_gpu(graph, machine, est, best)

    def _prefer_one_gpu(self, graph, machine, est, best):
        '''
        The placement of all the objects on one GPU of the first node, except those
        that must run elsewhere, if it is feasible and its estimated time per step
        is within device_tolerance of that of *best*; otherwise *best*.
        '''
        node = machine.nodes[0]
        if len(node.gpu_slots) < 2 or len(set(best.assign.values()) & set(machine.gpu_slots())) < 2:
            return best
        layout = Layout(machine, 'single')
        try:
            pins = self._pins(graph, machine, layout)
        except ValueError:
            return best
        candidates = []
        for gpu in node.gpu_slots:
            assign = {}
            for name in graph.procs:
                if name in pins:
                    assign[name] = pins[name]
                elif graph.cls(name).runs_on_cpu_only():
                    assign[name] = node.cpu_slot
                elif not graph.device_settable[name]:
                    assign[name] = best.assign[name]
                else:
                    assign[name] = gpu
            evaluation = est.evaluate(layout.placement(assign))
            if evaluation.feasible:
                candidates.append(Solution(assign, layout, evaluation))
        if not candidates:
            return best
        one = min(candidates, key=lambda s: s.evaluation.step_ns)
        if one.evaluation.step_ns <= (1 + self.arch.device_tolerance) * best.evaluation.step_ns:
            self.logger.info(f'Allocator: all objects on one GPU, {one.evaluation.step_ns / 1e6:.2f} ms '
                             f'per step, preferred to {best.evaluation.step_ns / 1e6:.2f} ms '
                             f'(device_tolerance {self.arch.device_tolerance})')
            return one
        return best

    def _policies(self, machine):
        if self.arch.processes != 'auto':
            return [self.arch.processes]
        if len(machine.nodes) > 1:
            return ['per_gpu', 'single']
        policies = ['single']
        if len(machine.gpu_slots()) > 1:
            if self.arch.nodes != 'local' or importlib.util.find_spec('mpi4py') is not None:
                policies.append('per_gpu')
            else:
                self.logger.info('Allocator: mpi4py not available, only a single process is considered')
        return policies

    def _default_device(self):
        '''Device of the objects whose class does not accept target_device_idx'''
        if self.arch.nodes == 'local':
            return max(iteras.default_target_device_idx, -1)
        return 0

    def _pins(self, graph, machine, layout):
        '''
        Slots of the objects placed in the parameter files. Their target_rank
        is mapped to a node with pin_ranks_per_node ranks per node, or, if not
        given, as in *layout*.
        '''
        pins = {}
        per_node = self.arch.pin_ranks_per_node
        if per_node is None:
            per_node = max(len(machine.nodes[0].gpu_slots), 1) if layout.policy == 'per_gpu' else 1
        for name, (rank, dev) in self._kept_pins(graph).items():
            node = (rank or 0) // per_node
            if node >= len(machine.nodes):
                raise ValueError(f'Object {name} is placed on rank {rank}, which does not exist '
                                 'in this architecture')
            n = machine.nodes[node]
            if dev is None:
                dev = 0 if n.gpu_slots else -1
            if dev < 0:
                pins[name] = n.cpu_slot
            elif dev < len(n.gpu_slots):
                pins[name] = n.gpu_slots[dev]
            else:
                raise ValueError(f'Object {name} is placed on GPU {dev}, which does not exist')
        return pins

    # -- profiling -------------------------------------------------------

    def _has_cpu_costs(self, graph, profile):
        return all(profile.cost(graph, n, 'cpu', self.cost_tag) is not None for n in graph.procs)

    def _missing(self, graph, profile, machine):
        '''
        Objects without the costs needed: on the CPU if they cannot use a GPU;
        on the local GPU if its model is one of those of *machine* (so that
        planning on a node of the target computer measures its GPUs, even if
        the profile file holds the costs measured on another GPU model);
        otherwise on any GPU.
        '''
        local = self._local_gpu_model() if self._use_gpu(machine) else None
        if local is not None and not any(same_gpu_model(local, machine.slots[s].model)
                                         for s in machine.gpu_slots()):
            local = None
        missing = []
        for name in graph.procs:
            kinds = profile.kinds(graph, name, self.cost_tag)
            if graph.cls(name).runs_on_cpu_only() or not self._use_gpu(machine):
                if 'cpu' not in kinds:
                    missing.append(name)
            elif local is not None:
                if f'gpu:{local}' not in kinds:
                    missing.append(name)
            elif not any(k.startswith('gpu:') for k in kinds):
                missing.append(name)
        return missing

    def _local_gpu_model(self):
        dev = self._profile_device()
        return iteras.cp.cuda.runtime.getDeviceProperties(dev)['name'].decode()

    @staticmethod
    def _use_gpu(machine):
        return iteras.cp is not None and bool(machine.gpu_slots())

    def _profile_device(self):
        cp = iteras.cp
        if cp is None:
            return -1
        if self.arch.profile_device is not None:
            return self.arch.profile_device
        free = []
        for i in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(i):
                free.append((cp.cuda.runtime.memGetInfo()[0], -i))
        return -max(free)[1]

    def _profile_steps(self, graph):
        '''
        Warm-up and measured steps of the profiling runs. The warm-up goes past
        the largest "start_time" parameter (objects often start working later),
        and at least two periods of the slowest object (largest "dt") are measured.
        '''
        dt = graph.main_params['time_step']
        start, period = 0.0, 1
        for pars in graph.params.values():
            if not isinstance(pars, dict):
                continue
            if isinstance(pars.get('start_time'), (int, float)):
                start = max(start, pars['start_time'])
            if isinstance(pars.get('dt'), (int, float)) and pars['dt'] > 0:
                period = max(period, round(pars['dt'] / dt))
        warmup = max(self.arch.warmup_steps, round(start / dt) + 2)
        steps = max(self.arch.profile_steps, 2 * period)
        return warmup, steps

    def _workdir(self):
        stem = self.arch.allocation_file.rsplit('.', 1)[0]
        return stem + '_profiling'

    def _ensure_profile(self, graph, params, profile, machine):
        missing = self._missing(graph, profile, machine)
        if self.arch.profile_cpu:
            missing += [n for n in graph.procs if profile.cost(graph, n, 'cpu', self.cost_tag) is None]
        if not missing:
            self.logger.info(f'Allocator: costs read from {self.arch.profile_file}')
            return
        dev = self._profile_device() if self._use_gpu(machine) else -1
        gpu_model = ''
        if dev >= 0:
            gpu_model = iteras.cp.cuda.runtime.getDeviceProperties(dev)['name'].decode()
        self.logger.info(f'Allocator: measuring {len(missing)} objects '
                         f'on {"GPU " + str(dev) if dev >= 0 else "the CPU"}')
        if self._missing(graph, profile, machine):
            self._profile_gpu(graph, params, profile, dev, gpu_model)
        if self.arch.profile_cpu and not self._has_cpu_costs(graph, profile):
            devices = {n: -1 for n in graph.procs}
            warmup, steps = self._profile_steps(graph)
            res = profile_run(params, graph, devices, self._workdir(), steps, warmup, async_run=False)
            profile.add_run(graph, res, self.cost_tag)
        profile.save()

    def _profile_gpu(self, graph, params, profile, dev, gpu_model):
        '''All objects on one GPU; if they do not fit, chunks on the GPU and the rest on the CPU'''
        cpu_only = {n for n in graph.procs if graph.cls(n).runs_on_cpu_only() or dev < 0}
        candidates = [n for n in graph.procs if n not in cpu_only]
        groups = build_groups(graph, [])
        pending = [candidates]
        first = True
        warmup, steps = self._profile_steps(graph)
        while pending:
            chunk = pending.pop(0)
            devices = {n: (dev if n in chunk else -1) for n in graph.procs}
            try:
                res = profile_run(params, graph, devices, self._workdir(), steps, warmup,
                                  gpu_model=gpu_model)
            except OutOfDeviceMemory:
                if len(chunk) <= 1:
                    raise MemoryError(f'Allocator: object {chunk} does not fit alone on GPU {dev}')
                a, b = split_chunk(chunk, groups)
                self.logger.info(f'Allocator: the objects do not fit on GPU {dev}: profiling in '
                                 f'chunks of {len(a)} and {len(b)} objects, the rest on the CPU')
                pending = [a, b] + pending
                first = False
                continue
            # Keep the GPU costs of the chunk, and the CPU costs of the others
            res.costs = {n: kc for n, kc in res.costs.items()
                         if n in chunk or n in cpu_only or kc[0] == 'cpu'}
            profile.add_run(graph, res, self.cost_tag)
            if first and len(chunk) == len(candidates):
                profile.calibration[self._calib_key(graph)] = {
                    'step_ns': res.step_ns, 'kind': f'gpu:{gpu_model}' if dev >= 0 else 'cpu',
                    'devices': res.devices}

    def _profile_cpu_candidates(self, graph, params, profile, machine, est):
        '''Measure on the CPU the objects that may have to run there (see cpu_candidates())'''
        chosen = cpu_candidates(graph, machine, est)
        self.logger.info(f'Allocator: measuring on the CPU: {chosen}')
        dev = self._profile_device()
        gpu_model = iteras.cp.cuda.runtime.getDeviceProperties(dev)['name'].decode()
        devices = {n: (-1 if n in chosen or graph.cls(n).runs_on_cpu_only() else dev)
                   for n in graph.procs}
        warmup, steps = self._profile_steps(graph)
        res = profile_run(params, graph, devices, self._workdir(), steps, warmup, gpu_model=gpu_model,
                          async_run=False)
        res.costs = {n: kc for n, kc in res.costs.items() if kc[0] == 'cpu'}
        profile.add_run(graph, res, self.cost_tag)
        profile.save()


def same_gpu_model(a, b):
    '''True if two GPU model names designate the same model, one containing the other'''
    a, b = a.lower().strip(), b.lower().strip()
    return bool(a) and bool(b) and (a in b or b in a)


def cpu_candidates(graph, machine, est):
    '''
    Objects to measure on the CPU when the GPUs are too small: starting from all
    the objects on one GPU, they are moved to the CPU one at a time, each time
    the one that frees most GPU memory (including the data objects only it uses)
    per unit of GPU time, until what is left fits in the memory of all the GPUs,
    with 20% to spare.
    '''
    gpus = machine.gpu_slots()
    ref = gpus[0]
    capacity = sum(machine.slots[s].memory for s in gpus)
    p = Proc(0, 0)
    # Only the objects left on the GPU: the CPU costs are not known yet
    on_gpu = {n: (p, ref) for n in graph.procs
              if not graph.cls(n).runs_on_cpu_only() and est.cost(n, ref) is not None}
    chosen = []
    used = est.memory(on_gpu).get(ref, 0.0)
    while used > 0.8 * capacity:
        best = None
        for n in on_gpu:
            if not graph.device_settable[n]:
                continue
            rest = {k: v for k, v in on_gpu.items() if k != n}
            left = est.memory(rest).get(ref, 0.0)
            ratio = (used - left) / (est.cost(n, ref).g + 1e5)
            if best is None or ratio > best[0]:
                best = (ratio, n, left)
        if best is None:
            break
        _, n, used = best
        del on_gpu[n]
        chosen.append(n)
    return chosen


# ---------------------------------------------------------------------------
# Allocation: placement of every object, as overrides

def _node_signature(machine, n):
    node = machine.nodes[n]
    return [(machine.slots[s].kind, machine.slots[s].memory) for s in node.gpu_slots + [node.cpu_slot]]


def compact_processes(machine, layout, placement):
    '''
    Processes to launch: only the nodes that host objects are used. When all the
    nodes are identical, the used ones are renumbered from 0, otherwise the nodes
    up to the last used one are kept. Return {old rank: new rank}, {old node: new node}.
    '''
    used = sorted({p.node for p, _ in placement.values()})
    same = all(_node_signature(machine, n) == _node_signature(machine, 0) for n in range(len(machine.nodes)))
    nodes = used if same else list(range(used[-1] + 1))
    node_map = {n: i for i, n in enumerate(nodes)}
    procs = [p for p in layout.procs if p.node in node_map]
    procs.sort(key=lambda p: (node_map[p.node], p.rank))
    return {p.rank: i for i, p in enumerate(procs)}, node_map


def make_allocation(graph, machine, solution):
    layout = solution.layout
    placement = solution.placement()
    rank_map, node_map = compact_processes(machine, layout, placement)
    multi = len(rank_map) > 1
    objects = {}
    for name in graph.procs:
        if name not in placement:
            continue
        p, s = placement[name]
        objects[name] = {'target_device_idx': machine.slots[s].idx} if graph.device_settable[name] else {}
        if multi:
            objects[name]['target_rank'] = rank_map[p.rank]
    data = {}
    devices = {}        # data object -> {rank: device}
    for d in graph.data_objs:
        for p, s in (placement[c] for c in graph.ref_consumers.get(d, []) if c in placement):
            devices.setdefault(d, {}).setdefault(rank_map[p.rank], machine.slots[s].idx)
    changed = True
    while changed:
        changed = False
        for d, refs in graph.data_data_refs.items():
            for r in refs:
                for rank, dev in devices.get(d, {}).items():
                    if rank not in devices.setdefault(r, {}):
                        devices[r][rank] = dev
                        changed = True
    for d, per_rank in devices.items():
        if not per_rank:
            continue
        if multi:
            ranks = sorted(per_rank)
            data[d] = {'target_rank': ranks if len(ranks) > 1 else ranks[0],
                       'target_device_idx': per_rank if len(ranks) > 1 else per_rank[ranks[0]]}
        else:
            data[d] = {'target_device_idx': per_rank[0]}
    procs = [{'rank': rank_map[p.rank], 'node': node_map[p.node],
              'devices': sorted({machine.slots[s].idx for q, s in placement.values() if q == p})}
             for p in layout.procs if p.rank in rank_map]
    procs.sort(key=lambda p: p['rank'])
    ev = solution.evaluation
    return {'objects': objects, 'data_objects': data, 'n_ranks': len(rank_map) if multi else 1,
            'n_nodes': len(node_map), 'layout': layout.policy, 'processes': procs,
            'estimated_step_ms': ev.step_ns / 1e6}


def apply_allocation(allocation, params):
    for name, place in list(allocation['objects'].items()) + list(allocation['data_objects'].items()):
        if name in params:
            params[name].update(place)


def write_allocation(path, allocation, report):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = {k: allocation[k] for k in ('meta', 'n_ranks', 'n_nodes', 'layout', 'processes',
                                       'estimated_step_ms')}
    lines = [f'# Allocation written by the iteras allocator on '
             f'{datetime.datetime.now().isoformat(timespec="seconds")}.',
             '# It can be passed as an additional parameter file. Recompute it with --reallocate.',
             f'# Layout: {allocation["layout"]}, nodes: {allocation["n_nodes"]}, '
             f'MPI ranks: {allocation["n_ranks"]}, '
             f'estimated time per step: {allocation["estimated_step_ms"]:.2f} ms',
             META_PREFIX + json.dumps(meta, sort_keys=True), '']
    body = {}
    for name, place in list(allocation['objects'].items()) + list(allocation['data_objects'].items()):
        if place:
            body[f'{name}_override'] = place
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
        yaml.safe_dump(body, f, sort_keys=False, default_flow_style=None)
    if report:
        with open(path.rsplit('.', 1)[0] + '_report.txt', 'w') as f:
            f.write(report)


def read_allocation(path):
    meta = None
    with open(path) as f:
        text = f.read()
    for line in text.splitlines():
        if line.startswith(META_PREFIX):
            meta = json.loads(line[len(META_PREFIX):])
            break
    if meta is None:
        raise ValueError(f'{path} is not an allocation file written by the allocator')
    body = yaml.safe_load(text) or {}
    allocation = dict(meta)
    allocation['objects'] = {key[:-len('_override')]: place for key, place in body.items()}
    allocation['data_objects'] = {}
    return allocation


def check_meta(saved, current, path):
    '''Raise StaleAllocation, naming what changed, if *saved* does not match *current*'''
    problems = []
    s_sig, c_sig = saved['signature'], current['signature']
    added = sorted(set(c_sig) - set(s_sig))
    removed = sorted(set(s_sig) - set(c_sig))
    changed = sorted(k for k in set(s_sig) & set(c_sig) if s_sig[k] != c_sig[k])
    if added:
        problems.append(f'new objects {added}')
    if removed:
        problems.append(f'removed objects {removed}')
    if changed:
        problems.append(f'changed objects {changed}')
    if saved['pins'] != current['pins']:
        problems.append('different target_rank/target_device_idx in the parameter files')
    if saved['architecture'] != current['architecture']:
        problems.append('different architecture')
    if saved['precision'] != current['precision']:
        problems.append('different precision')
    if saved.get('accelerate', False) != current.get('accelerate', False):
        problems.append('accelerator ' + ('disabled' if saved.get('accelerate') else 'enabled'))
    if problems:
        raise StaleAllocation(f'The allocation in {path} was computed for a different simulation: '
                              + '; '.join(problems) + '. Recompute it with --reallocate '
                              '(or recompute: true in the Architecture section)')


from .report import make_report     # noqa: E402
