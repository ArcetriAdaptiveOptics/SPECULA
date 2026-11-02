'''
Search for the placement of the processing objects.

1. Objects that must stay together (an object and the processing objects it
   references, the users of a data object that some object modifies) form
   groups, placed as a unit.
2. Initial placement with a memory-aware HEFT (Heterogeneous Earliest Finish
   Time, Topcuoglu et al. 2002): groups in decreasing order of their longest
   path to the end of the step, each one on the device where the estimated
   time per step is lowest, among the devices that still have room for it.
3. Local search: move single groups, split the work of objects over several
   GPUs (see BaseProcessingObj.split_options()) or join it again, then swap
   pairs of groups, while the estimated time per step decreases.
4. Process layouts: one process per GPU, or one process per node driving all
   its GPUs; the best one is kept. An object split over several GPUs needs a
   process that drives all of them.
'''
import time
from dataclasses import dataclass, field

from iteras.lib.utils import import_class
from .cost import Proc


class Infeasible(Exception):
    '''No placement fits in memory'''
    def __init__(self, msg, group=None, need=0.0):
        super().__init__(msg)
        self.group = group
        self.need = need


class Layout:
    '''
    Processes of a run and the devices they drive.

    policy 'per_gpu': one process per GPU; the CPU of a node belongs to the
    process of its first GPU. 'single': one process per node, driving all its
    GPUs (no MPI with a single node). Ranks are numbered node by node.
    '''

    def __init__(self, machine, policy):
        self.policy = policy
        self.procs = []
        self.proc_of_slot = {}
        for n, node in enumerate(machine.nodes):
            if policy == 'single' or not node.gpu_slots:
                p = Proc(len(self.procs), n)
                self.procs.append(p)
                for s in node.gpu_slots + [node.cpu_slot]:
                    self.proc_of_slot[s] = p
            else:
                for s in node.gpu_slots:
                    p = Proc(len(self.procs), n)
                    self.procs.append(p)
                    self.proc_of_slot[s] = p
                self.proc_of_slot[node.cpu_slot] = self.proc_of_slot[node.gpu_slots[0]]

    def placement(self, assign):
        return {name: (self.proc_of_slot[s], s) for name, s in assign.items()}


@dataclass
class Solution:
    assign: dict            # object name -> slot
    layout: Layout
    evaluation: object      # cost.Evaluation
    split: dict = field(default_factory=dict)           # object name -> number of devices, if split
    split_params: dict = field(default_factory=dict)    # object name -> parameters of the split object

    def placement(self):
        return self.layout.placement(self.assign)


def build_groups(graph, mutable_refs):
    '''Groups of processing objects that must be on the same device'''
    parent = {n: n for n in graph.procs}

    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb, key=graph.position.get)] = min(ra, rb, key=graph.position.get)

    for name in graph.procs:
        for other in graph.proc_refs[name]:
            union(name, other)
    for d in mutable_refs:
        users = graph.ref_consumers.get(d, [])
        for u in users[1:]:
            union(users[0], u)
    groups = {}
    for name in graph.procs:
        groups.setdefault(find(name), []).append(name)
    return [tuple(g) for g in groups.values()]


class Solver:
    '''
    Parameters
    ----------
    graph, machine, estimator : see model, machine and cost
    groups : list of tuples of object names
    pins : dict
        name -> slot, for objects that must not move
    search_time : float
        Seconds for the local search of each layout
    split_objects : bool
        If True, the objects that offer it (see BaseProcessingObj.split_options())
        may be split over several GPUs of a node. Objects placed in the parameter
        files, and those already split there, keep what is written.
    '''

    def __init__(self, graph, machine, estimator, groups, pins, search_time=60.0, log=None,
                 default_device=0, split_objects=True):
        self.graph = graph
        self.default_device = default_device
        self.machine = machine
        self.est = estimator
        self.groups = groups
        self.pins = pins
        self.search_time = search_time
        self.log = log or (lambda msg: None)
        self.n_evals = 0
        # name -> {number of devices: parameters of the split object}
        self.options = {}
        max_gpus = max((len(node.gpu_slots) for node in machine.nodes), default=0)
        if split_objects and max_gpus > 1:
            for name in graph.procs:
                if graph.spread[name] > 1 or name in pins or not graph.device_settable[name]:
                    continue
                opts = graph.cls(name).split_options(graph.params[name], max_gpus)
                opts = {k: p for k, p in opts.items() if 1 < k <= max_gpus}
                if opts:
                    self.options[name] = opts
                    # The work that the parts repeat, as declared by the class of the split object
                    for k, p in opts.items():
                        klass = import_class(p['class'], graph.main_params.get('add_modules', [])) \
                            if 'class' in p else graph.cls(name)
                        estimator.repeated[(name, k)] = klass.repeated_work_fraction(p)

    # -- constraints ---------------------------------------------------

    def eligible(self, group, layout):
        slots = []
        m = self.machine
        pinned = {self.pins[n] for n in group if n in self.pins}
        if len(pinned) > 1:
            raise Infeasible(f'Objects {group} must be on the same device, but are pinned to '
                             f'different devices')
        for s, slot in enumerate(m.slots):
            if pinned and s not in pinned:
                continue
            if not all(self.graph.device_settable[n] for n in group) and \
                    slot.idx != (self.default_device if m.nodes[slot.node].gpu_slots else -1):
                continue
            if any(self.est.cost(n, s) is None for n in group):
                continue
            klasses = [self.graph.cls(n) for n in group]
            if slot.is_gpu and any(k.runs_on_cpu_only() for k in klasses):
                continue
            if not slot.is_gpu and any(self.graph.spread[n] > 1 for n in group):
                continue
            # An object split in the parameter file uses GPUs that its process must drive
            # (unless placed by hand)
            if any(self.graph.spread[n] > 1 and n not in self.pins and not self.owns(layout, n, s)
                   for n in group):
                continue
            if any(k.runs_on_main_process_only() for k in klasses) and layout.proc_of_slot[s].rank != 0:
                continue
            slots.append(s)
        if not slots:
            raise Infeasible(f'No device can run {group}: check the pinned placements, the '
                             'CPU-only and main-process-only classes, and the objects split over '
                             'several GPUs (they need a process driving all of them)')
        return slots

    def owns(self, layout, name, slot, split=None):
        '''True if the process of *slot* drives all the devices that *name* uses there'''
        proc = layout.proc_of_slot[slot]
        return all(layout.proc_of_slot[d] == proc for d in self.est.devices(name, slot, split))

    def valid_split(self, layout, names, slot, split):
        '''True if the objects *names* placed on *slot* can be split as in *split*'''
        if not any(split.get(n, 1) > 1 for n in names):
            return True
        return self.machine.slots[slot].is_gpu and all(self.owns(layout, n, slot, split) for n in names)

    # -- evaluation ----------------------------------------------------

    def evaluate(self, assign, layout, split=None):
        self.n_evals += 1
        return self.est.evaluate(layout.placement(assign), split)

    def fits(self, assign, layout, split=None):
        mem = self.est.memory(layout.placement(assign), split)
        return all(use <= self.machine.slots[s].memory for s, use in mem.items())

    # -- search --------------------------------------------------------

    def solve(self, layout):
        eligible = {g: self.eligible(g, layout) for g in self.groups}
        ref_slot = self._reference_slot()

        def w(name):
            c = self.est.cost(name, ref_slot) or self.est.cost(name, self.machine.nodes[0].cpu_slot)
            return (c.host + c.g) if c is not None else 0.0
        bw = self.machine.nodes[0].links['gpu_gpu_gb_s']
        rank = self.graph.upward_rank(w, lambda e: self.est.out_bytes(e, ref_slot) / bw)
        order = sorted(self.groups, key=lambda g: (-max(rank[n] for n in g),
                                                   min(self.graph.position[n] for n in g)))
        starts = [{}]
        if self.options:
            # A second start with every object split as much as it can be: splitting a
            # single object often does not lower the time per step while the others
            # are not split (the busiest GPU still has two of them), so the search
            # from the first start alone may not get there
            starts.append({n: max(opts) for n, opts in self.options.items()})
        best = None
        for split in starts:
            start = 'all splittable objects split' if split else 'no object split'
            try:
                assign = self._heft(order, eligible, layout, split)
            except Infeasible as e:
                if split:
                    self.log(f'Layout {layout.policy}: no start with {start} ({e})')
                    continue
                self.log(f'HEFT placement failed ({e}), trying a placement by memory first')
                assign = self._first_fit(eligible, layout)
            assign, split = self._local_search(assign, eligible, layout, split,
                                               self.search_time / len(starts))
            sol = Solution(assign, layout, self.evaluate(assign, layout, split), split,
                           {n: self.options[n][k] for n, k in split.items()})
            self.log(f'Layout {layout.policy}, start with {start}: '
                     f'{sol.evaluation.step_ns / 1e6:.2f} ms per step, {len(split)} objects split')
            if best is None or sol.evaluation.key() < best.evaluation.key():
                best = sol
        return best

    def _reference_slot(self):
        gpus = self.machine.gpu_slots()
        return gpus[0] if gpus else self.machine.nodes[0].cpu_slot

    def _heft(self, order, eligible, layout, split=None):
        split = split or {}
        slots = self.machine.slots
        symmetric = len(self.machine.nodes) > 1 and self.machine.identical_nodes()
        assign = {}
        for g in order:
            best = None
            # On identical nodes, a device of a node still empty gives the same time
            # per step as the same device of another empty node: only the first is tried
            used_nodes = {slots[s].node for s in assign.values()}
            tried_empty = set()
            for s in sorted(eligible[g]):
                if symmetric and slots[s].node not in used_nodes:
                    if slots[s].idx in tried_empty:
                        continue
                    tried_empty.add(slots[s].idx)
                if not self.valid_split(layout, g, s, split):
                    continue
                trial = dict(assign)
                trial.update({n: s for n in g})
                if not self.fits(trial, layout, split):
                    continue
                ev = self.evaluate(trial, layout, split)
                key = (ev.step_ns, ev.comm_bytes, s)
                if best is None or key < best[0]:
                    best = (key, s)
            if best is None:
                if not any(self.valid_split(layout, g, s, split) for s in eligible[g]):
                    raise Infeasible(f'{list(g)} split over several GPUs, but no process drives '
                                     'all of them', group=g)
                need = sum(self.est.cost(n, eligible[g][0]).mem for n in g)
                raise Infeasible(f'{list(g)} do not fit in any device', group=g, need=need)
            assign.update({n: best[1] for n in g})
        return assign

    def _first_fit(self, eligible, layout):
        '''Largest groups first, each on the eligible device with most free memory'''
        def group_mem(g):
            return sum(self.est.cost(n, eligible[g][0]).mem for n in g)
        assign = {}
        for g in sorted(self.groups, key=group_mem, reverse=True):
            best = None
            for s in eligible[g]:
                trial = dict(assign)
                trial.update({n: s for n in g})
                mem = self.est.memory(layout.placement(trial))
                if any(use > self.machine.slots[x].memory for x, use in mem.items()):
                    continue
                free = self.machine.slots[s].memory - mem.get(s, 0.0)
                if best is None or free > best[0]:
                    best = (free, s)
            if best is None:
                raise Infeasible(f'{list(g)} do not fit in any device (need about '
                                 f'{group_mem(g) / 2**30:.2f} GiB)', group=g, need=group_mem(g))
            assign.update({n: best[1] for n in g})
        return assign

    def _local_search(self, assign, eligible, layout, split=None, search_time=None):
        '''
        Improve *assign*, with the objects split as in *split*, while the estimated
        time per step decreases, for at most *search_time* seconds (default: the
        search time of the solver). Return the placement and the objects split over
        several GPUs ({name: number of GPUs}).
        '''
        deadline = time.monotonic() + (self.search_time if search_time is None else search_time)
        split = dict(split or {})
        cur = self.evaluate(assign, layout, split)

        def better(ev):
            return ev.key() < cur.key() and ev.step_ns < cur.step_ns * 0.999 + 1

        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            # Moves: each group to its best device
            for g in self.groups:
                here = assign[g[0]]
                for s in eligible[g]:
                    if s == here or not self.valid_split(layout, g, s, split):
                        continue
                    trial = dict(assign)
                    trial.update({n: s for n in g})
                    if not self.fits(trial, layout, split):
                        continue
                    ev = self.evaluate(trial, layout, split)
                    if better(ev):
                        assign, cur, here, improved = trial, ev, s, True
                if time.monotonic() > deadline:
                    break
            if improved or time.monotonic() > deadline:
                continue
            # Splits: an object over another number of GPUs (1: not split), with its
            # group on any device where its process drives all of those GPUs
            for g in self.groups:
                for n in [n for n in g if n in self.options]:
                    for k in [1] + sorted(self.options[n]):
                        if k == split.get(n, 1):
                            continue
                        trial_split = {m: v for m, v in split.items() if m != n}
                        if k > 1:
                            trial_split[n] = k
                        for s in eligible[g]:
                            if not self.valid_split(layout, g, s, trial_split):
                                continue
                            trial = dict(assign)
                            trial.update({m: s for m in g})
                            if not self.fits(trial, layout, trial_split):
                                continue
                            ev = self.evaluate(trial, layout, trial_split)
                            if better(ev):
                                assign, split, cur, improved = trial, trial_split, ev, True
                    if time.monotonic() > deadline:
                        break
                if time.monotonic() > deadline:
                    break
            if improved or time.monotonic() > deadline:
                continue
            # Swaps of two groups on different devices
            for i, g1 in enumerate(self.groups):
                for g2 in self.groups[i + 1:]:
                    s1, s2 = assign[g1[0]], assign[g2[0]]
                    if s1 == s2 or s2 not in eligible[g1] or s1 not in eligible[g2]:
                        continue
                    if not (self.valid_split(layout, g1, s2, split)
                            and self.valid_split(layout, g2, s1, split)):
                        continue
                    trial = dict(assign)
                    trial.update({n: s2 for n in g1})
                    trial.update({n: s1 for n in g2})
                    if not self.fits(trial, layout, split):
                        continue
                    ev = self.evaluate(trial, layout, split)
                    if ev.step_ns < cur.step_ns * 0.999:
                        assign, cur, improved = trial, ev, True
                        break
                if improved or time.monotonic() > deadline:
                    break
        return assign, split
