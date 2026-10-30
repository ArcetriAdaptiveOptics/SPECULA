'''
Search for the placement of the processing objects.

1. Objects that must stay together (an object and the processing objects it
   references, the users of a data object that some object modifies) form
   groups, placed as a unit.
2. Initial placement with a memory-aware HEFT (Heterogeneous Earliest Finish
   Time, Topcuoglu et al. 2002): groups in decreasing order of their longest
   path to the end of the step, each one on the device where the estimated
   time per step is lowest, among the devices that still have room for it.
3. Local search: move single groups, then swap pairs of groups, while the
   estimated time per step decreases.
4. Process layouts: one process per GPU, or one process per node driving all
   its GPUs; the best one is kept.
'''
import time
from dataclasses import dataclass

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
    '''

    def __init__(self, graph, machine, estimator, groups, pins, search_time=60.0, log=None,
                 default_device=0):
        self.graph = graph
        self.default_device = default_device
        self.machine = machine
        self.est = estimator
        self.groups = groups
        self.pins = pins
        self.search_time = search_time
        self.log = log or (lambda msg: None)
        self.n_evals = 0

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
            if any(k.runs_on_main_process_only() for k in klasses) and layout.proc_of_slot[s].rank != 0:
                continue
            slots.append(s)
        if not slots:
            raise Infeasible(f'No device can run {group}: check the pinned placements and the '
                             'CPU-only and main-process-only classes')
        return slots

    # -- evaluation ----------------------------------------------------

    def evaluate(self, assign, layout):
        self.n_evals += 1
        return self.est.evaluate(layout.placement(assign))

    def fits(self, assign, layout):
        mem = self.est.memory(layout.placement(assign))
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
        try:
            assign = self._heft(order, eligible, layout)
        except Infeasible as e:
            self.log(f'HEFT placement failed ({e}), trying a placement by memory first')
            assign = self._first_fit(eligible, layout)
        assign = self._local_search(assign, eligible, layout)
        return Solution(assign, layout, self.evaluate(assign, layout))

    def _reference_slot(self):
        gpus = self.machine.gpu_slots()
        return gpus[0] if gpus else self.machine.nodes[0].cpu_slot

    def _heft(self, order, eligible, layout):
        assign = {}
        for g in order:
            best = None
            for s in eligible[g]:
                trial = dict(assign)
                trial.update({n: s for n in g})
                if not self.fits(trial, layout):
                    continue
                ev = self.evaluate(trial, layout)
                key = (ev.step_ns, ev.comm_bytes, s)
                if best is None or key < best[0]:
                    best = (key, s)
            if best is None:
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

    def _local_search(self, assign, eligible, layout):
        deadline = time.monotonic() + self.search_time
        cur = self.evaluate(assign, layout)
        improved = True
        while improved and time.monotonic() < deadline:
            improved = False
            # Moves: each group to its best device
            for g in self.groups:
                here = assign[g[0]]
                for s in eligible[g]:
                    if s == here:
                        continue
                    trial = dict(assign)
                    trial.update({n: s for n in g})
                    if not self.fits(trial, layout):
                        continue
                    ev = self.evaluate(trial, layout)
                    if ev.key() < cur.key() and ev.step_ns < cur.step_ns * 0.999 + 1:
                        assign, cur, here, improved = trial, ev, s, True
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
                    trial = dict(assign)
                    trial.update({n: s2 for n in g1})
                    trial.update({n: s1 for n in g2})
                    if not self.fits(trial, layout):
                        continue
                    ev = self.evaluate(trial, layout)
                    if ev.step_ns < cur.step_ns * 0.999:
                        assign, cur, improved = trial, ev, True
                        break
                if improved or time.monotonic() > deadline:
                    break
        return assign
