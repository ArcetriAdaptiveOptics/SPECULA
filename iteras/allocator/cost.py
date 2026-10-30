'''
Memory and time per step of a placement.

A placement maps each processing object to (process, slot), where a slot is
a device of a node (see machine.Slot). The time per step is estimated by
simulating a few steps of the loop as LoopControl runs it:

* each process walks the trigger levels in order, with only its objects; for
  each level it gathers the inputs of all its objects (waiting for messages
  from other processes, copying data between its devices), then triggers
  them, then runs post_trigger and sends the outputs to other processes;
* the host work of a process is sequential; the GPU work of each device is
  a sequential stream, which runs while the host goes on, except for objects
  that wait for the GPU in trigger or post_trigger (see profiler) and before
  copying data out of a GPU;
* an object that splits its work over k devices (graph.spread, e.g.
  DistributedSH) launches k parts, each with 1/k of the measured GPU time and
  memory, copies its inputs to its other devices and gathers the results;
* MPI messages are sent as iteras.mpi_progress sends them. A small message
  (up to EAGER_BYTES) travels at once. The data of a larger one moves once
  both the send and the receive have been posted: on a single-copy link the
  receiver pulls it, otherwise it moves only while the sender is inside MPI
  (sending, receiving, or polling while it waits for its GPU).

The estimate is taken in steady state, so that a process running ahead of
the others (allowed by the asynchronous MPI sends) is accounted for.
'''
import bisect
from collections import defaultdict
from dataclasses import dataclass

from iteras import mpi_progress

MPI_BUFFER_BYTES = 40e6     # attached MPI buffer of each process (see iteras.__init__)
EAGER_BYTES = 64 * 1024     # larger MPI messages use a rendezvous
MAX_PASSES = 10             # simulation passes to settle the MPI progress of large messages


@dataclass(frozen=True)
class Proc:
    rank: int
    node: int


@dataclass
class Evaluation:
    step_ns: float
    comm_bytes: float
    memory: dict            # slot -> bytes
    feasible: bool
    overflow: dict          # slot -> bytes over capacity

    def key(self):
        return (not self.feasible, self.step_ns, self.comm_bytes)


class Estimator:
    '''
    Parameters
    ----------
    graph : model.SimGraph
    machine : machine.Machine
    costs : dict
        {(name, kind): Cost} for every object and every slot kind it may use.
    refs : dict
        {data object name: bytes}
    shared : dict
        {kind: [(owner names, bytes)]}, arrays shared by objects on one device.
    overhead_ns : float
        Loop overhead per step and per process, not attributed to any object.
    steps : int
        Number of simulated steps; the second half is used.
    '''

    def __init__(self, graph, machine, costs, refs, shared, overhead_ns=0.0, steps=8):
        self.graph = graph
        self.machine = machine
        self.costs = costs
        self.refs = refs
        self.shared = shared
        self.overhead_ns = overhead_ns
        self.steps = steps
        levels = defaultdict(list)
        for name in graph.procs:
            levels[graph.level[name]].append(name)
        self.levels = [levels[k] for k in sorted(levels)]
        self.ref_consumers = {d: graph.ref_consumers[d] for d in refs}
        # The MPI behavior of the framework (see iteras.mpi_progress)
        self.buffered_sends = mpi_progress.buffered_sends()
        self.polling = mpi_progress.polling_enabled() and not self.buffered_sends

    def cost(self, name, slot):
        return self.costs.get((name, self.machine.slots[slot].kind))

    def devices(self, name, slot):
        '''
        Slots used by object *name* placed on *slot*: the following GPUs of the
        node too (modulo their number) for objects that split their work.
        '''
        k = self.graph.spread.get(name, 1)
        if k == 1 or not self.machine.slots[slot].is_gpu:
            return [slot]
        gpus = self.machine.nodes[self.machine.slots[slot].node].gpu_slots
        first = gpus.index(slot)
        return [gpus[(first + i) % len(gpus)] for i in range(k)]

    def out_bytes(self, edge, slot):
        '''Bytes of the output of *edge*, copied between devices'''
        c = self.cost(edge.src, slot)
        return c.outputs.get(edge.out, 0.0) if c is not None else 0.0

    def copy_host_ns(self, edge, src, dst):
        '''
        Host time to copy the output of *edge* from slot *src* to slot *dst* of
        the same process: measured for each output by the profiler (to another
        GPU, to the CPU), otherwise estimated from its size.
        '''
        c = self.cost(edge.src, src)
        slots = self.machine.slots
        measured = None
        if c is not None and slots[src].is_gpu:
            measured = (c.copy_gpu if slots[dst].is_gpu else c.copy_cpu).get(edge.out)
        if measured is not None:
            return measured
        return self.machine.copy_ns(src, dst, self.out_bytes(edge, src))

    def mpi_bytes(self, edge, slot):
        '''Bytes of the output of *edge*, sent by MPI'''
        c = self.cost(edge.src, slot)
        if c is None:
            return 0.0
        return c.mpi_outputs.get(edge.out, c.outputs.get(edge.out, 0.0))

    # -- memory --------------------------------------------------------

    def memory(self, placement):
        '''
        Bytes used on each slot. Each process has its own copy of the data
        objects, of the arrays shared by its objects and of the inputs coming
        from other devices or processes.
        '''
        slots = self.machine.slots
        use = defaultdict(float)
        on_proc_slot = defaultdict(set)
        used = {name: self.devices(name, s) for name, (_, s) in placement.items()}
        for name, (p, s) in placement.items():
            mem = self.cost(name, s).mem
            for d in used[name]:
                use[d] += mem / len(used[name])
                on_proc_slot[(p, d)].add(name)
        for (p, s), names in on_proc_slot.items():
            for owners, size in self.shared.get(slots[s].kind, []):
                if names.intersection(owners):
                    use[s] += size
            if slots[s].is_gpu:
                use[s] += self.machine.context_overhead
        for d, consumers in self.ref_consumers.items():
            for p, slot in {(placement[c][0], x) for c in consumers if c in placement
                            for x in used[c]}:
                use[slot] += self.refs[d]
        clones = set()
        for name, (p, s) in placement.items():
            for e in self.graph.in_edges[name]:
                if e.src not in placement:
                    continue
                ps, ss = placement[e.src]
                for d in used[name]:
                    if (ps, ss) != (p, d) and (e.src, e.out, p, d) not in clones:
                        clones.add((e.src, e.out, p, d))
                        use[d] += self.out_bytes(e, ss)
                if ps != p:
                    # host copies: up to two steps in flight at the sender, one at the receiver
                    nbytes = self.mpi_bytes(e, ss)
                    use[self.machine.nodes[ps.node].cpu_slot] += 2 * nbytes
                    if slots[s].is_gpu:
                        use[self.machine.nodes[p.node].cpu_slot] += nbytes
        for p in {p for p, _ in placement.values()}:
            use[self.machine.nodes[p.node].cpu_slot] += MPI_BUFFER_BYTES
        return dict(use)

    # -- time ----------------------------------------------------------

    def step_time(self, placement):
        '''
        Steady-state time per step (ns) and bytes exchanged between devices per step.

        Without single-copy, the data of a large MPI message moves only while
        its sender is inside MPI after both sides are ready. Since those times
        depend on the deliveries, the simulation is repeated with the MPI
        activity of the senders in the previous pass, until the time per step
        settles.
        '''
        remote = any(placement[e.src][0] != placement[e.dst][0] for e in self.graph.edges
                     if e.src in placement and e.dst in placement)
        events = None
        per_step = None
        stable = 0
        for _ in range(MAX_PASSES if remote else 1):
            new_step, comm_bytes, events = self._simulate(placement, events)
            if per_step is not None and abs(new_step - per_step) <= 0.005 * per_step:
                stable += 1
                if stable == 2:
                    per_step = new_step
                    break
            else:
                stable = 0
            per_step = new_step
        return per_step, comm_bytes

    @staticmethod
    def _next_progress(intervals, t):
        '''First time >= t at which a process is inside MPI, or None'''
        i = bisect.bisect_left(intervals, (t, t))
        if i > 0 and intervals[i - 1][1] >= t:
            return t
        return intervals[i][0] if i < len(intervals) else None

    def _delivery(self, sent, t_recv, src_proc, s_src, s_dst, prev_events):
        '''Time when a message sent at *sent* = (time, bytes) is received, if waited for from *t_recv*'''
        m = self.machine
        t_send, nbytes = sent
        bw, lat = m.link(s_src, s_dst)
        if nbytes <= EAGER_BYTES:
            return max(t_recv, t_send + lat + nbytes / bw)
        ready = max(t_send + lat, t_recv)
        if not self.buffered_sends and m.single_copy(s_src, s_dst):
            start = ready
        else:
            start = ready
            if prev_events is not None:
                nxt = self._next_progress(prev_events.get(src_proc, []), ready)
                if nxt is not None:
                    start = nxt
        return start + nbytes / bw

    def _simulate(self, placement, prev_events):
        '''
        One pass of the simulation. *prev_events* are the intervals of each
        process inside MPI in the previous pass, or None. Return the time per
        step, the bytes exchanged between devices, and the intervals of this pass.
        '''
        m = self.machine
        slots = m.slots
        host = defaultdict(float)       # process -> time
        stream = defaultdict(float)     # GPU slot -> time when its queued work ends
        done = {}                       # (name, step) -> time when its results are ready
        sent = {}                       # (edge, step) -> (send time, bytes)
        events = defaultdict(list)      # process -> [(start, end)] inside MPI
        remote_out = {}
        used = {name: self.devices(name, s) for name, (_, s) in placement.items()}
        comm_bytes = 0.0
        for name, (p, s) in placement.items():
            remote_out[name] = [e for e in self.graph.out_edges[name]
                                if e.dst in placement and placement[e.dst][0] != p]
            for e in self.graph.in_edges[name]:
                if e.src in placement:
                    ps, ss = placement[e.src]
                    if ps != p:
                        comm_bytes += self.mpi_bytes(e, ss)
                    elif ss != s:
                        comm_bytes += self.out_bytes(e, ss)

        step_end = []
        for k in range(self.steps):
            for level in self.levels:
                by_proc = defaultdict(list)
                for name in level:
                    if name in placement:
                        by_proc[placement[name][0]].append(name)
                for p, names in by_proc.items():
                    t = host[p]
                    msg_ns = m.nodes[p.node].links['mpi_message_us'] * 1e3
                    gpu_ready = defaultdict(float)  # name -> time when its inputs from other GPUs are there
                    # 1. inputs
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        for e in self.graph.in_edges[name]:
                            ks = k + e.delay
                            if ks < 0 or e.src not in placement:
                                continue
                            ps, ss = placement[e.src]
                            if ps != p:
                                t0 = t
                                t = self._delivery(sent[(e, ks)], t, ps, ss, s, prev_events)
                                events[p].append((t0, t))
                                t += m.host_gpu_ns(s, self.mpi_bytes(e, ss)) + msg_ns
                            elif ss != s and slots[ss].is_gpu and slots[s].is_gpu:
                                # Ordered on the GPUs: the host does not wait for the producer
                                t += self.copy_host_ns(e, ss, s)
                                gpu_ready[name] = max(gpu_ready[name], done.get((e.src, ks), 0.0)
                                                      + m.copy_ns(ss, s, self.out_bytes(e, ss)))
                            elif ss != s:
                                t = max(t, done.get((e.src, ks), 0.0))
                                t += self.copy_host_ns(e, ss, s)
                            # the parts on other devices copy the input
                            for d in set(used[name]) - {s}:
                                t += m.copy_ns(s, d, self.out_bytes(e, ss))
                        t += c.h_in
                    # 2. trigger
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        parts = used[name]
                        t += c.h_prep + c.h_trig * len(parts)
                        if slots[s].is_gpu:
                            end = 0.0
                            for d in parts:
                                start = max(t, stream[d], gpu_ready[name])
                                stream[d] = start + c.g / len(parts)
                                end = max(end, stream[d])
                            done[(name, k)] = end
                            if c.syncs == 'trigger':
                                t = max(t, end)
                        else:
                            done[(name, k)] = t
                    # 3. post_trigger and sends
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        if c.syncs == 'post':
                            t0 = t
                            t = max(t, done[(name, k)])
                            if c.graph and self.polling:
                                events[p].append((t0, t))      # polls its sends meanwhile
                        if len(used[name]) > 1:
                            # gathers the results of the parts on the other devices
                            t = max(t, done[(name, k)])
                            nbytes = sum(c.outputs.values()) / len(used[name])
                            for d in used[name]:
                                if d != s:
                                    t += m.copy_ns(d, s, nbytes)
                        t += c.h_post
                        if remote_out[name]:
                            t0 = t
                            t = max(t, done[(name, k)])
                            if self.polling:
                                events[p].append((t0, t))      # wait_current_stream() polls
                            for e in remote_out[name]:
                                nbytes = self.mpi_bytes(e, s)
                                t += m.host_gpu_ns(s, nbytes) + msg_ns
                                sent[(e, k)] = (t, nbytes)
                                events[p].append((t, t))
                    host[p] = t
            for p in list(host):
                host[p] += self.overhead_ns
            step_end.append(max(list(host.values()) + list(stream.values()) + [0.0]))
        half = self.steps // 2
        per_step = (step_end[-1] - step_end[half - 1]) / (self.steps - half)
        for p in events:
            events[p].sort()
        return per_step, comm_bytes, events

    def evaluate(self, placement):
        mem = self.memory(placement)
        overflow = {s: use - self.machine.slots[s].memory for s, use in mem.items()
                    if use > self.machine.slots[s].memory}
        step, comm = self.step_time(placement)
        return Evaluation(step, comm, mem, not overflow, overflow)
