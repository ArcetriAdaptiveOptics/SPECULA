'''
Computational resources seen by the allocator: nodes, devices (the GPUs and
the CPU of each node), processes, and the speed of the links between them.
'''
import os
import time
from dataclasses import dataclass, field

import iteras
from iteras.log import get_iteras_logger
from iteras.data_objects.architecture import DEFAULT_LINKS

_logger = get_iteras_logger(__name__)

GiB = 1024 ** 3


@dataclass
class Slot:
    '''A device of a node: a GPU (idx >= 0) or the CPU (idx = -1)'''
    node: int
    idx: int
    memory: float           # bytes available to the simulation
    model: str = ''         # GPU model name, '' for the CPU
    speed: float = 1.0      # declared speed relative to the profiled GPU (GPUs only)
    cores: int = 0          # CPU only

    @property
    def is_gpu(self):
        return self.idx >= 0

    @property
    def kind(self):
        return f'gpu:{self.model}' if self.is_gpu else 'cpu'

    def label(self):
        return f'node{self.node}:' + (f'gpu{self.idx}' if self.is_gpu else 'cpu')


@dataclass
class Node:
    name: str
    links: dict
    gpu_slots: list = field(default_factory=list)
    cpu_slot: int = -1


class Machine:
    '''
    Nodes and their devices ("slots"), built from an Architecture.

    Attributes
    ----------
    slots : list of Slot
    nodes : list of Node, whose gpu_slots and cpu_slot index *slots*
    inter_node : dict with bandwidth_gb_s and latency_us
    context_overhead : bytes used by each process on each GPU it uses
    '''

    def __init__(self, nodes, inter_node, context_overhead, memory_margin):
        self.slots = []
        self.nodes = []
        self.inter_node = inter_node
        self.context_overhead = context_overhead
        self.memory_margin = memory_margin
        for spec in nodes:
            for _ in range(spec.get('count', 1)):
                n = len(self.nodes)
                node = Node(name=spec.get('name', f'node{n}'),
                            links={**DEFAULT_LINKS, **spec.get('links', {})})
                gpus = spec.get('gpus') or {}
                for i in range(gpus.get('count', 0)):
                    mem = gpus['memory_gb'][i] if isinstance(gpus['memory_gb'], list) else gpus['memory_gb']
                    node.gpu_slots.append(len(self.slots))
                    self.slots.append(Slot(node=n, idx=i, memory=mem * GiB * (1 - memory_margin),
                                           model=gpus.get('model', 'gpu'),
                                           speed=gpus.get('relative_speed', 1.0)))
                node.cpu_slot = len(self.slots)
                self.slots.append(Slot(node=n, idx=-1,
                                       memory=spec['cpu']['memory_gb'] * GiB * (1 - memory_margin),
                                       cores=spec['cpu']['cores']))
                self.nodes.append(node)

    @classmethod
    def from_architecture(cls, arch, measure_links=True):
        if arch.nodes == 'local':
            nodes = [local_node(free=(arch.memory_source == 'probe'), measure_links=measure_links)]
        elif arch.memory_source == 'slurm':
            nodes = slurm_nodes(arch.nodes)
        else:
            nodes = arch.nodes
        return cls(nodes, arch.inter_node, arch.context_overhead_gb * GiB, arch.memory_margin)

    def identical_nodes(self):
        '''True if all the nodes have the same devices'''
        def signature(node):
            return [(self.slots[s].kind, self.slots[s].memory) for s in node.gpu_slots + [node.cpu_slot]]
        return all(signature(n) == signature(self.nodes[0]) for n in self.nodes)

    def first_nodes(self, k):
        '''Machine made of the first *k* nodes of this one (same slots, same indices of the nodes)'''
        m = Machine([], self.inter_node, self.context_overhead, self.memory_margin)
        for node in self.nodes[:k]:
            new = Node(name=node.name, links=node.links)
            for s in node.gpu_slots:
                new.gpu_slots.append(len(m.slots))
                m.slots.append(self.slots[s])
            new.cpu_slot = len(m.slots)
            m.slots.append(self.slots[node.cpu_slot])
            m.nodes.append(new)
        return m

    def gpu_slots(self):
        return [i for i, s in enumerate(self.slots) if s.is_gpu]

    def link(self, a, b):
        '''(bandwidth in bytes/ns, latency in ns) between the processes of slots *a* and *b*'''
        na, nb = self.slots[a].node, self.slots[b].node
        if na == nb:
            links = self.nodes[na].links
            return links['shm_gb_s'], links['shm_latency_us'] * 1e3
        return self.inter_node['bandwidth_gb_s'], self.inter_node['latency_us'] * 1e3

    def single_copy(self, a, b):
        '''True if the receiver of a large MPI message from slot *a* to slot *b* can pull it by itself'''
        na, nb = self.slots[a].node, self.slots[b].node
        if na == nb:
            return bool(self.nodes[na].links['shm_single_copy'])
        return bool(self.inter_node.get('single_copy', False))

    def copy_ns(self, a, b, nbytes):
        '''Time to copy *nbytes* between two devices of the same process'''
        if a == b:
            return 0.0
        links = self.nodes[self.slots[a].node].links
        if self.slots[a].is_gpu and self.slots[b].is_gpu:
            return nbytes / links['gpu_gpu_gb_s'] + 10e3
        return nbytes / links['host_gpu_gb_s'] + 10e3

    def host_gpu_ns(self, slot, nbytes):
        '''Time to copy *nbytes* between a GPU and its host (0 for the CPU)'''
        if not self.slots[slot].is_gpu:
            return 0.0
        return nbytes / self.nodes[self.slots[slot].node].links['host_gpu_gb_s'] + 10e3

    def describe(self):
        lines = []
        for n, node in enumerate(self.nodes):
            gpus = [self.slots[i] for i in node.gpu_slots]
            cpu = self.slots[node.cpu_slot]
            gpu_txt = ', '.join(f'gpu{s.idx} {s.model} {s.memory / GiB:.1f} GiB' for s in gpus) or 'no GPU'
            lines.append(f'node {n} ({node.name}): {gpu_txt}; cpu {cpu.cores} cores '
                         f'{cpu.memory / GiB:.1f} GiB; links {node.links}')
        if len(self.nodes) > 1:
            lines.append(f'between nodes: {self.inter_node}')
        return '\n'.join(lines)


def _host_memory(free):
    info = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, v = line.split(':', 1)
            info[k] = int(v.split()[0]) * 1024
    return info['MemAvailable'] if free else info['MemTotal']


def local_node(free, measure_links=True):
    '''Node type for this computer: its GPUs (total or free memory) and its CPU'''
    spec = {'name': 'local', 'count': 1,
            'cpu': {'cores': os.cpu_count(), 'memory_gb': _host_memory(free) / GiB}}
    cp = iteras.cp
    ngpu = cp.cuda.runtime.getDeviceCount() if cp is not None else 0
    if ngpu:
        mems, models = [], []
        for i in range(ngpu):
            with cp.cuda.Device(i):
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                # Memory held by this process in the cupy pool is available to the simulation
                pool = cp.get_default_memory_pool()
                free_b += pool.total_bytes() - pool.used_bytes()
            mems.append((free_b if free else total_b) / GiB)
            models.append(cp.cuda.runtime.getDeviceProperties(i)['name'].decode())
        if len(set(models)) > 1:
            raise ValueError(f'Allocator: GPUs of different models on this computer ({models}) '
                             'are not supported yet')
        spec['gpus'] = {'count': ngpu, 'memory_gb': mems, 'model': models[0]}
        if measure_links:
            spec['links'] = measure_local_links(ngpu)
    return spec


def measure_local_links(ngpu, nbytes=16 * 1024 * 1024, repeat=5):
    '''
    Measure host <-> GPU and GPU <-> GPU copy speeds (GB/s), as Simul copies data.
    If the GPUs have no room for the test arrays, the default speeds are kept.
    '''
    cp = iteras.cp
    links = dict(DEFAULT_LINKS)

    def best(f):
        f()
        times = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            f()
            times.append(time.perf_counter() - t0)
        return nbytes / min(times) / 1e9

    import numpy as np
    host = np.ones(nbytes // 4, dtype=np.float32)
    arrays = []
    try:
        with cp.cuda.Device(0):
            dev0 = cp.asarray(host)
            arrays.append(dev0)

            def h2d():
                dev0.set(host)
                cp.cuda.Device(0).synchronize()
            h2d_speed = best(h2d)
            d2h_speed = best(lambda: dev0.get(out=host))
        links['host_gpu_gb_s'] = round(min(h2d_speed, d2h_speed), 2)
        if ngpu > 1:
            with cp.cuda.Device(1):
                dev1 = cp.empty_like(dev0)
                arrays.append(dev1)

                def d2d():
                    dev1[:] = dev0
                    cp.cuda.Device(1).synchronize()
                links['gpu_gpu_gb_s'] = round(best(d2d), 2)
    except cp.cuda.memory.OutOfMemoryError:
        _logger.warning('Allocator: no GPU memory to measure the copy speeds, using the defaults')
    finally:
        arrays.clear()
        dev0 = dev1 = None
        for i in range(min(ngpu, 2)):
            with cp.cuda.Device(i):
                cp.get_default_memory_pool().free_all_blocks()
    return links


def slurm_nodes(declared):
    '''
    Node types of the current slurm allocation: the number of nodes, CPU cores
    and memory come from slurm, the GPU memory and links from the declared
    node type (the first one).
    '''
    env = os.environ
    spec = dict(declared[0])
    count = int(env.get('SLURM_JOB_NUM_NODES', env.get('SLURM_NNODES', 1)))
    cpu = dict(spec['cpu'])
    if 'SLURM_CPUS_ON_NODE' in env:
        cpu['cores'] = int(env['SLURM_CPUS_ON_NODE'])
    if 'SLURM_MEM_PER_NODE' in env:
        cpu['memory_gb'] = int(env['SLURM_MEM_PER_NODE']) * 1024 ** 2 / GiB
    spec['cpu'] = cpu
    gpus = dict(spec.get('gpus') or {})
    if 'SLURM_GPUS_ON_NODE' in env:
        gpus['count'] = int(env['SLURM_GPUS_ON_NODE'])
    elif 'CUDA_VISIBLE_DEVICES' in env:
        gpus['count'] = len([d for d in env['CUDA_VISIBLE_DEVICES'].split(',') if d])
    if gpus:
        spec['gpus'] = gpus
    spec['count'] = count
    return [spec]
