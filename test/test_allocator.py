'''
Tests of the allocator (iteras.allocator) and of the Simul features it uses.
'''
import specula
specula.init(0)  # Default target device

import os
import sys
import json
import tempfile
import textwrap
import unittest
from unittest import mock

import yaml
from dataclasses import asdict

import iteras
from iteras.simul import Simul
from iteras.data_objects.architecture import Architecture
from iteras.processing_objects.data_store import DataStore
from iteras.processing_objects.data_print import DataPrint
from iteras.processing_objects.display_server import DisplayServer
from iteras.processing_objects.interactive_input import InteractiveInput
from iteras.base_processing_obj import BaseProcessingObj
from iteras.allocator import (Allocator, StaleAllocation, make_allocation, write_allocation,
                              read_allocation, check_meta, apply_allocation, cpu_candidates)
from iteras.allocator.model import SimGraph
from iteras.allocator.machine import Machine, GiB, slurm_nodes
from iteras.allocator.cost import Estimator, Proc, MPI_BUFFER_BYTES
from iteras.allocator import profiler as profiler_module
from iteras.allocator.profiler import Cost, Profile, split_chunk
from iteras.allocator.solver import Solver, Layout, Infeasible, build_groups
from iteras.allocator.report import placement_from_params

MiB = 1024 ** 2
ms = 1e6

OBJECTS = '''
from iteras.base_processing_obj import BaseProcessingObj
from iteras.base_data_obj import BaseDataObj
from iteras.base_value import BaseValue
from iteras.connections import InputValue
from iteras.processing_objects.data_store import DataStore


class CpuStore(DataStore):
    # A DataStore that must run on the CPU
    @classmethod
    def runs_on_cpu_only(cls):
        return True


class Table(BaseDataObj):
    def __init__(self, value: list=None, other: 'Table'=None, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.value = self.to_xp(value or [1.0], dtype=self.dtype)
        self.other = other


class Gen(BaseProcessingObj):
    def __init__(self, size: int=100, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.out = BaseValue(value=self.xp.zeros(size, dtype=self.dtype),
                             target_device_idx=target_device_idx)
        self.outputs['out_value'] = self.out

    def trigger_code(self):
        self.out.value += 1

    def post_trigger(self):
        super().post_trigger()
        self.out.generation_time = self.current_time


class Work(BaseProcessingObj):
    def __init__(self, size: int=100, table: Table=None, partner: BaseProcessingObj=None,
                 target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.table = table
        self.buf = self.xp.zeros(size, dtype=self.dtype)
        self.out = BaseValue(value=self.xp.zeros(size, dtype=self.dtype),
                             target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out

    def trigger_code(self):
        scale = 1 if self.table is None else self.table.value[0]
        self.out.value[:] = self.local_inputs['in_value'].value * scale

    def post_trigger(self):
        super().post_trigger()
        self.out.generation_time = self.current_time


class SplitWork(Work):
    """Like Work, as if its work were split over *parts* devices"""
    def __init__(self, size: int=100, parts: int=1, table: Table=None,
                 target_device_idx=None, precision=None):
        super().__init__(size=size, table=table, target_device_idx=target_device_idx,
                         precision=precision)

    @classmethod
    def devices_used(cls, params):
        return params.get('parts', 1)

    @classmethod
    def single_device_params(cls, params):
        return {**params, 'parts': 1}
'''


class AllocatorTestCase(unittest.TestCase):
    '''Temporary package with simple objects, and a parameter file using them'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        pkg = os.path.join(root, 'alloc_objs')
        os.makedirs(pkg)
        with open(os.path.join(pkg, '__init__.py'), 'w') as f:
            f.write(textwrap.dedent(OBJECTS))
        for name, klass in [('gen', 'Gen'), ('work', 'Work'), ('table', 'Table'),
                            ('split_work', 'SplitWork'), ('cpu_store', 'CpuStore')]:
            with open(os.path.join(pkg, f'{name}.py'), 'w') as f:
                f.write(f'from alloc_objs import {klass}\n')
        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.addCleanup(lambda: [sys.modules.pop(m) for m in list(sys.modules)
                                 if m.split('.')[0] == 'alloc_objs'])
        self.root = root
        self.params = {
            'main': {'class': 'SimParams', 'root_dir': root, 'total_time': 0.005,
                     'time_step': 0.001, 'add_modules': ['alloc_objs']},
            'table': {'class': 'Table', 'value': [2.0]},
            'gen': {'class': 'Gen', 'size': 1000},
            'work1': {'class': 'Work', 'size': 1000, 'table_ref': 'table',
                      'inputs': {'in_value': 'gen.out_value'}},
            'work2': {'class': 'Work', 'size': 1000, 'table_ref': 'table',
                      'inputs': {'in_value': 'gen.out_value'}},
            'last': {'class': 'Work', 'size': 1000, 'inputs': {'in_value': 'work1.out_value'}},
            'store': {'class': 'CpuStore', 'store_dir': os.path.join(root, 'store'),
                      'create_tn': False,
                      'inputs': {'input_list': ['a-last.out_value', 'b-work2.out_value:-1']}},
        }

    def write(self, params, name='params.yml'):
        path = os.path.join(self.root, name)
        with open(path, 'w') as f:
            yaml.dump(params, f)
        return path

    def graph(self, params=None):
        params = params or self.params
        return SimGraph(params, Simul(self.write(params)))


def cost(host_ms=0.1, gpu_ms=1.0, mem_mib=10, out_mib=1, syncs=''):
    return Cost(h_trig=host_ms * ms, g=gpu_ms * ms, mem=mem_mib * MiB,
                outputs={'out_value': out_mib * MiB}, syncs=syncs)


def machine(ngpu=2, gpu_gib=10, nodes=1, cpu_gib=100, links=None):
    spec = {'name': 'n', 'count': nodes, 'cpu': {'cores': 8, 'memory_gb': cpu_gib}}
    if links:
        spec['links'] = links
    if ngpu:
        spec['gpus'] = {'count': ngpu, 'memory_gb': gpu_gib, 'model': 'test'}
    return Machine([spec], {'bandwidth_gb_s': 10, 'latency_us': 10}, 0.0, 0.0)


class TestClassFlags(unittest.TestCase):

    def test_defaults(self):
        self.assertFalse(BaseProcessingObj.runs_on_cpu_only())
        self.assertFalse(BaseProcessingObj.runs_on_main_process_only())
        self.assertFalse(BaseProcessingObj.is_schedule_barrier())

    def test_flagged_classes(self):
        for klass in (DataPrint, DisplayServer, InteractiveInput):
            self.assertTrue(klass.runs_on_cpu_only(), klass)
        # DataStore copies GPU data asynchronously when it is on the GPU of its inputs
        self.assertFalse(DataStore.runs_on_cpu_only())
        for klass in (DisplayServer, InteractiveInput):
            self.assertTrue(klass.runs_on_main_process_only(), klass)
            self.assertTrue(klass.is_schedule_barrier(), klass)
        self.assertFalse(DataStore.runs_on_main_process_only())

    def test_data_store_on_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            store = DataStore(store_dir=d, target_device_idx=-1)
            self.assertEqual(store.target_device_idx, -1)


class TestArchitecture(unittest.TestCase):

    def node(self, **kw):
        n = {'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 10}}
        n.update(kw)
        return n

    def test_defaults(self):
        a = Architecture()
        self.assertEqual(a.nodes, 'local')
        self.assertEqual(a.profile_file, 'allocation_profile.json')
        self.assertEqual(a.mode, 'apply')

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            Architecture(objective='cheap')
        with self.assertRaises(ValueError):
            Architecture(memory_margin=1.5)
        with self.assertRaises(ValueError):
            Architecture(nodes=[self.node()], memory_source='probe')
        with self.assertRaises(ValueError):
            Architecture(nodes=[{'name': 'x'}])
        with self.assertRaises(ValueError):
            Architecture(nodes=[self.node(gpus={'count': 2})])
        with self.assertRaises(ValueError):
            Architecture(nodes=[self.node(links={'warp_speed': 1})])

    def test_machine_from_declared_nodes(self):
        arch = Architecture(nodes=[self.node(count=3, gpus={'count': 4, 'memory_gb': 64, 'model': 'A100'})],
                            memory_margin=0.25)
        m = Machine.from_architecture(arch)
        self.assertEqual(len(m.nodes), 3)
        self.assertEqual(len(m.gpu_slots()), 12)
        self.assertEqual(len(m.slots), 15)
        gpu = m.slots[m.nodes[2].gpu_slots[3]]
        self.assertEqual((gpu.node, gpu.idx, gpu.kind), (2, 3, 'gpu:A100'))
        self.assertAlmostEqual(gpu.memory, 48 * GiB)
        cpu = m.slots[m.nodes[1].cpu_slot]
        self.assertEqual((cpu.node, cpu.idx, cpu.kind), (1, -1, 'cpu'))

    def test_slurm_nodes(self):
        env = {'SLURM_JOB_NUM_NODES': '7', 'SLURM_CPUS_ON_NODE': '16',
               'SLURM_MEM_PER_NODE': str(1024 * 100), 'SLURM_GPUS_ON_NODE': '2'}
        old = dict(os.environ)
        os.environ.update(env)
        try:
            nodes = slurm_nodes([self.node(gpus={'count': 4, 'memory_gb': 64})])
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(nodes[0]['count'], 7)
        self.assertEqual(nodes[0]['cpu']['cores'], 16)
        self.assertAlmostEqual(nodes[0]['cpu']['memory_gb'], 100)
        self.assertEqual(nodes[0]['gpus']['count'], 2)
        self.assertEqual(nodes[0]['gpus']['memory_gb'], 64)


class TestSimGraph(AllocatorTestCase):

    def test_structure(self):
        g = self.graph()
        self.assertEqual(g.main_key, 'main')
        self.assertEqual(set(g.procs), {'gen', 'work1', 'work2', 'last', 'store'})
        self.assertLess(g.position['gen'], g.position['work1'])
        self.assertLess(g.position['work1'], g.position['last'])
        self.assertEqual(g.data_objs, ['table'])
        self.assertEqual(g.ref_consumers['table'], ['work1', 'work2'])
        delays = {(e.src, e.dst): e.delay for e in g.edges}
        self.assertEqual(delays[('work2', 'store')], -1)
        self.assertEqual(delays[('last', 'store')], 0)
        self.assertTrue(g.device_settable['gen'])

    def test_main_parameters_with_defaults(self):
        # Main parameters completed with the defaults of their class, as Simul does
        # (time_step and total_time are required, they have no default)
        params = json.loads(json.dumps(self.params))
        params['main'].pop('stepping', None)
        g = self.graph(params)
        from iteras.data_objects.sim_params import SimParams
        self.assertEqual(g.main_params['stepping'], SimParams().stepping)
        p = profiler_module._profile_params(params, g, {n: -1 for n in g.procs}, self.root,
                                           steps=2, warmup=1)
        self.assertAlmostEqual(p['main']['total_time'], 3 * params['main']['time_step'], places=6)

    def test_processing_object_reference(self):
        params = dict(self.params)
        params['last'] = dict(params['last'], partner_ref='work2')
        g = self.graph(params)
        self.assertEqual(g.proc_refs['last'], ['work2'])
        groups = build_groups(g, mutable_refs=[])
        self.assertIn({'work2', 'last'}, [set(g) for g in groups])

    def test_mutable_reference_groups_users(self):
        groups = build_groups(self.graph(), mutable_refs=['table'])
        self.assertIn({'work1', 'work2'}, [set(g) for g in groups])

    def test_keys_follow_upstream_changes(self):
        g1 = self.graph()
        params = json.loads(json.dumps(self.params))
        params['gen']['size'] = 2000
        g2 = self.graph(params)
        self.assertNotEqual(g1.keys['gen'], g2.keys['gen'])
        self.assertNotEqual(g1.keys['last'], g2.keys['last'])    # downstream of gen
        # Placement and run length do not change the keys or the signature
        params = json.loads(json.dumps(self.params))
        params['work1']['target_device_idx'] = 1
        params['main']['total_time'] = 10
        g3 = self.graph(params)
        self.assertEqual(g1.keys, g3.keys)
        self.assertEqual(g1.signature(), g3.signature())
        self.assertEqual(g3.pinned, {'work1': (None, 1)})


class TestEstimator(AllocatorTestCase):

    def estimator(self, m, gpu_ms=1.0, mem_mib=10, refs=None, shared=None, syncs=()):
        g = self.graph()
        costs = {}
        for name in g.procs:
            for kind in {s.kind for s in m.slots}:
                c = cost(gpu_ms=0 if kind == 'cpu' else gpu_ms, mem_mib=mem_mib,
                         syncs='trigger' if name in syncs else '')
                if kind == 'cpu':
                    c.h_trig = 5 * gpu_ms * ms
                costs[(name, kind)] = c
        return Estimator(g, m, costs, refs or {'table': 100 * MiB}, shared or {})

    def test_parallel_branches_on_two_gpus(self):
        m = machine(ngpu=2)
        est = self.estimator(m, gpu_ms=5.0)
        p = Proc(0, 0)
        one = {n: (p, 0) for n in est.graph.procs}
        two = dict(one, work2=(p, 1))
        t_one, comm_one = est.step_time(one)
        t_two, comm_two = est.step_time(two)
        self.assertEqual(comm_one, 0)
        self.assertGreater(comm_two, 0)
        self.assertLess(t_two, t_one)

    def test_syncing_object_waits_for_queued_work(self):
        m = machine(ngpu=1)
        p = Proc(0, 0)
        base = self.estimator(m, gpu_ms=5.0)
        synced = self.estimator(m, gpu_ms=5.0, syncs=('last',))
        place = {n: (p, 0) for n in base.graph.procs}
        self.assertGreaterEqual(synced.step_time(place)[0], base.step_time(place)[0])

    def test_waiting_in_post_trigger_keeps_gpus_parallel(self):
        # Two objects of the same level on two GPUs of one process: waiting in
        # post_trigger lets both GPUs work, waiting in trigger serializes them
        m = machine(ngpu=2)
        p = Proc(0, 0)
        self.params['store']['inputs']['input_list'] = ['a-last.out_value', 'b-work2.out_value']
        est = self.estimator(m, gpu_ms=5.0)
        self.assertEqual(est.graph.level['work1'], est.graph.level['work2'])
        for n in ('gen', 'last', 'store'):
            est.costs[(n, 'gpu:test')].g = 0.01 * ms
        place = {n: (p, 0) for n in est.graph.procs}
        place['work2'] = (p, 1)
        times = {}
        for phase in ('trigger', 'post'):
            for n in ('work1', 'work2'):
                for kind in ('gpu:test', 'cpu'):
                    est.costs[(n, kind)].syncs = phase
            times[phase] = est.step_time(place)[0]
        self.assertLess(times['post'], times['trigger'])

    def test_remote_process_adds_messages(self):
        m = machine(ngpu=2)
        est = self.estimator(m)
        p0, p1 = Proc(0, 0), Proc(1, 0)
        local = {n: (p0, 0) for n in est.graph.procs}
        local['work2'] = (p0, 1)
        remote = dict(local, work2=(p1, 1))
        self.assertGreater(est.step_time(remote)[0], est.step_time(local)[0])

    def mpi_case(self, single_copy, env):
        '''
        gen sends a large output (1 MiB) to work2, in another process; then gen's
        process waits for the GPU work of work1, which runs a CUDA graph.
        Return (first simulation pass, settled) times per step.
        '''
        self.params['store']['inputs']['input_list'] = ['a-last.out_value', 'b-work2.out_value']
        m = machine(ngpu=2, links={'shm_single_copy': single_copy})
        with mock.patch.dict(os.environ, env):
            est = self.estimator(m, gpu_ms=5.0)
        for n in est.graph.procs:
            for kind in ('gpu:test', 'cpu'):
                est.costs[(n, kind)].syncs = 'post'
                est.costs[(n, kind)].graph = True
        p0, p1 = Proc(0, 0), Proc(1, 0)
        place = {n: (p0, 0) for n in est.graph.procs}
        place['work2'] = (p1, 1)
        one_pass = est._simulate(place, None)[0]
        return one_pass, est.step_time(place)[0]

    def test_large_messages_wait_for_the_sender(self):
        # Buffered sends, no single-copy: the data moves only when gen's process
        # calls MPI again, after waiting for its GPU
        one_pass, settled = self.mpi_case(False, {'ITERAS_MPI_SEND_MODE': 'buffered'})
        self.assertGreater(settled, one_pass)

    def test_polling_moves_large_messages(self):
        # Non-blocking sends: the process polls them while it waits for its GPU
        _, blocking = self.mpi_case(False, {'ITERAS_MPI_POLL': '0'})
        _, polling = self.mpi_case(False, {'ITERAS_MPI_POLL': '1'})
        self.assertLess(polling, blocking)

    def test_single_copy_does_not_need_the_sender(self):
        one_pass, settled = self.mpi_case(True, {'ITERAS_MPI_POLL': '0'})
        _, blocking = self.mpi_case(False, {'ITERAS_MPI_POLL': '0'})
        self.assertAlmostEqual(settled, one_pass)
        self.assertLess(settled, blocking)

    def split_estimator(self, parts, gpu_ms=10.0):
        self.params['work2'] = dict(self.params['work2'], **{'class': 'SplitWork', 'parts': parts})
        self.params['store']['inputs']['input_list'] = ['a-last.out_value', 'b-work2.out_value']
        m = machine(ngpu=4)
        est = self.estimator(m, gpu_ms=0.01, mem_mib=40)
        for kind in ('gpu:test', 'cpu'):
            est.costs[('work2', kind)].g = gpu_ms * ms
        p = Proc(0, 0)
        place = {n: (p, 0) for n in est.graph.procs}
        place['work2'] = (p, 1)
        return m, est, place

    def test_split_object_divides_memory_and_time(self):
        m, est, place = self.split_estimator(parts=1)
        self.assertEqual(est.devices('work2', 1), [1])
        one_mem = est.memory(place)
        one_time = est.step_time(place)[0]
        m, est, place = self.split_estimator(parts=4)
        self.assertEqual(est.graph.spread['work2'], 4)
        self.assertEqual(est.devices('work2', 1), [1, 2, 3, 0])    # following GPUs, modulo
        mem = est.memory(place)
        # 40 MiB divided among four GPUs, and a copy of its 1 MiB input on each new one
        self.assertAlmostEqual(mem[1], one_mem[1] - 30 * MiB)
        self.assertAlmostEqual(mem[2], 10 * MiB + 1 * MiB + 100 * MiB)   # with table
        self.assertLess(est.step_time(place)[0], one_time - 5 * ms)

    def test_memory_per_process(self):
        # work2 in a second process on the same GPU: that process has its own copy
        # of the referenced data object (100 MiB) and of its input from gen (1 MiB),
        # and the first process receives a copy of work2's output for store (1 MiB)
        m = machine(ngpu=1)
        est = self.estimator(m, mem_mib=0)
        p0, p1 = Proc(0, 0), Proc(1, 0)
        place = {n: (p0, 0) for n in est.graph.procs}
        one = est.memory(place)
        place['work2'] = (p1, 0)
        two = est.memory(place)
        self.assertAlmostEqual(two[0] - one[0], (100 + 1 + 1) * MiB)
        # host: two messages (gen -> work2, work2 -> store), each with two steps in flight
        # at the sender and a receive buffer, and the MPI buffer of the second process
        self.assertAlmostEqual(two[1] - one[1], 2 * (2 + 1) * MiB + MPI_BUFFER_BYTES)

    def test_memory(self):
        m = machine(ngpu=2)
        est = self.estimator(m, mem_mib=10, shared={'gpu:test': [(['work1', 'work2'], 50 * MiB)]})
        est.machine.context_overhead = 1 * MiB
        p = Proc(0, 0)
        place = {n: (p, 0) for n in est.graph.procs}
        place['work2'] = (p, 1)
        place['store'] = (p, 2)      # the CPU
        mem = est.memory(place)
        # gpu0: gen, work1, last + table + shared block + context
        self.assertAlmostEqual(mem[0], (30 + 100 + 50 + 1) * MiB)
        # gpu1: work2 + its copy of table + shared block + its input from gpu0 + context
        self.assertAlmostEqual(mem[1], (10 + 100 + 50 + 1 + 1) * MiB)
        ev = est.evaluate(place)
        self.assertTrue(ev.feasible)
        small = machine(ngpu=2, gpu_gib=0.1)
        est.machine = small
        self.assertFalse(est.evaluate(place).feasible)


class TestSolver(AllocatorTestCase):

    def solve(self, m, policy='single', pins=None, mem_mib=10, cpu=True, gpu_ms=5.0, mutable=()):
        g = self.graph()
        costs = {}
        for name in g.procs:
            for s in m.slots:
                if s.kind == 'cpu' and not cpu and not g.cls(name).runs_on_cpu_only():
                    continue
                c = cost(gpu_ms=0 if s.kind == 'cpu' else gpu_ms, mem_mib=mem_mib)
                if s.kind == 'cpu' and not g.cls(name).runs_on_cpu_only():
                    # The work of a GPU object, done by the CPU
                    c.h_trig = 20 * gpu_ms * ms
                costs[(name, s.kind)] = c
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        solver = Solver(g, m, est, build_groups(g, list(mutable)), pins or {}, search_time=10)
        return g, solver.solve(Layout(m, policy))

    def test_uses_both_gpus_and_cpu_only_on_cpu(self):
        m = machine(ngpu=2)
        g, sol = self.solve(m)
        slots = {n: sol.assign[n] for n in g.procs}
        self.assertEqual(m.slots[slots['store']].idx, -1)
        self.assertNotEqual(slots['work1'], slots['work2'])
        self.assertTrue(sol.evaluation.feasible)

    def test_measured_copy_time_used(self):
        '''The host time to copy an output to another GPU, when measured, replaces the estimate'''
        from iteras.allocator.cost import Proc
        m = machine(ngpu=2)
        g = self.graph()
        costs = {(n, s.kind): cost(gpu_ms=0.1) for n in g.procs for s in m.slots}
        cpu = next(i for i, s in enumerate(m.slots) if not s.is_gpu)
        place = {n: (Proc(0, 0), 0) for n in g.procs}
        place['store'] = (Proc(0, 0), cpu)
        place['work2'] = (Proc(0, 0), 1)
        before = Estimator(g, m, costs, {'table': 1 * MiB}, {}).step_time(place)[0]
        costs[('gen', 'gpu:test')].copy_gpu = {'out_value': 3 * ms}
        after = Estimator(g, m, costs, {'table': 1 * MiB}, {}).step_time(place)[0]
        self.assertGreater(after - before, 2.5 * ms)

    def test_memory_forces_split(self):
        m = machine(ngpu=2, gpu_gib=2.5 * 10 * MiB / GiB)
        g, sol = self.solve(m, gpu_ms=0.01, cpu=False)
        used = {sol.assign[n] for n in g.procs if n != 'store'}
        self.assertEqual(used, set(m.gpu_slots()))
        self.assertTrue(sol.evaluation.feasible)

    def test_cpu_spill_when_gpus_are_full(self):
        m = machine(ngpu=1, gpu_gib=2.5 * 10 * MiB / GiB)
        g, sol = self.solve(m)
        on_cpu = [n for n in g.procs if m.slots[sol.assign[n]].idx == -1]
        self.assertGreater(len(on_cpu), 1)
        self.assertTrue(sol.evaluation.feasible)

    def test_infeasible(self):
        m = machine(ngpu=1, gpu_gib=5 * MiB / GiB)
        with self.assertRaises(Infeasible):
            self.solve(m, cpu=False)

    def test_split_object_on_gpus_only_and_pin_kept(self):
        self.params['work2'] = dict(self.params['work2'], **{'class': 'SplitWork', 'parts': 2,
                                                             'target_device_idx': 1})
        g = self.graph()
        self.assertEqual(g.spread['work2'], 2)
        m = machine(ngpu=2)
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': MiB}, {})
        layout = Layout(m, 'single')
        solver = Solver(g, m, est, build_groups(g, []), {}, search_time=5)
        self.assertNotIn(m.nodes[0].cpu_slot, solver.eligible(('work2',), layout))
        # A placement by hand of an object split over several devices is kept,
        # even when the allocator may move the others
        arch = {'nodes': [{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 10},
                           'gpus': {'count': 2, 'memory_gb': 10, 'model': 'test'}}],
                'override_placement': True}
        params = json.loads(json.dumps(self.params))
        params['work1']['target_device_idx'] = 0
        g = self.graph(params)
        alloc = Allocator(Simul(self.write(params)), arch, plan_only=True)
        pins = alloc._pins(g, m, layout)
        self.assertEqual(pins, {'work2': m.nodes[0].gpu_slots[1]})

    def test_pin_ranks_per_node(self):
        params = json.loads(json.dumps(self.params))
        params['work1'].update({'target_rank': 3, 'target_device_idx': 0})
        g = self.graph(params)
        m = machine(ngpu=2, nodes=4)
        arch = {'nodes': [{'name': 'n', 'count': 4, 'cpu': {'cores': 4, 'memory_gb': 10},
                           'gpus': {'count': 2, 'memory_gb': 10, 'model': 'test'}}]}
        sim = Simul(self.write(params))
        # by default as in the layout: one rank per GPU (rank 3 on node 1), or per node
        alloc = Allocator(sim, arch, plan_only=True)
        self.assertEqual(alloc._pins(g, m, Layout(m, 'per_gpu'))['work1'], m.nodes[1].gpu_slots[0])
        self.assertEqual(alloc._pins(g, m, Layout(m, 'single'))['work1'], m.nodes[3].gpu_slots[0])
        alloc = Allocator(sim, dict(arch, pin_ranks_per_node=1), plan_only=True)
        self.assertEqual(alloc._pins(g, m, Layout(m, 'per_gpu'))['work1'], m.nodes[3].gpu_slots[0])

    def test_pins(self):
        m = machine(ngpu=2)
        g, sol = self.solve(m, pins={'work1': 1, 'work2': 1})
        self.assertEqual(sol.assign['work1'], 1)
        self.assertEqual(sol.assign['work2'], 1)

    def test_groups_stay_together(self):
        m = machine(ngpu=2)
        g, sol = self.solve(m, mutable=['table'])
        self.assertEqual(sol.assign['work1'], sol.assign['work2'])

    def test_layouts(self):
        m = machine(ngpu=2, nodes=2)
        per_gpu = Layout(m, 'per_gpu')
        self.assertEqual(len(per_gpu.procs), 4)
        self.assertEqual(per_gpu.proc_of_slot[m.nodes[1].cpu_slot],
                         per_gpu.proc_of_slot[m.nodes[1].gpu_slots[0]])
        self.assertEqual([p.node for p in per_gpu.procs], [0, 0, 1, 1])
        single = Layout(machine(ngpu=2), 'single')
        self.assertEqual(len(single.procs), 1)

    def test_main_process_only(self):
        m = machine(ngpu=2, nodes=2)
        with mock.patch.object(DataStore, 'runs_on_main_process_only', classmethod(lambda cls: True)):
            g, sol = self.solve(m, policy='per_gpu')
        self.assertEqual(sol.layout.proc_of_slot[sol.assign['store']].rank, 0)

    def test_cpu_candidates(self):
        g = self.graph()
        m = machine(ngpu=1, gpu_gib=60 * MiB / GiB)
        costs = {}
        for n in g.procs:
            costs[(n, 'gpu:test')] = cost(gpu_ms=1.0, mem_mib=30 if n == 'gen' else 10)
            costs[(n, 'cpu')] = cost(gpu_ms=0)
        costs[('work1', 'gpu:test')].g = 10 * ms      # same memory, more GPU time: moved last
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        chosen = cpu_candidates(g, m, est)
        self.assertEqual(chosen[0], 'gen')
        self.assertNotIn('work1', chosen)
        self.assertNotIn('store', chosen)       # already on the CPU

    def test_split_object_profiled_on_one_device(self):
        self.params['work2'] = dict(self.params['work2'], **{'class': 'SplitWork', 'parts': 4})
        g = self.graph()
        p = profiler_module._profile_params(self.params, g, {n: 0 for n in g.procs},
                                           self.root, steps=2, warmup=1)
        self.assertEqual(p['work2']['parts'], 1)
        self.assertEqual(self.params['work2']['parts'], 4)      # not modified

    def test_split_chunk_keeps_groups(self):
        a, b = split_chunk(['a', 'b', 'c', 'd'], [('b', 'c')])
        self.assertTrue({'b', 'c'} <= set(a) or {'b', 'c'} <= set(b))
        self.assertEqual(sorted(a + b), ['a', 'b', 'c', 'd'])


class TestDeviceTolerance(AllocatorTestCase):

    def check(self, gpu_ms, tolerance):
        '''Placement kept for work1 and work2 (two objects of the same level), split on two GPUs by the search'''
        from iteras.allocator.solver import Layout, Solution
        m = machine(ngpu=2)
        g = self.graph()
        costs = {(n, s.kind): cost(gpu_ms=gpu_ms if s.is_gpu else 0) for n in g.procs for s in m.slots}
        arch = {'nodes': [{'name': 'n', 'cpu': {'cores': 8, 'memory_gb': 100},
                           'gpus': {'count': 2, 'memory_gb': 10, 'model': 'test'}}],
                'device_tolerance': tolerance}
        alloc = Allocator(Simul(self.write(self.params)), arch, plan_only=True)
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        layout = Layout(m, 'single')
        cpu = m.nodes[0].cpu_slot
        assign = {n: (cpu if n == 'store' else 0) for n in g.procs}
        assign['work2'] = 1
        split = Solution(assign, layout, est.evaluate(layout.placement(assign)))
        chosen = alloc._prefer_one_gpu(g, m, est, split)
        return {chosen.assign['work1'], chosen.assign['work2']}, split, chosen

    def test_one_gpu_preferred_within_tolerance(self):
        # Large tolerance: the split is not faster enough
        used, split, chosen = self.check(gpu_ms=0.5, tolerance=10.0)
        self.assertEqual(len(used), 1)
        self.assertEqual(chosen.assign['store'], split.assign['store'])

    def test_split_kept_when_clearly_faster(self):
        # Long GPU work in parallel on two GPUs: the split is much faster
        used, split, chosen = self.check(gpu_ms=20.0, tolerance=0.05)
        self.assertIs(chosen, split)


class TestGpuModels(AllocatorTestCase):

    def test_same_model(self):
        from iteras.allocator import same_gpu_model
        self.assertTrue(same_gpu_model('A100', 'NVIDIA A100-SXM-64GB'))
        self.assertTrue(same_gpu_model('NVIDIA A100-SXM-64GB', 'a100'))
        self.assertFalse(same_gpu_model('A100', 'NVIDIA H100 80GB HBM3'))
        self.assertFalse(same_gpu_model('', 'A100'))

    def test_profiled_model_used_for_a_declared_name(self):
        g = self.graph()
        profile = Profile()
        c = cost(gpu_ms=2.0)
        for n in g.procs:
            profile.entries[f'{g.keys[n]}|gpu:NVIDIA A100-SXM-64GB|p1'] = dict(asdict(c), name=n)
            profile.entries[f'{g.keys[n]}|gpu:NVIDIA RTX PRO 5000|p1'] = dict(asdict(cost(gpu_ms=1.0)), name=n)
        arch = {'nodes': [{'name': 'b', 'cpu': {'cores': 4, 'memory_gb': 10},
                           'gpus': {'count': 2, 'memory_gb': 64, 'model': 'A100', 'relative_speed': 0.1}}]}
        alloc = Allocator(Simul(self.write(self.params)), arch, plan_only=True)
        alloc.precision = 1
        m = Machine.from_architecture(alloc.arch)
        costs, _, _ = alloc._tables(g, m, profile)
        # measured on the A100: used as is, not scaled from the other model
        self.assertAlmostEqual(costs[('work1', 'gpu:A100')].g, 2.0 * ms)
        self.assertTrue(costs[('work1', 'gpu:A100')].measured)


class TestAllocationFile(AllocatorTestCase):

    def allocation(self, policy='per_gpu'):
        m = machine(ngpu=2)
        g = self.graph()
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': MiB}, {})
        layout = Layout(m, policy)
        sol = Solver(g, m, est, build_groups(g, []), {}, search_time=5).solve(layout)
        return g, make_allocation(g, m, sol)

    def test_round_trip_and_apply(self):
        g, alloc = self.allocation()
        alloc['meta'] = {'signature': g.signature(), 'pins': {}, 'architecture': {}, 'precision': 1}
        path = os.path.join(self.root, 'out', 'allocation.yml')
        write_allocation(path, alloc, 'report text')
        self.assertTrue(os.path.exists(os.path.join(self.root, 'out', 'allocation_report.txt')))
        back = read_allocation(path)
        self.assertEqual(back['n_ranks'], alloc['n_ranks'])
        self.assertEqual(back['meta'], alloc['meta'])
        params = json.loads(json.dumps(self.params))
        apply_allocation(back, params)
        for name in g.procs:
            self.assertIn('target_rank', params[name])
        # Data objects get the ranks and per-rank devices of their users
        users = {params[u]['target_rank'] for u in ('work1', 'work2')}
        if len(users) > 1:
            self.assertEqual(sorted(params['table']['target_rank']), sorted(users))
            self.assertIsInstance(params['table']['target_device_idx'], dict)
        # The file is also a valid override file
        with open(path) as f:
            body = yaml.safe_load(f)
        self.assertTrue(all(k.endswith('_override') for k in body))

    def test_data_objects_follow_referencing_data_objects(self):
        self.params['base'] = {'class': 'Table', 'value': [3.0]}
        self.params['table']['other_ref'] = 'base'
        g, alloc = self.allocation()
        self.assertEqual(alloc['data_objects']['base'], alloc['data_objects']['table'])

    def test_unused_nodes_are_not_launched(self):
        m = machine(ngpu=2, nodes=3)
        g = self.graph()
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': MiB}, {})
        layout = Layout(m, 'per_gpu')
        node2 = m.nodes[2]
        assign = {n: node2.gpu_slots[i % 2] for i, n in enumerate(g.procs)}
        assign['store'] = node2.cpu_slot
        from iteras.allocator.solver import Solution
        alloc = make_allocation(g, m, Solution(assign, layout, est.evaluate(layout.placement(assign))))
        self.assertEqual(alloc['n_nodes'], 1)
        self.assertEqual(alloc['n_ranks'], 2)
        self.assertEqual({p['node'] for p in alloc['processes']}, {0})
        self.assertEqual({o['target_rank'] for o in alloc['objects'].values()}, {0, 1})

    def test_single_process_has_no_ranks(self):
        g, alloc = self.allocation(policy='single')
        self.assertEqual(alloc['n_ranks'], 1)
        self.assertTrue(all('target_rank' not in p for p in alloc['objects'].values()))

    def test_stale_allocation(self):
        g = self.graph()
        meta = {'signature': g.signature(), 'pins': {}, 'architecture': {'a': 1}, 'precision': 1}
        check_meta(meta, json.loads(json.dumps(meta)), 'x')
        params = json.loads(json.dumps(self.params))
        params['work1']['size'] = 7
        del params['work2']
        params['store']['inputs']['input_list'] = ['a-last.out_value']
        new = dict(meta, signature=self.graph(params).signature())
        with self.assertRaises(StaleAllocation) as cm:
            check_meta(meta, new, 'x')
        self.assertIn('work1', str(cm.exception))
        self.assertIn('removed objects', str(cm.exception))
        with self.assertRaises(StaleAllocation):
            check_meta(meta, dict(meta, architecture={'a': 2}), 'x')

    def test_placement_from_params(self):
        params = json.loads(json.dumps(self.params))
        params['work2'].update({'target_rank': 1, 'target_device_idx': 1})
        m = machine(ngpu=2)
        place = placement_from_params(self.graph(params), m, ranks_per_node=2)
        self.assertEqual(place['work2'][0].rank, 1)
        self.assertEqual(place['work2'][1], m.nodes[0].gpu_slots[1])
        self.assertEqual(place['gen'][0].rank, 0)


class TestSimulPlacement(AllocatorTestCase):

    def test_rank_list_and_per_rank_device(self):
        params = json.loads(json.dumps(self.params))
        params['table'].update({'target_rank': [0, 1], 'target_device_idx': {0: -1, 1: 0}})
        simul = Simul(self.write(params), speed_report=False)
        simul.run()
        # Without MPI, rank 0 settings are used
        self.assertEqual(simul.objs['table'].target_device_idx, -1)

    def test_tag_object_on_another_rank(self):
        params = json.loads(json.dumps(self.params))
        params['table'] = {'class': 'Table', 'tag': 'only_on_rank_0', 'target_rank': 0}
        for name in ('gen', 'work1', 'work2', 'last', 'store'):
            params[name]['target_rank'] = 1
            params[name].pop('table_ref', None)
        simul = Simul(self.write(params), speed_report=False)
        with mock.patch.object(iteras, 'process_rank', 1):
            simul.build_objects(params)
        self.assertNotIn('table', simul.objs)
        self.assertIn('gen', simul.objs)

    def test_data_object_input_on_another_rank(self):
        # A data object connected as an input, built on every rank, whose consumer
        # is on another rank: each rank uses its own copy, nothing is sent
        params = json.loads(json.dumps(self.params))
        params['table_user'] = {'class': 'Work', 'inputs': {'in_value': 'table'}, 'target_rank': 0}
        for name in ('gen', 'work1', 'work2', 'last', 'store'):
            params[name]['target_rank'] = 1
        simul = Simul(self.write(params), speed_report=False)
        with mock.patch.object(iteras, 'process_rank', 1):
            simul.build_objects(params)
            simul.create_input_list_inputs(params)
            simul.connect_objects(params)
        self.assertIn('table', simul.objs)
        self.assertNotIn('table_user', simul.objs)

    def test_rank_list_not_allowed_for_processing_objects(self):
        params = json.loads(json.dumps(self.params))
        params['gen']['target_rank'] = [0, 1]
        with self.assertRaises(ValueError):
            Simul(self.write(params), speed_report=False).run()

    @unittest.skipIf(iteras.cp is None, 'needs a GPU')
    def test_reference_duplicated_on_consumer_device(self):
        params = json.loads(json.dumps(self.params))
        params['table']['target_device_idx'] = -1
        params['work1']['target_device_idx'] = 0
        params['work2']['target_device_idx'] = -1
        simul = Simul(self.write(params), speed_report=False)
        simul.run()
        t1 = simul.objs['work1'].table
        t2 = simul.objs['work2'].table
        self.assertEqual(t1.target_device_idx, 0)
        self.assertIs(t2, simul.objs['table'])


class TestAllocatorRun(AllocatorTestCase):
    '''Profiling, planning, reuse and stale allocation on a small simulation'''

    def arch(self, **kw):
        a = {'class': 'Architecture',
             'nodes': [{'name': 'cpu', 'cpu': {'cores': 4, 'memory_gb': 4}}],
             'allocation_file': os.path.join(self.root, 'alloc', 'allocation.yml'),
             'search_time': 5, 'profile_steps': 3, 'warmup_steps': 2}
        a.update(kw)
        return a

    def test_plan_reuse_and_stale(self):
        params = json.loads(json.dumps(self.params))
        params['architecture'] = self.arch(mode='plan')
        path = self.write(params)
        simul = Simul(path, speed_report=False)
        simul.run()
        alloc_file = params['architecture']['allocation_file']
        self.assertTrue(os.path.exists(alloc_file))
        self.assertTrue(os.path.exists(alloc_file.replace('.yml', '_profile.json')))
        self.assertEqual(simul.allocation['n_ranks'], 1)
        self.assertFalse(hasattr(simul, 'loop'))     # plan only: the simulation did not run
        with open(alloc_file.replace('.yml', '_report.txt')) as f:
            report = f.read()
        self.assertIn('Estimated time per step', report)

        # Apply mode reuses the allocation and runs the simulation
        params['architecture'] = self.arch(mode='apply')
        simul = Simul(self.write(params), speed_report=False)
        simul.run()
        self.assertTrue(hasattr(simul, 'loop'))
        self.assertEqual(simul.objs['store'].target_device_idx, -1)

        # A changed simulation is an error, unless a new allocation is requested
        params['gen']['size'] = 500
        params['work1']['size'] = params['work2']['size'] = params['last']['size'] = 500
        path = self.write(params)
        with self.assertRaises(StaleAllocation):
            Simul(path, speed_report=False).run()
        Simul(path, speed_report=False, reallocate=True).run()

    def test_fewest_nodes(self):
        # Three identical nodes: the placement is searched on 1, 2 and 3 nodes, and
        # the one on the fewest nodes within node_tolerance of the fastest is kept.
        # Which number of nodes is the fastest depends on the speed of the computer
        # running the test (on a slow one, work1 and work2 gain from running in
        # parallel on two nodes, and the latency of the links does not limit a
        # pipeline): with a large tolerance, the fewest nodes are kept anyway.
        params = json.loads(json.dumps(self.params))
        params['architecture'] = self.arch(
            mode='plan', nodes=[{'name': 'cpu', 'count': 3, 'cpu': {'cores': 4, 'memory_gb': 4}}],
            node_tolerance=10.0)
        simul = Simul(self.write(params), speed_report=False)
        simul.run()
        self.assertEqual(simul.allocation['n_nodes'], 1)
        with open(params['architecture']['allocation_file'].replace('.yml', '_report.txt')) as f:
            report = f.read()
        self.assertRegex(report, r'with the first k nodes: 1: .*, 2: .*, 3: ')

    def test_allocate_without_architecture(self):
        with self.assertRaises(ValueError):
            Simul(self.write(self.params), allocate=True).run()

    @unittest.skipIf(iteras.cp is None, 'needs a GPU')
    def test_gpu_profile_in_chunks(self):
        # The GPU memory pool is limited: all the objects do not fit at once, the
        # profiler measures them in chunks (the others on the CPU)
        size = 1_000_000        # 8 MB per array (double precision)
        for name in ('gen', 'work1', 'work2', 'last'):
            self.params[name]['size'] = size
        params = json.loads(json.dumps(self.params))
        params['architecture'] = self.arch(mode='plan', nodes='local', memory_source='probe',
                                           profile_device=0, profile_steps=2, warmup_steps=1)
        path = self.write(params)
        pool = iteras.cp.get_default_memory_pool()
        with iteras.cp.cuda.Device(0):
            pool.free_all_blocks()
            # 45 MB more than what the objects of other tests may still hold
            pool.set_limit(size=pool.total_bytes() + 45 * 1024 ** 2)
        calls = []
        real_run = profiler_module.profile_run

        def counting_run(*args, **kwargs):
            calls.append(sorted(n for n, d in args[2].items() if d >= 0))
            return real_run(*args, **kwargs)
        try:
            with mock.patch('iteras.allocator.profile_run', counting_run):
                Simul(path, speed_report=False).run()
        finally:
            with iteras.cp.cuda.Device(0):
                pool.set_limit(size=0)
        self.assertGreater(len(calls), 2)
        profile = Profile(params['architecture']['allocation_file'].replace('.yml', '_profile.json'))
        g = self.graph()
        for name in ('gen', 'work1', 'work2', 'last'):
            kinds = profile.kinds(g, name, iteras.global_precision)
            self.assertTrue(any(k.startswith('gpu:') for k in kinds), (name, kinds))

    @unittest.skipIf(iteras.cp is None, 'needs a GPU')
    def test_gpu_profile(self):
        params = json.loads(json.dumps(self.params))
        params['architecture'] = self.arch(mode='plan', nodes='local', memory_source='probe')
        simul = Simul(self.write(params), speed_report=False)
        simul.run()
        profile = Profile(params['architecture']['allocation_file'].replace('.yml', '_profile.json'))
        kinds = {k.split('|')[1] for k in profile.entries}
        self.assertTrue(any(k.startswith('gpu:') for k in kinds))
        self.assertIn('cpu', kinds)     # the CpuStore
        g = self.graph()
        gpu_kind = next(k for k in kinds if k.startswith('gpu:'))
        c = profile.cost(g, 'work1', gpu_kind, iteras.global_precision)
        self.assertGreater(c.outputs['out_value'], 0)
        self.assertGreater(c.mem, 0)
        self.assertGreater(profile.ref(g, 'table', iteras.global_precision)['bytes'], 0)
        self.assertFalse(profile.ref(g, 'table', iteras.global_precision)['mutable'])


if __name__ == '__main__':
    unittest.main()
