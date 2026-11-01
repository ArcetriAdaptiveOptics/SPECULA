'''
Tests of the allocator (iteras.allocator) and of the Simul features it uses.
'''
import specula
specula.init(0)  # Default target device

import os
import sys
import json

import numpy as np
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
        # A memory pool of 45 MB for this test only: the default pool may hold free
        # memory left by other tests, inside blocks it cannot release
        with iteras.cp.cuda.Device(0):
            pool = iteras.cp.cuda.MemoryPool()
            pool.set_limit(size=45 * 1024 ** 2)
        calls = []
        real_run = profiler_module.profile_run

        def counting_run(*args, **kwargs):
            calls.append(sorted(n for n, d in args[2].items() if d >= 0))
            return real_run(*args, **kwargs)
        try:
            with mock.patch('iteras.allocator.profile_run', counting_run), \
                    iteras.cp.cuda.using_allocator(pool.malloc):
                Simul(path, speed_report=False).run()
        finally:
            pool.free_all_blocks()
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


class FakeComm:
    '''Stands for an MPI communicator of *size* ranks: bcast() returns what rank 0 sends'''

    def __init__(self, size, received=None):
        self.size = size
        self.received = received
        self.sent = []

    def Get_size(self):
        return self.size

    def bcast(self, value, root=0):
        self.sent.append(value)
        return value if self.received is None else self.received


class TestAllocatorBranches(AllocatorTestCase):
    '''Branches of the allocator that the runs of the other tests do not reach'''

    def allocator(self, params=None, **arch):
        a = {'nodes': [{'name': 'n', 'count': 1, 'cpu': {'cores': 4, 'memory_gb': 4},
                        'gpus': {'count': 2, 'memory_gb': 10, 'model': 'test'}}],
             'allocation_file': os.path.join(self.root, 'alloc', 'allocation.yml'),
             'profile_file': os.path.join(self.root, 'alloc', 'profile.json')}
        a.update(arch)
        return Allocator(Simul(self.write(params or self.params)), a, plan_only=True)

    def profile(self, alloc, g, kinds, **cost_kw):
        p = Profile()
        for n in g.procs:
            for kind in kinds:
                p.entries[f'{g.keys[n]}|{kind}|p{alloc.cost_tag}'] = dict(asdict(cost(**cost_kw)), name=n)
        return p

    def test_allocate_on_rank_0_sends_the_allocation(self):
        alloc = self.allocator()
        allocation = {'objects': {'work1': {'target_rank': 1}}, 'data_objects': {}, 'n_ranks': 2}
        comm = FakeComm(2)
        params = json.loads(json.dumps(self.params))
        with mock.patch.object(iteras, 'process_comm', comm), mock.patch.object(iteras, 'process_rank', 0), \
                mock.patch.object(Allocator, '_load_or_compute', return_value=allocation) as load:
            self.assertIs(alloc.allocate(params), allocation)
        self.assertEqual(load.call_args.kwargs['n_ranks'], 2)
        self.assertEqual(comm.sent, [(allocation, False)])
        self.assertEqual(params['work1']['target_rank'], 1)

    def test_allocate_on_other_ranks_receives_the_allocation(self):
        alloc = self.allocator()
        allocation = {'objects': {'work1': {'target_rank': 1}}, 'data_objects': {}, 'n_ranks': 2}
        params = json.loads(json.dumps(self.params))
        with mock.patch.object(iteras, 'process_comm', FakeComm(2, (allocation, False))), \
                mock.patch.object(iteras, 'process_rank', 1), \
                mock.patch.object(Allocator, '_load_or_compute') as load:
            alloc.allocate(params)
        load.assert_not_called()
        self.assertEqual(params['work1']['target_rank'], 1)
        # Rank 0 failed: the other ranks stop too
        with mock.patch.object(iteras, 'process_comm', FakeComm(2, (None, True))), \
                mock.patch.object(iteras, 'process_rank', 1):
            with self.assertRaisesRegex(RuntimeError, 'failed on rank 0'):
                alloc.allocate(params)

    def test_allocate_error_on_rank_0_is_raised_after_the_broadcast(self):
        alloc = self.allocator()
        comm = FakeComm(2)
        with mock.patch.object(iteras, 'process_comm', comm), mock.patch.object(iteras, 'process_rank', 0), \
                mock.patch.object(Allocator, '_load_or_compute', side_effect=ValueError('no fit')):
            with self.assertRaisesRegex(ValueError, 'no fit'):
                alloc.allocate(json.loads(json.dumps(self.params)))
        self.assertEqual(comm.sent, [(None, True)])

    def test_saved_allocation_for_another_number_of_ranks(self):
        alloc = self.allocator()
        alloc.plan_only = False
        g = self.graph()
        with mock.patch.object(Allocator, 'compute',
                               return_value=({'objects': {}, 'data_objects': {}, 'n_ranks': 2, 'n_nodes': 1,
                                              'layout': 'per_gpu', 'processes': [],
                                              'estimated_step_ms': 1.0}, '')):
            with self.assertRaisesRegex(ValueError, 'needs 2 MPI ranks, but 1 were launched'):
                alloc._load_or_compute(g, self.params, n_ranks=1)

    def test_infeasible_with_cpu_costs(self):
        # The objects do not fit on the GPUs, and their CPU costs are known: no new measurement
        alloc = self.allocator(nodes=[{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 1e-6},
                                       'gpus': {'count': 1, 'memory_gb': 1e-6, 'model': 'test'}}])
        g = self.graph()
        profile = self.profile(alloc, g, ['cpu', 'gpu:test'])
        profile.path = alloc.arch.profile_file
        os.makedirs(os.path.dirname(profile.path), exist_ok=True)
        profile.save()
        with self.assertRaises(Infeasible):
            alloc.compute(g, self.params)

    def test_numbers_of_nodes_that_do_not_fit(self):
        alloc = self.allocator(nodes=[{'name': 'n', 'count': 2, 'cpu': {'cores': 4, 'memory_gb': 100},
                                       'gpus': {'count': 1, 'memory_gb': 10, 'model': 'test'}}],
                               processes='per_gpu')
        g = self.graph()
        profile = self.profile(alloc, g, ['cpu', 'gpu:test'])
        m = Machine.from_architecture(alloc.arch)
        real_solve = Allocator._solve

        def one_node_does_not_fit(self, graph, sub, est, prof):
            if len(sub.nodes) == 1:
                raise Infeasible('does not fit')
            return real_solve(self, graph, sub, est, prof)
        with mock.patch.object(Allocator, '_solve', one_node_does_not_fit):
            with self.assertLogs('iteras.allocator', level='INFO') as logs:
                _, _, _, by_nodes = alloc._solve_nodes(g, m, profile)
        self.assertEqual([k for k, _ in by_nodes], [2])
        self.assertTrue(any('1 node(s): does not fit' in line for line in logs.output))
        # No number of nodes fits: the error of the search on all the nodes
        with mock.patch.object(Allocator, '_solve', side_effect=Infeasible('never fits')):
            with self.assertRaisesRegex(Infeasible, 'never fits'):
                alloc._solve_nodes(g, m, profile)

    def test_gpu_model_not_profiled_is_scaled(self):
        alloc = self.allocator(nodes=[{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 10},
                                       'gpus': {'count': 1, 'memory_gb': 10, 'model': 'H100',
                                                'relative_speed': 2.0}}])
        g = self.graph()
        profile = self.profile(alloc, g, ['cpu', 'gpu:A100'], gpu_ms=4.0)
        costs, _, _ = alloc._tables(g, Machine.from_architecture(alloc.arch), profile)
        c = costs[('work1', 'gpu:H100')]
        self.assertAlmostEqual(c.g, 2.0 * ms)
        self.assertFalse(c.measured)

    def test_mutable_data_object_keeps_its_users_together(self):
        alloc = self.allocator()
        g = self.graph()
        profile = self.profile(alloc, g, ['cpu', 'gpu:test'])
        profile.refs[f'{g.keys["table"]}|p{alloc.cost_tag}'] = {'bytes': 8.0, 'mutable': True}
        m = Machine.from_architecture(alloc.arch)
        est = alloc.estimator(g, m, profile)
        with self.assertLogs('iteras.allocator', level='INFO') as logs:
            sol = alloc._solve(g, m, est, profile)
        self.assertTrue(any('stay together' in line for line in logs.output))
        self.assertEqual(sol.assign['work1'], sol.assign['work2'])

    def test_layouts_that_do_not_fit(self):
        alloc = self.allocator(processes='auto')
        g = self.graph()
        m = Machine.from_architecture(alloc.arch)
        profile = self.profile(alloc, g, ['cpu', 'gpu:test'])
        est = alloc.estimator(g, m, profile)
        real_solve = Solver.solve

        def single_does_not_fit(solver, layout):
            if layout.policy == 'single':
                raise Infeasible('does not fit')
            return real_solve(solver, layout)
        with mock.patch.object(Solver, 'solve', single_does_not_fit), \
                mock.patch('importlib.util.find_spec', return_value=True):
            self.assertEqual(alloc._solve(g, m, est, profile).layout.policy, 'per_gpu')
        with mock.patch.object(Solver, 'solve', side_effect=Infeasible('does not fit')), \
                mock.patch('importlib.util.find_spec', return_value=True):
            with self.assertRaisesRegex(Infeasible, 'No layout fits'):
                alloc._solve(g, m, est, profile)

    def test_policies(self):
        self.assertEqual(self.allocator(processes='single')._policies(machine(ngpu=2)), ['single'])
        local = self.allocator(nodes='local')
        with mock.patch('importlib.util.find_spec', return_value=None):
            with self.assertLogs('iteras.allocator', level='INFO') as logs:
                self.assertEqual(local._policies(machine(ngpu=2)), ['single'])
        self.assertIn('mpi4py not available', logs.output[0])
        self.assertEqual(local._default_device(), max(iteras.default_target_device_idx, -1))

    def test_pins(self):
        from iteras.allocator.solver import Layout
        g_params = json.loads(json.dumps(self.params))
        g_params['work1']['target_rank'] = 0
        g_params['work1']['target_device_idx'] = -1
        g_params['work2']['target_rank'] = 0
        alloc = self.allocator(g_params)
        g = self.graph(g_params)
        m = machine(ngpu=2)
        pins = alloc._pins(g, m, Layout(m, 'single'))
        self.assertEqual(pins['work1'], m.nodes[0].cpu_slot)     # target_device_idx -1: the CPU
        self.assertEqual(pins['work2'], m.nodes[0].gpu_slots[0])  # no target_device_idx: first GPU
        for pin, error in (({'target_rank': 5}, 'rank 5'), ({'target_device_idx': 7}, 'GPU 7')):
            p = json.loads(json.dumps(self.params))
            p['work1'].update(pin)
            with self.assertRaisesRegex(ValueError, error):
                self.allocator(p)._pins(self.graph(p), m, Layout(m, 'single'))

    def test_one_gpu_kept_with_pinned_and_fixed_objects(self):
        from iteras.allocator.solver import Layout, Solution
        p = json.loads(json.dumps(self.params))
        p['work1']['target_device_idx'] = 1
        alloc = self.allocator(p, device_tolerance=10.0)
        g = self.graph(p)
        m = machine(ngpu=2)
        costs = {(n, s.kind): cost(gpu_ms=0.5 if s.is_gpu else 0) for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        layout = Layout(m, 'single')
        assign = {n: (m.nodes[0].cpu_slot if n == 'store' else m.nodes[0].gpu_slots[0]) for n in g.procs}
        assign['work1'] = m.nodes[0].gpu_slots[1]
        assign['work2'] = m.nodes[0].gpu_slots[1]
        split = Solution(assign, layout, est.evaluate(layout.placement(assign)))
        with mock.patch.dict(g.device_settable, {'gen': False}):
            chosen = alloc._prefer_one_gpu(g, m, est, split)
        self.assertEqual(chosen.assign['work1'], m.nodes[0].gpu_slots[1])     # pinned
        self.assertEqual(chosen.assign['gen'], split.assign['gen'])           # device not settable
        # Pins that do not exist in this machine, or no placement on one GPU fits: the solution is kept
        with mock.patch.object(Allocator, '_pins', side_effect=ValueError('no such rank')):
            self.assertIs(alloc._prefer_one_gpu(g, m, est, split), split)
        tiny = Machine([{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 100},
                         'gpus': {'count': 2, 'memory_gb': 1e-6, 'model': 'test'}}],
                       {'bandwidth_gb_s': 10, 'latency_us': 10}, 0.0, 0.0)
        est_tiny = Estimator(g, tiny, costs, {'table': 1 * MiB}, {})
        self.assertIs(alloc._prefer_one_gpu(g, tiny, est_tiny, split), split)

    def test_profile_steps(self):
        p = json.loads(json.dumps(self.params))
        p['work1']['start_time'] = 0.004        # 4 steps
        p['work2']['dt'] = 0.003                # 3 steps
        p['last']['dt'] = 'not a number'
        alloc = self.allocator(p, warmup_steps=1, profile_steps=2)
        warmup, steps = alloc._profile_steps(self.graph(p))
        self.assertEqual(warmup, 6)
        self.assertEqual(steps, 6)

    def test_missing_cpu_costs_measured_with_profile_cpu(self):
        alloc = self.allocator(nodes=[{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 10}}], profile_cpu=True)
        g = self.graph()
        profile = Profile(alloc.arch.profile_file)
        alloc._ensure_profile(g, self.params, profile, Machine.from_architecture(alloc.arch))
        self.assertTrue(alloc._has_cpu_costs(g, profile))
        # Read from the profile file the next time
        with self.assertLogs('iteras.allocator', level='INFO') as logs:
            alloc._ensure_profile(g, self.params, Profile(alloc.arch.profile_file),
                                  Machine.from_architecture(alloc.arch))
        self.assertTrue(any('costs read from' in line for line in logs.output))

    def test_cpu_candidates(self):
        from iteras.allocator import cpu_candidates
        g = self.graph()
        m = machine(ngpu=1, gpu_gib=25 * MiB / GiB)
        costs = {(n, s.kind): cost(mem_mib=10) for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        with mock.patch.dict(g.device_settable, {'gen': False}):
            chosen = cpu_candidates(g, m, est)
        self.assertNotIn('gen', chosen)
        self.assertTrue(chosen)
        # Nothing can be moved: no candidate
        with mock.patch.dict(g.device_settable, {n: False for n in g.procs}):
            self.assertEqual(cpu_candidates(g, m, est), [])

    def test_make_allocation_with_objects_out_of_the_placement(self):
        from iteras.allocator import make_allocation
        from iteras.allocator.solver import Layout, Solution
        g = self.graph()
        m = machine(ngpu=1)
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        layout = Layout(m, 'single')
        assign = {n: m.nodes[0].gpu_slots[0] for n in g.procs if n not in ('work1', 'work2')}
        assign['store'] = m.nodes[0].cpu_slot
        sol = Solution(assign, layout, est.evaluate(layout.placement(assign)))
        allocation = make_allocation(g, m, sol)
        self.assertNotIn('work1', allocation['objects'])
        self.assertNotIn('table', allocation['data_objects'])    # its users are not placed

    def test_read_allocation_of_another_file(self):
        from iteras.allocator import read_allocation
        path = os.path.join(self.root, 'other.yml')
        with open(path, 'w') as f:
            f.write('work1_override: {target_device_idx: 0}\n')
        with self.assertRaisesRegex(ValueError, 'not an allocation file'):
            read_allocation(path)

    def test_stale_allocation_names_what_changed(self):
        from iteras.allocator import check_meta, StaleAllocation
        saved = {'signature': {'a': '1', 'b': '2'}, 'pins': {}, 'architecture': {}, 'precision': 1}
        current = {'signature': {'a': '9', 'c': '3'}, 'pins': {'a': [0, 1]}, 'architecture': {'x': 1},
                   'precision': 0}
        with self.assertRaises(StaleAllocation) as e:
            check_meta(saved, current, 'f')
        for what in ('new objects', 'removed objects', 'changed objects', 'target_rank',
                     'different architecture', 'different precision'):
            self.assertIn(what, str(e.exception))


class TestProfilerHelpers(AllocatorTestCase):
    '''Profiler functions on the CPU'''

    def devices(self, g, dev=-1):
        return {n: dev for n in g.procs}

    def test_is_oom(self):
        self.assertTrue(profiler_module._is_oom(MemoryError('CUDA: out of memory')))
        self.assertTrue(profiler_module._is_oom(RuntimeError('cudaErrorMemoryAllocation OUT_OF_MEMORY')))
        self.assertFalse(profiler_module._is_oom(ValueError('wrong value')))

    def test_run_or_oom(self):
        with mock.patch.object(profiler_module, 'run_once', side_effect=MemoryError('out of memory')):
            with self.assertRaises(profiler_module.OutOfDeviceMemory):
                profiler_module._run_or_oom()
        with mock.patch.object(profiler_module, 'run_once', side_effect=ValueError('other error')):
            with self.assertRaisesRegex(ValueError, 'other error'):
                profiler_module._run_or_oom()

    def test_array_blocks_of_a_view(self):
        base = np.zeros(100)
        blocks = profiler_module.array_blocks(base[10:20][2:5], set())
        self.assertEqual(blocks, {('h', base.__array_interface__['data'][0]): base.nbytes})

    def test_array_checksum_of_lists(self):
        a = profiler_module.array_checksum([np.ones(3), (np.zeros(2), {'x': np.ones(2)})])
        b = profiler_module.array_checksum([np.ones(3), (np.zeros(2), {'x': np.ones(2) * 2})])
        self.assertEqual(len(a), 3)
        self.assertNotEqual(a, b)

    def test_mpi_bytes_without_value(self):
        class NoValue:
            def get_value(self):
                raise AttributeError('no value')
        self.assertEqual(profiler_module._mpi_bytes(NoValue(), 123.0), 123.0)

    def test_profile_params(self):
        from iteras.display.plot_display import PlotDisplay
        params = json.loads(json.dumps(self.params))
        params['main']['display_server'] = True
        params['disp'] = {'class': 'PlotDisplay'}
        g = self.graph()
        g.classes['disp'] = PlotDisplay
        params['gen']['target_device_idx'] = 0
        with mock.patch.dict(g.device_settable, {'gen': False}):
            p = profiler_module._profile_params(params, g, dict(self.devices(g), gen=0), self.root,
                                                steps=2, warmup=1)
        self.assertFalse(p['main']['display_server'])
        self.assertNotIn('disp', p)
        self.assertNotIn('target_device_idx', p['gen'])     # its class does not accept it

    def test_profile_run_on_the_cpu(self):
        from iteras.tracing import tracer
        g = self.graph()
        trace = os.path.join(self.root, 'outer.tsv')
        # A profiling run while the timings of the simulation are traced: they are not mixed
        tracer.open(trace)
        try:
            res = profiler_module.profile_run(self.params, g, self.devices(g), os.path.join(self.root, 'w'),
                                              steps=2, warmup=1, async_run=False)
            self.assertTrue(tracer.recording)
        finally:
            tracer.close()
        self.assertIsNone(res.step_ns)          # no asynchronous run
        self.assertEqual({kind for kind, _ in res.costs.values()}, {'cpu'})
        res2 = profiler_module.profile_run(self.params, g, self.devices(g), os.path.join(self.root, 'w'),
                                           steps=2, warmup=1)
        self.assertGreater(res2.step_ns, 0)

    def test_profile_run_of_objects_not_built(self):
        # A display is removed from the profiling run: it has no cost
        from iteras.display.plot_display import PlotDisplay
        params = json.loads(json.dumps(self.params))
        g = self.graph()
        g.procs.append('disp')
        g.classes['disp'] = PlotDisplay
        params['disp'] = {'class': 'PlotDisplay'}
        res = profiler_module.profile_run(params, g, self.devices(g), os.path.join(self.root, 'w'),
                                          steps=2, warmup=1, async_run=False)
        self.assertNotIn('disp', res.costs)

    def test_shared_blocks_in_the_profile(self):
        g = self.graph()
        result = profiler_module.RunResult(
            costs={}, refs={}, step_ns=None, devices={},
            shared=[{'kind': 'cpu', 'owners': ['work1', 'work2'], 'bytes': 8.0},
                    {'kind': 'gpu:test', 'owners': ['gen', 'last'], 'bytes': 4.0}])
        profile = Profile()
        profile.add_run(g, result, 1)
        # work1 and work2 have the same parameters: the same key, both are owners
        self.assertEqual(profile.shared_blocks(g, 'cpu', 1), [(['work1', 'work2'], 8.0)])
        self.assertEqual(profile.shared_blocks(g, 'gpu:test', 1), [(['gen', 'last'], 4.0)])
        # Owners that are not in this simulation
        profile.shared['unknown|cpu|p1'] = 2.0
        self.assertEqual(len(profile.shared_blocks(g, 'cpu', 1)), 1)
        # A profile without a file is not saved
        profile.save()


class TestReportSolverMachine(AllocatorTestCase):
    '''Branches of the report, solver, graph and machine'''

    def test_placement_from_params(self):
        g = self.graph()
        m = machine(ngpu=1)
        with mock.patch.dict(g.params['work1'], {'target_device_idx': None}):
            placement = placement_from_params(g, m, default_device=0)
        self.assertEqual(placement['work1'][1], m.nodes[0].gpu_slots[0])
        with mock.patch.dict(g.device_settable, {'work2': False}), \
                mock.patch.dict(g.params['work2'], {'target_device_idx': -1}):
            placement = placement_from_params(g, m, default_device=0)
        self.assertEqual(placement['work2'][1], m.nodes[0].gpu_slots[0])   # default device
        for pin, error in (({'target_rank': 3}, 'node 3'), ({'target_device_idx': 4}, 'GPU 4')):
            with mock.patch.dict(g.params['work1'], pin):
                with self.assertRaisesRegex(ValueError, error):
                    placement_from_params(g, m)

    def test_describe_placement_and_object_table(self):
        from iteras.allocator.report import describe_placement, object_table
        g = self.graph()
        m = machine(ngpu=1)
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        placement = {n: (Proc(0, 0), m.nodes[0].gpu_slots[0]) for n in ('gen', 'work1')}
        text = describe_placement(g, m, placement)
        self.assertIn('rank 0 node0:gpu0: gen, work1', text)
        table = object_table(g, m, est, placement)
        self.assertIn('work1', table)
        self.assertNotIn('last', table)         # not in the placement

    def test_report_names_scaled_gpu_times(self):
        alloc = Allocator(Simul(self.write(self.params)),
                          {'nodes': [{'name': 'n', 'cpu': {'cores': 4, 'memory_gb': 10},
                                      'gpus': {'count': 1, 'memory_gb': 10, 'model': 'H100',
                                               'relative_speed': 2.0}}],
                           'allocation_file': os.path.join(self.root, 'a', 'allocation.yml'),
                           'search_time': 1}, plan_only=True)
        g = self.graph()
        profile = Profile(alloc.arch.profile_file)
        for n in g.procs:
            # Slow on the CPU: the objects that can go on the GPU go there
            profile.entries[f'{g.keys[n]}|cpu|p{alloc.cost_tag}'] = dict(asdict(cost(host_ms=50)), name=n)
            profile.entries[f'{g.keys[n]}|gpu:A100|p{alloc.cost_tag}'] = dict(asdict(cost()), name=n)
        os.makedirs(os.path.dirname(profile.path), exist_ok=True)
        profile.save()
        _, report = alloc.compute(g, self.params)
        self.assertIn('scaled from another GPU model', report)

    def test_solver_constraints(self):
        g = self.graph()
        m = machine(ngpu=2)
        costs = {(n, s.kind): cost() for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        layout = Layout(m, 'single')
        gpu0, gpu1 = m.nodes[0].gpu_slots
        solver = Solver(g, m, est, [('work1', 'work2')], {'work1': gpu0, 'work2': gpu1})
        with self.assertRaisesRegex(Infeasible, 'pinned to different devices'):
            solver.eligible(('work1', 'work2'), layout)
        # A class whose device cannot be set: only the default device
        solver = Solver(g, m, est, [('gen',)], {}, default_device=1)
        with mock.patch.dict(g.device_settable, {'gen': False}):
            self.assertEqual(solver.eligible(('gen',), layout), [gpu1])
        # No cost for any device
        empty = Estimator(g, m, {}, {'table': 1 * MiB}, {})
        with self.assertRaisesRegex(Infeasible, 'No device can run'):
            Solver(g, m, empty, [('gen',)], {}).eligible(('gen',), layout)

    def test_first_fit(self):
        g = self.graph()
        m = machine(ngpu=2, gpu_gib=25 * MiB / GiB)
        costs = {(n, s.kind): cost(mem_mib=10) for n in g.procs for s in m.slots}
        est = Estimator(g, m, costs, {'table': 1 * MiB}, {})
        layout = Layout(m, 'single')
        groups = build_groups(g, [])
        solver = Solver(g, m, est, groups, {})
        eligible = {grp: solver.eligible(grp, layout) for grp in groups}
        assign = solver._first_fit(eligible, layout)
        self.assertEqual(set(assign), set(g.procs))
        # Devices too small for any group
        small = machine(ngpu=1, gpu_gib=1e-6, cpu_gib=1e-6)
        est = Estimator(g, small, {(n, s.kind): cost(mem_mib=10) for n in g.procs for s in small.slots},
                        {'table': 1 * MiB}, {})
        layout = Layout(small, 'single')
        solver = Solver(g, small, est, groups, {})
        eligible = {grp: solver.eligible(grp, layout) for grp in groups}
        with self.assertRaisesRegex(Infeasible, 'do not fit in any device'):
            solver._first_fit(eligible, layout)

    def test_graph_errors(self):
        params = json.loads(json.dumps(self.params))
        params['other'] = {'class': 'NoSuchClassAnywhere'}
        with self.assertRaisesRegex(ValueError, 'cannot determine the class of section other'):
            self.graph(params)
        params = json.loads(json.dumps(self.params))
        params['work1']['inputs'] = {'in_value': 'main'}
        with self.assertRaisesRegex(ValueError, 'does not come from a processing object or a data object'):
            self.graph(params)
        params = json.loads(json.dumps(self.params))
        params['work1']['target_device_idx'] = 1
        self.assertEqual(self.graph(params).pins(), {'work1': [None, 1]})

    def test_data_object_as_input(self):
        params = json.loads(json.dumps(self.params))
        params['work1']['inputs'] = {'in_value': 'table'}
        g = self.graph(params)
        self.assertNotIn('work1', [e.dst for e in g.edges])

    def test_machine_from_slurm(self):
        env = {'SLURM_JOB_NUM_NODES': '3', 'SLURM_CPUS_ON_NODE': '16', 'SLURM_MEM_PER_NODE': '2048',
               'CUDA_VISIBLE_DEVICES': '0,1'}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop('SLURM_GPUS_ON_NODE', None)
            arch = Architecture(nodes=[{'name': 'b', 'cpu': {'cores': 4, 'memory_gb': 10},
                                        'gpus': {'count': 4, 'memory_gb': 64, 'model': 'A100'}}],
                                memory_source='slurm')
            m = Machine.from_architecture(arch)
        self.assertEqual(len(m.nodes), 3)
        self.assertEqual(len(m.nodes[0].gpu_slots), 2)
        self.assertEqual(m.slots[m.nodes[0].cpu_slot].cores, 16)

    def test_local_machine_without_gpu(self):
        from iteras.allocator import machine as machine_module
        with mock.patch.object(iteras, 'cp', None):
            spec = machine_module.local_node(free=True)
            m = Machine.from_architecture(Architecture(nodes='local', memory_source='probe'))
        self.assertNotIn('gpus', spec)
        self.assertEqual(spec['cpu']['cores'], os.cpu_count())
        self.assertGreater(spec['cpu']['memory_gb'], 0)
        self.assertEqual(m.gpu_slots(), [])

    @unittest.skipIf(sys.platform == 'win32', 'no sysconf on Windows')
    def test_host_memory_without_proc_meminfo(self):
        # As on macOS: the memory comes from sysconf
        from iteras.allocator import machine as machine_module
        real_exists = os.path.exists
        with mock.patch.object(machine_module.os.path, 'exists',
                               lambda p: False if p == '/proc/meminfo' else real_exists(p)), \
                mock.patch.object(machine_module.sys, 'platform', 'darwin'):
            total = machine_module._host_memory(free=False)
            free = machine_module._host_memory(free=True)
        self.assertGreater(total, 0)
        self.assertGreater(free, 0)
        self.assertLessEqual(free, total)
        real_sysconf = os.sysconf

        def no_available_pages(name):
            if name == 'SC_AVPHYS_PAGES':
                raise ValueError(name)
            return real_sysconf(name)
        with mock.patch.object(machine_module.os, 'sysconf', no_available_pages):
            self.assertEqual(machine_module._sysconf_memory(free=True), machine_module._sysconf_memory(free=False))

    @unittest.skipUnless(sys.platform == 'win32', 'Windows only')
    def test_host_memory_on_windows(self):
        from iteras.allocator import machine as machine_module
        self.assertGreater(machine_module._windows_memory(free=True), 0)

    def test_first_nodes_and_copies(self):
        m = machine(ngpu=2, nodes=2)
        sub = m.first_nodes(1)
        self.assertEqual(len(sub.nodes), 1)
        self.assertEqual(len(sub.gpu_slots()), 2)
        self.assertEqual(m.copy_ns(0, 0, 1e6), 0.0)


class TestSmallBranches(AllocatorTestCase):

    def test_architecture_errors(self):
        for nodes, error in (([], 'non-empty list'), (['x'], 'must be a dictionary'),
                             ([{'name': 'n', 'cpu': {'cores': 1, 'memory_gb': 1}, 'disk': 1}], 'unknown node keys'),
                             ([{'name': 'n', 'cpu': {'cores': 1}}], 'cpu.memory_gb is missing')):
            with self.assertRaisesRegex(ValueError, error):
                Architecture(nodes=nodes)

    def test_command_line_accelerate(self):
        from iteras.scripts.iteras_main import build_parser
        args = build_parser().parse_args(['--accelerate', '--allocate', '--reallocate', 'p.yml'])
        self.assertTrue(args.accelerate and args.allocate and args.reallocate)

    def test_mpi_bytes_without_cost(self):
        g = self.graph()
        m = machine(ngpu=1)
        est = Estimator(g, m, {}, {'table': 1 * MiB}, {})
        self.assertEqual(est.mpi_bytes(g.edges[0], m.nodes[0].cpu_slot), 0.0)

