'''
Tests of iteras.mpi_progress: non-blocking MPI sends, and waits that keep
them moving.
'''
import specula
specula.init(-1, precision=1)

import os
import sys
import json
import shutil
import tempfile
import textwrap
import subprocess
import unittest
from unittest import mock

import yaml

import iteras
from iteras import mpi_progress

try:
    import mpi4py      # noqa: F401
    MPI_AVAILABLE = shutil.which('mpirun') is not None
except ImportError:
    MPI_AVAILABLE = False


def mpirun_is_open_mpi():
    '''True if the mpirun used by the tests is the one of Open MPI'''
    try:
        out = subprocess.run(['mpirun', '--version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return 'Open MPI' in out.stdout + out.stderr


class FakeRequest:
    '''Completes after *n* calls to Test()'''
    def __init__(self, n):
        self.n = n
        self.tests = 0

    def Test(self):
        self.tests += 1
        return self.tests >= self.n


class FakeComm:
    def __init__(self, n_tests=3):
        self.n_tests = n_tests
        self.isends = []
        self.ibsends = []

    def Isend(self, buffer, dest, tag):
        req = FakeRequest(self.n_tests)
        self.isends.append((buffer, dest, tag, req))
        return req

    def Ibsend(self, buffer, dest, tag):
        self.ibsends.append((buffer, dest, tag))


class FakeEvent:
    '''A CUDA event that is done after *n* queries of "done"'''
    def __init__(self, n):
        self.n = n
        self.queries = 0
        self.synchronized = False

    @property
    def done(self):
        self.queries += 1
        return self.queries >= self.n

    def synchronize(self):
        self.synchronized = True


class FakeStream:
    def __init__(self, n):
        self.event = FakeEvent(n)
        self.synchronized = False

    def record(self):
        return self.event

    def synchronize(self):
        self.synchronized = True


class TestMpiProgress(unittest.TestCase):

    def setUp(self):
        mpi_progress.reset()
        self.addCleanup(mpi_progress.reset)
        self.comm = FakeComm()
        patcher = mock.patch.object(iteras, 'process_comm', self.comm)
        patcher.start()
        self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('ITERAS_MPI_SEND_MODE', None)
        os.environ.pop('ITERAS_MPI_POLL', None)

    def test_buffer_kept_until_the_send_completes(self):
        buf = iteras.np.ones(4)
        mpi_progress.isend(buf, dest=1, tag=10, step=0)
        self.assertEqual(mpi_progress.pending(), 1)
        self.assertIs(mpi_progress._pending[0].buffer, buf)
        self.assertEqual(mpi_progress.progress(), 1)      # 1 test of 3
        self.assertEqual(mpi_progress.progress(), 1)
        self.assertEqual(mpi_progress.progress(), 0)
        self.assertEqual(self.comm.ibsends, [])

    def test_throttle_waits_for_older_steps(self):
        self.comm.n_tests = 5
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=1)
        first = self.comm.isends[0][3]
        self.assertEqual(first.tests, 0)
        # Two older steps pending on this channel: the third one waits for step 0
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=2)
        self.assertGreaterEqual(first.tests, 5)
        self.assertNotIn(0, [s.step for s in mpi_progress._pending])
        # Other channels are not throttled
        mpi_progress.isend(iteras.np.ones(1), 1, 12, step=2)
        self.assertEqual(self.comm.isends[-1][3].tests, 0)

    def test_several_sends_of_the_same_step_are_one_step(self):
        # Outputs that are lists send several items of the same step on one channel
        for _ in range(5):
            mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=1)
        self.assertTrue(all(r.tests == 0 for *_, r in self.comm.isends))

    def test_wait_without_sends_blocks(self):
        event = FakeEvent(1000)
        mpi_progress.wait_event(event)
        self.assertTrue(event.synchronized)
        self.assertEqual(event.queries, 0)
        stream = FakeStream(1000)
        mpi_progress.wait_stream(stream)
        self.assertTrue(stream.synchronized)

    def test_wait_with_sends_polls_and_tests_them(self):
        self.comm.n_tests = 1000
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        event = FakeEvent(20)
        mpi_progress.wait_event(event)
        self.assertEqual(event.queries, 20)
        self.assertFalse(event.synchronized)
        self.assertGreaterEqual(self.comm.isends[0][3].tests, 19)

    def test_wait_blocks_once_the_sends_are_done(self):
        self.comm.n_tests = 3
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        event = FakeEvent(1000)
        mpi_progress.wait_event(event)
        self.assertTrue(event.synchronized)
        self.assertLess(event.queries, 10)
        self.assertEqual(mpi_progress.pending(), 0)

    def test_wait_stream_with_sends_polls(self):
        self.comm.n_tests = 1000
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        stream = FakeStream(5)
        mpi_progress.wait_stream(stream)
        self.assertFalse(stream.synchronized)
        self.assertEqual(stream.event.queries, 5)

    def test_polling_disabled(self):
        os.environ['ITERAS_MPI_POLL'] = '0'
        self.comm.n_tests = 1000
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        event = FakeEvent(1000)
        mpi_progress.wait_event(event)
        self.assertTrue(event.synchronized)
        self.assertEqual(event.queries, 0)

    def test_buffered_mode(self):
        os.environ['ITERAS_MPI_SEND_MODE'] = 'buffered'
        mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)
        self.assertEqual(len(self.comm.ibsends), 1)
        self.assertEqual(self.comm.isends, [])
        self.assertEqual(mpi_progress.pending(), 0)
        os.environ['ITERAS_MPI_SEND_MODE'] = 'carrier pigeon'
        with self.assertRaises(ValueError):
            mpi_progress.isend(iteras.np.ones(1), 1, 10, step=0)

    def test_wait_all(self):
        self.comm.n_tests = 4
        for tag in (10, 12, 14):
            mpi_progress.isend(iteras.np.ones(1), 1, tag, step=0)
        mpi_progress.wait_all()
        self.assertEqual(mpi_progress.pending(), 0)

    def test_host_copy(self):
        from iteras.base_processing_obj import BaseProcessingObj
        a = iteras.np.arange(3.0)
        b = BaseProcessingObj._host_copy(a)
        a += 1
        iteras.np.testing.assert_array_equal(b, [0, 1, 2])


STALL_SCRIPT = '''
import sys, time
from mpi4py import MPI
import iteras
iteras.init(-1, rank=MPI.COMM_WORLD.Get_rank(), comm=MPI.COMM_WORLD)
from iteras import mpi_progress, np

class SlowEvent:
    """Stands for a GPU that is busy for 0.5 s"""
    def __init__(self):
        self.end = time.perf_counter() + 0.5
    @property
    def done(self):
        return time.perf_counter() >= self.end
    def synchronize(self):
        time.sleep(max(self.end - time.perf_counter(), 0))

comm = MPI.COMM_WORLD
MPI.Attach_buffer(bytearray(64 * 1024 * 1024))
data = np.ones(8 * 1024 * 1024 // 4, dtype=np.float32)
comm.Barrier()
t0 = MPI.Wtime()
if comm.Get_rank() == 1:
    mpi_progress.isend(data.copy(), dest=0, tag=1, step=0)
    mpi_progress.wait_event(SlowEvent())
    mpi_progress.wait_all()
else:
    time.sleep(0.1)
    comm.Recv(data, source=1, tag=1)
    print('RECEIVED', MPI.Wtime() - t0, flush=True)
comm.Barrier()
MPI.Detach_buffer()
'''


@unittest.skipUnless(MPI_AVAILABLE, 'MPI not available')
class TestMpiProgressTwoRanks(unittest.TestCase):
    '''Real MPI runs with two processes'''

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def mpirun(self, script, env=None, args=()):
        path = os.path.join(self.tmp.name, 'run.py')
        with open(path, 'w') as f:
            f.write(textwrap.dedent(script))
        full_env = dict(os.environ)
        full_env.update(env or {})
        # Open MPI forwards the OMPI_MCA_* variables to the processes; other MPIs ignore them
        cmd = ['mpirun', '-n', '2']
        result = subprocess.run(cmd + [sys.executable, path, *args], env=full_env,
                                capture_output=True, text=True, timeout=300, cwd=self.tmp.name)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_send_moves_while_the_sender_waits_for_the_gpu(self):
        # Over TCP (Open MPI), the sender must push the data of a large message:
        # it does so while it polls its (simulated) GPU. Other MPI implementations
        # ignore the OMPI_MCA_* variables and move the data with other transports
        # and progress rules: the time of arrival does not test the polling there.
        if not mpirun_is_open_mpi():
            self.skipTest('needs Open MPI (TCP transport selected with OMPI_MCA_* variables)')
        out = self.mpirun(STALL_SCRIPT, env={'OMPI_MCA_btl': 'self,tcp', 'OMPI_MCA_pml': 'ob1'})
        received = float(out.split('RECEIVED')[1].split()[0])
        self.assertLess(received, 0.35)

    def test_simulation_on_two_ranks_matches_one_process(self):
        root = self.tmp.name
        pkg = os.path.join(root, 'mpi_objs')
        os.makedirs(pkg)
        with open(os.path.join(pkg, '__init__.py'), 'w') as f:
            f.write(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue
                from iteras.connections import InputValue

                class Counter(BaseProcessingObj):
                    """Output updated in place at each step"""
                    def __init__(self, size: int=1000, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(size, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out

                    def trigger_code(self):
                        self.out.value += self.xp.arange(len(self.out.value), dtype=self.dtype)

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time

                class Scale(BaseProcessingObj):
                    def __init__(self, size: int=1000, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(size, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.inputs['in_value'] = InputValue(type=BaseValue)
                        self.outputs['out_value'] = self.out

                    def trigger_code(self):
                        self.out.value[:] = self.local_inputs['in_value'].value * 0.5

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
            '''))
        for name, klass in [('counter', 'Counter'), ('scale', 'Scale')]:
            with open(os.path.join(pkg, f'{name}.py'), 'w') as f:
                f.write(f'from mpi_objs import {klass}\n')

        def params(store, ranks):
            p = {
                'main': {'class': 'SimParams', 'root_dir': root, 'total_time': 0.010,
                         'time_step': 0.001, 'add_modules': ['mpi_objs']},
                # 400 kB per message: above the eager limit
                'counter': {'class': 'Counter', 'size': 100000},
                'scale': {'class': 'Scale', 'size': 100000,
                          'inputs': {'in_value': 'counter.out_value'}},
                'store': {'class': 'DataStore', 'store_dir': os.path.join(root, store),
                          'create_tn': False,
                          'inputs': {'input_list': ['s-scale.out_value', 'c-counter.out_value']}},
            }
            if ranks:
                # counter on rank 0 sends to scale and store on rank 1
                p['counter']['target_rank'] = 0
                for name in ('scale', 'store'):
                    p[name]['target_rank'] = 1
            path = os.path.join(root, f'{store}.yml')
            with open(path, 'w') as f:
                yaml.dump(p, f)
            return path

        launcher = f'''
            import sys
            sys.path.insert(0, {root!r})
            import iteras
            iteras.main_simul([sys.argv[1]], mpi=True, cpu=True, no_speed_report=True)
        '''
        self.mpirun(launcher, args=[params('mpi', ranks=True)],
                    env={'OMPI_MCA_btl': 'self,tcp', 'OMPI_MCA_pml': 'ob1'})

        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.addCleanup(lambda: [sys.modules.pop(m) for m in list(sys.modules)
                                 if m.split('.')[0] == 'mpi_objs'])
        from iteras.simul import Simul
        Simul(params('single', ranks=False), speed_report=False).run()

        from astropy.io import fits
        for name in ('s', 'c'):
            mpi = fits.getdata(os.path.join(root, 'mpi', f'{name}.fits'))
            single = fits.getdata(os.path.join(root, 'single', f'{name}.fits'))
            self.assertEqual(mpi.shape, single.shape)
            iteras.np.testing.assert_array_equal(mpi, single)
        # The counter grows at each step: the values were not overwritten in flight
        s = fits.getdata(os.path.join(root, 'mpi', 's.fits'))
        self.assertTrue((iteras.np.diff(s[:, 1]) > 0).all())


if __name__ == '__main__':
    unittest.main()
