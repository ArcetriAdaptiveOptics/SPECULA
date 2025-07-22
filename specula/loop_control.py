
import time
import numpy as np
from collections import defaultdict

from specula.base_time_obj import BaseTimeObj
from specula import process_comm, process_rank, MPI_DBG
from specula.processing_objects.dm import DM

class LoopControl(BaseTimeObj):
    def __init__(self, verbose=False):
        super().__init__(target_device_idx=-1, precision=1)
        self._trigger_lists = defaultdict(list)
        self._verbose = verbose
        self._run_time = None
        self._dt = None
        self._t0 = None
        self._t = None
        self._stop_on_data = None
        self._stop_at_time = 0
        self._profiling = False
        self._profiler_started = False
        self._speed_report = False
        self._cur_time = -1
        self._old_time = 0
        self._elapsed = []
        self._nframes_cnt = -1
        self.max_global_order = -1
        self._iter_counter = 0

    def add(self, obj, idx):
        self._trigger_lists[idx].append(obj)
        
    def remove_all(self):
        self._trigger_lists.clear()

    def niters(self):
        return int((self._run_time + self._t0) / self._dt) if self._dt != 0 else 0

    def run(self, run_time, dt, t0=0, stop_on_data=None, stop_at_time=None,
            profiling=False, speed_report=False):
        self.start(run_time, dt, t0=t0, stop_on_data=stop_on_data, stop_at_time=stop_at_time,
                   profiling=profiling, speed_report=speed_report)
        while self._t < self._t0 + self._run_time:            
            if MPI_DBG: print(process_rank, 'before barrier iter', flush=True)
            if process_comm is not None:
                process_comm.barrier()
            if MPI_DBG: print(process_rank, 'after barrier iter', flush=True)
            if MPI_DBG: print(process_rank, 'NEW ITERATION', self._t,flush=True)
            # time.sleep(1)
            self.iter()
            
        self.finish()

    def start(self, run_time, dt, t0=0, stop_on_data=None, stop_at_time=None,
              profiling=False, speed_report=False):

        self._profiling = profiling
        self._speed_report = speed_report
        self._stop_at_time = stop_at_time if stop_at_time is not None else 0
        self._stop_on_data = stop_on_data

        self._run_time = self.seconds_to_t(run_time)
        self._dt = self.seconds_to_t(dt)
        self._t0 = self.seconds_to_t(t0)
        
        if process_comm is not None:
            process_comm.barrier()
        if MPI_DBG: print(process_rank, 'Sending data pre-setup', flush=True)

        for i in sorted(self._trigger_lists.keys()):        
            # all the objects having this trigger order could be remote            
            for element in self._trigger_lists[i]:
                element.send_outputs()
            #if process_comm is not None:
            #    process_comm.barrier()

        if process_comm is not None:
            process_comm.barrier()
        if MPI_DBG: print(process_rank, 'Starting setups', flush=True)

        if MPI_DBG: print(process_rank, 'self._trigger_lists', self._trigger_lists, flush=True)

        for i in sorted(self._trigger_lists.keys()):
            # all the objects having this trigger order could be remote            
            for element in self._trigger_lists[i]:
                try:
                    if MPI_DBG: print(process_rank, element, 'startMemUsageCount', flush=True)
                    element.startMemUsageCount()
                    if MPI_DBG: print(process_rank, element, 'setup', flush=True)
                    element.setup()
                    if MPI_DBG: print(process_rank, element, 'stopMemUsageCount', flush=True)
                    element.stopMemUsageCount()
                    if MPI_DBG: print(process_rank, element, 'printMemUsage', flush=True)
                    element.printMemUsage()
                    if MPI_DBG: print(process_rank, 'setup', element)
                    #  workaround for objects that need to send outputs
                    # before the first iter() call
                    # because their outputs are used with ":-1"
                    element.send_outputs(delayed_only=True)
                except:
                    print('Exception in', element.name, flush=True)
                    raise
        if process_comm is not None:
            process_comm.barrier()
        
        if MPI_DBG: print(process_rank, 'Setups DONE', flush=True)
        
        self._t = self._t0

        self._cur_time = -1
        self._profiler_started = False

        nframes_elapsed = 10
        self._elapsed = np.zeros(nframes_elapsed)
        self._nframes_cnt = -1

    def iter(self):

        if self._profiling and self._t != self._t0 and not self._profiler_started:
            self.start_profiling()
            self._profiler_started = True

        # set the last_iter flag based on several conditions
        last_iter = (self._iter_counter == self.niters()-1)
        if self._stop_at_time and self._t >= self.seconds_to_t(self._stop_at_time):
            last_iter = True

        for i in sorted(self._trigger_lists.keys()): 
            # all the objects having this trigger order could be remote
            if MPI_DBG: print(process_rank, 'before check_ready', flush=True)                
            for element in self._trigger_lists[i]:
                try:
                    element.check_ready(self._t)
                except:
                    print('Exception in', element.name, flush=True)
                    raise

            # if MPI_DBG: print(process_rank, 'at barrier check_ready', flush=True)                
            # if MPI_DBG: print(process_rank, 'after barrier check_ready', flush=True)

            if MPI_DBG: print(process_rank, 'before trigger', flush=True)                
            for element in self._trigger_lists[i]:
                try:
                    element.trigger()
                except:
                    print('Exception in', element.name, flush=True)
                    raise

            if MPI_DBG: print(process_rank, 'before post_trigger', flush=True)                
            for element in self._trigger_lists[i]:
                try:
                    element.post_trigger()
                    element.send_outputs(skip_delayed=last_iter)
                except:
                    print('Exception in', element.name, flush=True)
                    raise

#            if process_comm is not None:
#                process_comm.barrier()

        if self._stop_on_data and self._stop_on_data.generation_time == self._t:
            return

        if self._stop_at_time and self._t >= self.seconds_to_t(self._stop_at_time):
            raise StopIteration

        msg = ''
        nframes_elapsed = len(self._elapsed)
        if self._speed_report:
            self._old_time = self._cur_time
            self._cur_time = time.time()
            if self._nframes_cnt >= 0:
                self._elapsed[self._nframes_cnt] = self._cur_time - self._old_time
            self._nframes_cnt += 1
            nframes_good = (self._nframes_cnt == nframes_elapsed)
            self._nframes_cnt %= nframes_elapsed
            if nframes_good:
                msg = f"{1.0 / (np.sum(self._elapsed) / nframes_elapsed):.2f} Hz"
                print(f't={self.t_to_seconds(self._t):.6f} {msg}')

        self._t += self._dt
        self._iter_counter += 1

    def finish(self):

        for i in sorted(self._trigger_lists.keys()):
            for element in self._trigger_lists[i]:
                try:
                    element.finalize()
                except:
                    print('Exception in', element.name)
                    raise
#            if process_comm is not None:
#                process_comm.barrier()

        if self._profiling:
            self.stop_profiling()

    def start_profiling(self):
        # Placeholder for profiling start
        pass

    def stop_profiling(self):
        # Placeholder for profiling end and report
        pass

