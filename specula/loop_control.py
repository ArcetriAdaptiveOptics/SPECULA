
import time
import numpy as np
from collections import defaultdict

from specula.base_time_obj import BaseTimeObj
from specula import process_comm, process_rank, MPI_DBG


class LoopControl(BaseTimeObj):
    def __init__(self, verbose=False):
        super().__init__(target_device_idx=-1, precision=1)
        self.trigger_lists = defaultdict(list)
        self.verbose = verbose
        self.run_time = None
        self.dt = None
        self.t0 = None
        self.t = None
        self.speed_report = False
        self.cur_time = -1
        self.old_time = 0
        self.max_global_order = -1
        self.iter_counter = 0

    def add(self, obj, idx):
        """
        Add an object to the trigger list for a given index.

        Parameters:
            obj (object): The object to be added to the trigger list.
            idx (int): The index of the trigger list to which the object should be added.
        """
        self.trigger_lists[idx].append(obj)
        
    def niters(self):
        """
        Calculate the number of iterations based on the run time and time step.

        Returns:
            int: The number of iterations.
        """
        return int((self.run_time + self.t0) / self.dt) if self.dt != 0 else 0

    def run(self, run_time, dt, t0=0, speed_report=False):
        """
        Run the loop control for a given run time, time step, and initial time.

        Parameters:
            run_time (float): The total run time in seconds.
            dt (float): The time step in seconds.
            t0 (float): The initial time in seconds (default: 0).
            speed_report (bool): Whether to report the speed of the loop (default: False).
        """
        self.start(run_time, dt, t0=t0, speed_report=speed_report)
        while self.t < self.t0 + self.run_time:            
            if MPI_DBG: print(process_rank, 'before barrier iter', flush=True)
            if MPI_DBG: print(process_rank, 'after barrier iter', flush=True)
            if MPI_DBG: print(process_rank, 'NEW ITERATION', self.t,flush=True)
            self.iter()
            
        self.finish()

    def start(self, run_time, dt, t0=0, speed_report=False):
        
        self.speed_report = speed_report

        self.run_time = self.seconds_to_t(run_time)
        self.dt = self.seconds_to_t(dt)
        self.t0 = self.seconds_to_t(t0)
        if MPI_DBG: print(process_rank, 'Sending data pre-setup', flush=True)

        for i in sorted(self.trigger_lists.keys()):        
            # all the objects having this trigger order could be remote            
            for element in self.trigger_lists[i]:
                element.send_outputs(skip_delayed=False, first_mpi_send=True)

        if process_comm is not None:
            process_comm.barrier()
        if MPI_DBG: print(process_rank, 'Starting setups', flush=True)

        if MPI_DBG: print(process_rank, 'self.trigger_lists', self.trigger_lists, flush=True)

        for i in sorted(self.trigger_lists.keys()):
            # all the objects having this trigger order could be remote            
            for element in self.trigger_lists[i]:
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
                    element.send_outputs(delayed_only=True, first_mpi_send=False)
                except:
                    print('Exception in', element.name, flush=True)
                    raise
        
        if MPI_DBG: print(process_rank, 'Setups DONE', flush=True)
        if process_comm is not None:
            process_comm.barrier()
        
        self.t = self.t0
        self.last_reported_time = time.time()
        self.last_reported_counter = 0
        self.report_interval = 10

    def iter(self):

        # set the last_iter flag based on several conditions
        last_iter = (self.iter_counter == self.niters()-1)

        for i in sorted(self.trigger_lists.keys()):
            # all the objects having this trigger order could be remote
            if MPI_DBG: print(process_rank, 'before check_ready', flush=True)
            for element in self.trigger_lists[i]:
                try:
                    element.check_ready(self.t)
                except:
                    print('Exception in', element.name, flush=True)
                    raise

            if MPI_DBG: print(process_rank, 'before trigger', flush=True)
            for element in self.trigger_lists[i]:
                try:
                    if element.inputs_changed:
                        element.trigger()
                except:
                    print('Exception in', element.name, flush=True)
                    raise

            if MPI_DBG: print(process_rank, 'before post_trigger', flush=True)
            for element in self.trigger_lists[i]:
                try:
                    if element.inputs_changed:
                        element.post_trigger()
                    # Always send MPI outputs, regardless of whether
                    # an object was triggered or not
                    element.send_outputs(skip_delayed=last_iter, first_mpi_send=False)
                except:
                    print('Exception in', element.name, flush=True)
                    raise

        if self.speed_report:
            if self.iter_counter == self.last_reported_counter + self.report_interval:
                cur_time = time.time()
                elapsed_time = cur_time - self.last_reported_time
                msg = f"{self.report_interval / elapsed_time:.2f} Hz"
                print(f't={self.t_to_seconds(self.t):.6f} {msg}')
                self.last_reported_time = cur_time
                self.last_reported_counter = self.iter_counter

        self.t += self.dt
        self.iter_counter += 1

    def finish(self):

        for i in sorted(self.trigger_lists.keys()):
            for element in self.trigger_lists[i]:
                try:
                    element.finalize()
                except:
                    print('Exception in', element.name)
                    raise


