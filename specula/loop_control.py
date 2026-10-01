
import time
from collections import defaultdict
from contextlib import nullcontext

from specula.base_time_obj import BaseTimeObj
from specula import process_comm, process_rank
from specula.tracing import tracer


class LoopControl(BaseTimeObj):
    def __init__(self, stepping=False):
        super().__init__(target_device_idx=-1, precision=1)
        self.logger.set_instance_name(None)
        self.trigger_lists = defaultdict(list)
        self.run_time = None
        self.dt = None
        self.t0 = None
        self.t = None
        self.speed_report = False
        self.cur_time = -1
        self.old_time = 0
        self.max_global_order = -1
        self.iter_counter = 0
        self.stepping = stepping
        # Optional callable(title) printing a GPU memory report, set by Simul
        self.mem_report = None
        # Objects whose GPU memory is still counted, until they are triggered once
        self._mem_pending = set()
        self._mem_first_trigger_report = False

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
        return int(self.run_time / self.dt) if self.dt != 0 else 0

    def run(self, run_time, dt, t0=0, speed_report=False, preroll_objs=None):
        """
        Run the loop control for a given run time, time step, and initial time.

        Parameters:
            run_time (float): The total run time in seconds, or -1 for an infinite loop.
            dt (float): The time step in seconds.
            t0 (float): The initial time in seconds (default: 0).
            speed_report (bool): Whether to report the speed of the loop (default: False).
            preroll_objs (list of str): Names of the objects to trigger from 0 to t0-dt
                before the loop starts, so that their state at t0 is the same as in a
                run started from 0 (default: None).
        """
        self.start(run_time, dt, t0=t0, speed_report=speed_report, preroll_objs=preroll_objs)
        self.next_time_to_stop = 0
        while self.run_time < 0 or self.t < self.t0 + self.run_time:
            if not process_rank and self.stepping and self.t > self.next_time_to_stop:
                nnStr = input("Press Enter to advance one timestep, or enter the number of timesteps to advance:")
                try:
                    nn = int(nnStr)
                except:
                    nn = 1
                self.next_time_to_stop = self.t + nn * dt * 1e9
            self.logger.mpi_debug(f'before barrier iter')
            self.logger.mpi_debug(f'after barrier iter')
            self.logger.mpi_debug(f'NEW ITERATION t={self.t}')
            self.iter()
        self.finish()

    def start(self, run_time, dt, t0=0, speed_report=False, preroll_objs=None):
        
        self.speed_report = speed_report

        self.run_time = self.seconds_to_t(run_time)
        self.dt = self.seconds_to_t(dt)
        self.t0 = self.seconds_to_t(t0)
        self.logger.mpi_debug(f'Sending data pre-setup')

        for i in sorted(self.trigger_lists.keys()):        
            # all the objects having this trigger order could be remote            
            for element in self.trigger_lists[i]:
                element.send_outputs(skip_delayed=False, first_mpi_send=True)

        if process_comm is not None:
            process_comm.barrier()
        self.logger.mpi_debug(f'Starting setups')

        self.logger.mpi_debug(f'{self.trigger_lists=}')

        for i in sorted(self.trigger_lists.keys()):
            # all the objects having this trigger order could be remote            
            for element in self.trigger_lists[i]:
                try:
                    self.logger.mpi_debug(f'' + str(element) + ' startMemUsageCount')
                    element.startMemUsageCount()
                    self.logger.mpi_debug(f'' + str(element) + ' setup')
                    with tracer('setup', element):
                        element.setup()
                    element.sanity_check()
                    self.logger.mpi_debug(f'' + str(element) + ' stopMemUsageCount')
                    element.stopMemUsageCount()
                    self.logger.mpi_debug(f'setup '+str(element))
                    #  workaround for objects that need to send outputs
                    # before the first iter() call
                    # because their outputs are used with ":-1"
                    element.send_outputs(delayed_only=True, first_mpi_send=False)
                except:
                    self.logger.error('Exception in ' + element.name)
                    raise
        
        self.logger.debug(f'Setups DONE')
        self._report_mem('after setup')
        self._mem_pending = {el for lst in self.trigger_lists.values() for el in lst}
        self._mem_first_trigger_report = True
        if process_comm is not None:
            process_comm.barrier()
        
        if preroll_objs:
            self.preroll(preroll_objs)

        self.t = self.t0
        self.last_reported_time = time.time()
        self.last_reported_counter = 0
        self.report_interval = 10

    def preroll(self, preroll_objs):
        """
        Trigger the given objects at t = 0, dt, ..., t0-dt, in trigger order,
        without MPI communication and without the other objects.
        """
        if self.t0 % self.dt != 0:
            raise ValueError(f'Pre-roll needs t0 multiple of dt: '
                             f't0={self.t_to_seconds(self.t0)} s, dt={self.t_to_seconds(self.dt)} s')

        names = set(preroll_objs)
        levels = [[el for el in self.trigger_lists[i] if el.name in names]
                  for i in sorted(self.trigger_lists.keys())]
        levels = [lev for lev in levels if lev]
        n_steps = self.t0 // self.dt
        self.logger.info(f'Pre-rolling {n_steps} steps up to t0={self.t_to_seconds(self.t0)} s: '
                         f'{[el.name for lev in levels for el in lev]}')


        # The pre-roll is traced as a single range. The phases of the objects
        # still show up in NVTX, but are not written to the trace file,
        # where they would be mixed with those of the loop.
        with tracer('preroll'), tracer.no_record():
            for t in range(0, self.t0, self.dt):
                for level in levels:
                    for element in level:
                        self._counted(element, element.check_ready, t)
                    for element in level:
                        try:
                            if element.inputs_changed:
                                with tracer('trigger', element):
                                    self._counted(element, element.trigger)
                                with tracer('post_trigger', element):
                                    self._counted(element, element.post_trigger)
                                self._mem_pending.discard(element)
                        except:
                            self.logger.error(f'Exception in {element.name} during pre-roll')
                            raise

    def iter(self):

        # set the last_iter flag based on several conditions
        last_iter = (self.iter_counter == self.niters()-1)

        if tracer.recording:
            tracer.begin_iteration(self.iter_counter, self.t_to_seconds(self.t))

        for i in sorted(self.trigger_lists.keys()):
            # all the objects having this trigger order could be remote
            self.logger.mpi_debug(f'before check_ready')
            for element in self.trigger_lists[i]:
                try:
                    self._counted(element, element.check_ready, self.t)
                except:
                    self.logger.error(f'Exception in {element.name}')
                    raise

            self.logger.mpi_debug(f'before trigger')
            for element in self.trigger_lists[i]:
                try:
                    if element.inputs_changed:
                        with tracer('trigger', element):
                            self._counted(element, element.trigger)
                except:
                    self.logger.error(f'Exception in {element.name}')
                    raise

            self.logger.mpi_debug(f'before post_trigger')
            for element in self.trigger_lists[i]:
                try:
                    if element.inputs_changed:
                        with tracer('post_trigger', element):
                            self._counted(element, element.post_trigger)
                        self._mem_pending.discard(element)                            

                    # Always send MPI outputs, regardless of whether
                    # an object was triggered or not
                    # Only traced with remote outputs, otherwise it does nothing
                    with tracer('send_outputs', element) if element.remote_outputs else nullcontext():
                        element.send_outputs(skip_delayed=last_iter, first_mpi_send=False)
                except:
                    self.logger.error(f'Exception in {element.name}')
                    raise

        if self.speed_report:
            if self.iter_counter == self.last_reported_counter + self.report_interval:
                cur_time = time.time()
                elapsed_time = cur_time - self.last_reported_time
                if elapsed_time > 0:
                    msg = f"{self.report_interval / elapsed_time:.2f} Hz,  {1000 * elapsed_time / self.report_interval :.3f} ms"
                else:
                    msg = "0.00 Hz,  0.000 ms"
                self.logger.info(f't={self.t_to_seconds(self.t):.6f} {msg}')
                self.last_reported_time = cur_time
                self.last_reported_counter = self.iter_counter

        if self._mem_first_trigger_report and not self._mem_pending:
            self._report_mem('after the first trigger of all objects')
            self._mem_first_trigger_report = False
        if tracer.recording:
            tracer.end_iteration()

        self.t += self.dt
        self.iter_counter += 1

    def _counted(self, element, method, *args):
        '''
        Call *method*, counting the GPU memory it allocates for *element*
        if the element has not been triggered yet: objects often allocate
        buffers lazily, on their first trigger.
        '''
        if element not in self._mem_pending:
            return method(*args)
        element.startMemUsageCount()
        try:
            return method(*args)
        finally:
            element.stopMemUsageCount()

    def _report_mem(self, title):
        if self.mem_report is not None:
            self.mem_report(title)

    def finish(self):
        if self._mem_pending:
            never = sorted(el.name for el in self._mem_pending)
            self._report_mem(f'at the end (never triggered: {", ".join(never)})')
        self._mem_pending = set()
        self._mem_first_trigger_report = False

        for i in sorted(self.trigger_lists.keys()):
            for element in self.trigger_lists[i]:
                try:
                    element.finalize()
                except:
                    self.logger.error(f'Exception in {element.name}')
                    raise

