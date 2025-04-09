import time

import numpy as np

class LoopControl:
    def __init__(self, run_time=None, dt=None, t0=None, verbose=False):
        self._ordered_lists = {}
        self._list = []
        self._init_run_time = run_time if run_time is not None else 0.0
        self._init_dt = dt if dt is not None else 0.001
        self._init_t0 = t0 if t0 is not None else 0.0
        self._verbose = verbose
        self._run_time = 0
        self._dt = 0
        self._t0 = 0
        self._t = 0
        self._time_resolution = 1
        self._stop_on_data = None
        self._stop_at_time = 0
        self._profiling = False
        self._profiler_started = False
        self._quiet = False
        self._speed_report = False
        self._cur_time = -1
        self._old_time = 0
        self._elapsed = []
        self._nframes_cnt = -1
        self._max_order = -1

    def add(self, obj, idx):
        if obj is None:
            raise ValueError("Cannot add null object to loop")
        
        self._list.append(obj)
        
        if idx>self._max_order:
            self._max_order = idx
            self._ordered_lists[idx] = []

        self._ordered_lists[idx].append(obj)
        
        
    def remove_all(self):
        self._list.clear()
        self._ordered_lists.clear()

    def niters(self):
        return (self._run_time + self._t0) / self._dt if self._dt != 0 else 0

    def get_dt(self):
        return self._dt

    def get_t(self):
        return self._t

    def set_dt(self, dt):
        self._dt = dt

    def run(self, run_time=None, t0=None, dt=None, stop_on_data=None, stop_at_time=None,
            profiling=False, quiet=False, speed_report=False):
        self.start(run_time=run_time, t0=t0, dt=dt, stop_on_data=stop_on_data, stop_at_time=stop_at_time,
                   profiling=profiling, quiet=quiet, speed_report=speed_report)
        while self._t < self._t0 + self._run_time:
            self.iter()
        self.finish()

    def start(self, run_time=None, t0=None, dt=None, stop_on_data=None, stop_at_time=None,
              profiling=False, quiet=False, speed_report=False):
        if run_time is not None:
            self._init_run_time = run_time
        if dt is not None:
            self._init_dt = dt
        if t0 is not None:
            self._init_t0 = t0

        self._profiling = profiling
        self._quiet = quiet
        self._speed_report = speed_report
        self._stop_at_time = stop_at_time if stop_at_time is not None else 0
        self._stop_on_data = stop_on_data

        self._time_resolution = 1e9 # TODO get from somewhere
        self._run_time = self.seconds_to_t(self._init_run_time)
        self._dt = self.seconds_to_t(self._init_dt)
        self._t0 = self.seconds_to_t(self._init_t0)

        for element in self._list:
            element.setup(self._dt, self.niters())

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

        for i in range(self._max_order+1):

            for element in self._ordered_lists[i]:
                element.check_ready(self._t)

            for element in self._ordered_lists[i]:
                element.trigger()

            for element in self._ordered_lists[i]:
                element.post_trigger()

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
                print(f't={self._t / self._time_resolution:.6f} {msg}')

#        if not self._quiet: # Verbose?
#            print(f't={self._t / self._time_resolution:.6f} {msg}')
        self._t += self._dt

    def finish(self):
        if self._profiling:
            self.stop_profiling()

    def seconds_to_t(self, seconds):
        return int(seconds * self._time_resolution)

    def start_profiling(self):
        # Placeholder for profiling start
        pass

    def stop_profiling(self):
        # Placeholder for profiling end and report
        pass

