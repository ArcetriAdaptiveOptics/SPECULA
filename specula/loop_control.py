import time
import numpy as np

from specula.base_time_obj import BaseTimeObj


class LoopControl(BaseTimeObj):
    def __init__(self, verbose=False):
        super().__init__(target_device_idx=-1, precision=1)
        self._ordered_lists = {}
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
        self._max_order = -1

    def add(self, obj, idx):
        if obj is None:
            raise ValueError("Cannot add null object to loop")
        
        if idx>self._max_order:
            self._max_order = idx
            self._ordered_lists[idx] = []

        self._ordered_lists[idx].append(obj)
        
    def remove_all(self):
        self._ordered_lists.clear()

    def niters(self):
        return (self._run_time + self._t0) / self._dt if self._dt != 0 else 0

    def run(self, run_time, dt, t0=0, stop_on_data=None, stop_at_time=None,
            profiling=False, speed_report=False):
        self.start(run_time, dt, t0=t0, stop_on_data=stop_on_data, stop_at_time=stop_at_time,
                   profiling=profiling, speed_report=speed_report)
        while self._t < self._t0 + self._run_time:
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

        for i in range(self._max_order+1):
            for element in self._ordered_lists[i]:
                try:
                    element.setup()
                except:
                    print('Exception in', element.name)
                    raise

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
                try:
                    element.trigger()
                except:
                    print('Exception in', element.name)
                    raise

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
                print(f't={self.t_to_seconds(self._t):.6f} {msg}')

        self._t += self._dt

    def finish(self):

        for i in range(self._max_order+1):
             for element in self._ordered_lists[i]:
                try:
                    element.finalize()
                except:
                    print('Exception in', element.name)
                    raise

        if self._profiling:
            self.stop_profiling()

    def start_profiling(self):
        # Placeholder for profiling start
        pass

    def stop_profiling(self):
        # Placeholder for profiling end and report
        pass

