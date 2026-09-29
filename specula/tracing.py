'''
Timing instrumentation of the simulation loop.

Each phase of a processing object (setup, input gathering, prepare_trigger,
trigger, post_trigger, output sending) is marked with:

- an NVTX range named "<object name>.<phase>", visible in NVIDIA Nsight
  Systems, whenever NVTX is available through cupy;
- optionally, a line in a tab-separated text file, enabled with
  :meth:`Tracer.open` (``--trace-file`` on the command line). When the
  file is closed, a summary sorted by total time is written next to it.

GPU work is asynchronous: without synchronization, the times in the
text file measure how long the host spends launching work, not how long
the GPU takes to execute it. :meth:`Tracer.open` with ``sync=True``
(``--trace-sync``) synchronizes the device at the end of each phase, so
that times include GPU execution. This serializes host and device, so
the total run time will be higher than in a normal run.

A usually lighter alternative is ``gpu_events=True`` (``--trace-gpu-events``):
the trigger phase of GPU objects is bracketed by two CUDA events,
recorded on the stream where the object's work runs (its own stream if
it uses a CUDA graph, the current stream otherwise). The elapsed time
between them is written as an additional "trigger_gpu" phase, a few
iterations later, when the events have completed. Note that this is the
time taken by that segment of the stream, which includes any time the
GPU spends waiting for the host to launch the next kernel.

The module-level :data:`tracer` instance is used by the simulation loop.
It can also be used as a context manager or decorator, to mark additional
sections of code (see :meth:`Tracer.__call__`).
'''

import os
import time
import logging
import functools
import contextlib

from specula import cp

# NVTX color ids, one per phase, so that phases are
# easy to tell apart in the Nsight Systems timeline.
PHASE_COLORS = {
    'setup': 0,
    'preroll': 0,
    'inputs': 1,
    'prepare_trigger': 2,
    'trigger': 3,
    'post_trigger': 4,
    'send_outputs': 5,
}

# GPU event results are read this many iterations after being recorded,
# so that reading them normally does not stall the host.
GPU_EVENTS_LAG = 2

_logger = logging.getLogger('specula.tracing')


def _get_nvtx():
    try:
        from cupy.cuda import nvtx
        if nvtx.available:
            return nvtx
    except Exception:
        pass
    return None


class Tracer:
    '''
    Records NVTX ranges and, optionally, per-phase timings to a text file.

    Usage::

        tracer.begin(obj, 'trigger')
        obj.trigger()
        tracer.end(obj, 'trigger')

    or, as a context manager or decorator::

        with tracer('interpolation', self):
            ...

        @tracer('compute_slopes')
        def compute_slopes(self):
            ...

    begin()/end() pairs can be nested. Durations are inclusive:
    the time of a nested phase is also counted in the enclosing one.
    '''

    def __init__(self):
        self._nvtx = _get_nvtx()
        self._file = None
        self._filename = None
        self._sync = False
        self._skip = 0
        self._gpu_events = False
        self._event_pool = {}
        self._reset()

    def _reset(self):
        self._stack = []
        self._gpu_stack = []
        self._gpu_pending = []
        self._stats = {}
        self._t_open = time.perf_counter_ns()
        self._t_iter = 0
        self._toplevel_ns = 0   # time in traced phases during loop iterations
        self._loop_ns = 0       # total time of loop iterations
        self._n_iter = 0
        self._active = True     # False while skipping the first iterations
        self.iteration = -1     # -1 before the loop starts (setup phase)
        self.sim_time = 0.0

    @property
    def recording(self):
        '''True if timings are being written to a file'''
        return self._file is not None

    def open(self, filename, sync=False, rank=None, skip=0, gpu_events=False):
        '''
        Start writing timings to *filename*.
        With MPI, *rank* is added to the file name to get one file per rank.
        The first *skip* loop iterations are not recorded. The setup phase,
        which runs before the loop, is always recorded.
        With *gpu_events*, the GPU time of each trigger is measured with
        CUDA events and written as a "trigger_gpu" phase.
        '''
        if rank is not None:
            root, ext = os.path.splitext(filename)
            filename = f'{root}.rank{rank}{ext}'
        self.close()
        self._filename = filename
        self._file = open(filename, 'w', buffering=1024 * 1024)
        self._sync = sync and cp is not None
        self._skip = max(int(skip), 0)
        self._gpu_events = gpu_events and cp is not None
        self._reset()
        self._file.write(f'# SPECULA trace, device sync: {self._sync}, '
                         f'GPU events: {self._gpu_events}, '
                         f'skipped iterations: {self._skip}. '
                         'Durations are inclusive of nested phases.\n')
        self._file.write('iter\tt_sim_s\tobject\tclass\tphase\tstart_us\tdur_us\n')

    def close(self):
        '''
        Stop recording and write the summary file, if a file was open.
        Returns the summary text, or None.
        '''
        if self._file is None:
            return None
        self._flush_gpu_events(wait_all=True)
        self._file.close()
        self._file = None
        summary = self.summary()
        summary_filename = os.path.splitext(self._filename)[0] + '.summary.txt'
        with open(summary_filename, 'w') as f:
            f.write(summary)
        _logger.info(f'Timing trace written to {self._filename}, summary in {summary_filename}')
        return summary

    def begin_iteration(self, iteration, sim_time):
        '''Called by the loop at the start of each iteration'''
        self.iteration = iteration
        self.sim_time = sim_time
        self._active = iteration >= self._skip
        self._t_iter = time.perf_counter_ns()

    def end_iteration(self):
        '''Called by the loop at the end of each iteration'''
        if not self._active:
            return
        self._loop_ns += time.perf_counter_ns() - self._t_iter
        self._n_iter += 1
        if self._gpu_pending:
            self._flush_gpu_events()

    def __call__(self, phase, obj=None, color_id=None):
        '''
        Return a context manager and decorator that marks a section of code
        as *phase* of *obj*, like a begin()/end() pair. The range is closed
        even if the section raises an exception.

        As a decorator with no *obj*, the range is attributed to the first
        argument of the decorated function if it has a ``name`` (typically
        ``self`` of a processing object), so that it is named
        "<object name>.<phase>". Otherwise, the range is named *phase* and
        written to the text file with no object.
        '''
        return _Range(self, phase, obj, color_id)

    @contextlib.contextmanager
    def no_record(self):
        '''
        Context manager that stops writing phases to the text file inside
        its block. NVTX ranges are still emitted. Ranges must not cross the
        block boundary.
        '''
        active = self._active
        self._active = False
        try:
            yield
        finally:
            self._active = active

    def begin(self, obj, phase, color_id=None):
        if self._nvtx is not None:
            if color_id is None:
                color_id = PHASE_COLORS.get(phase, -1)
            name = phase if obj is None else f'{obj.name}.{phase}'
            self._nvtx.RangePush(name, color_id)
        if self._file is not None and self._active:
            if self._gpu_events and phase == 'trigger' and self._is_gpu_obj(obj) and self.iteration >= 0:
                # Events and streams belong to the current device. The device context
                # restores the previous one, so that tracing does not change it.
                with obj._target_device:
                    stream = self._obj_stream(obj)
                    ev_start = self._get_event(obj.target_device_idx)
                    ev_start.record(stream)
                self._gpu_stack.append((obj, ev_start, stream))
            self._stack.append(time.perf_counter_ns())

    def end(self, obj, phase):
        if self._file is not None and self._active:
            # No sync while a CUDA graph is being captured: it is not allowed,
            # and the captured work does not run until the graph is launched.
            if self._sync and self._is_gpu_obj(obj):
                with obj._target_device:
                    if not cp.cuda.get_current_stream().is_capturing():
                        cp.cuda.runtime.deviceSynchronize()
            t_end = time.perf_counter_ns()
            t_start = self._stack.pop()
            dur = t_end - t_start
            if not self._stack and self.iteration >= 0:
                self._toplevel_ns += dur
            self._record(self.iteration, self.sim_time, obj, phase, t_start, dur)

            if self._gpu_stack and self._gpu_stack[-1][0] is obj and phase == 'trigger':
                _, ev_start, stream = self._gpu_stack.pop()
                with obj._target_device:
                    ev_end = self._get_event(obj.target_device_idx)
                    ev_end.record(stream)
                self._gpu_pending.append((self.iteration, self.sim_time, obj, t_start,
                                          ev_start, ev_end))
        if self._nvtx is not None:
            self._nvtx.RangePop()

    def _record(self, iteration, sim_time, obj, phase, t_start, dur):
        if obj is None:
            name, cls = '-', '-'
        else:
            name, cls = obj.name, obj.__class__.__name__
        self._file.write(f'{iteration}\t{sim_time:.6f}\t{name}\t{cls}\t{phase}\t'
                         f'{(t_start - self._t_open) / 1000:.1f}\t{dur / 1000:.1f}\n')
        key = (name, cls, phase)
        st = self._stats.get(key)
        if st is None:
            self._stats[key] = [1, dur, dur]
        else:
            st[0] += 1
            st[1] += dur
            if dur > st[2]:
                st[2] = dur

    @staticmethod
    def _is_gpu_obj(obj):
        return getattr(obj, 'target_device_idx', -1) >= 0

    @staticmethod
    def _obj_stream(obj):
        '''
        Stream where the object's trigger work is queued: its own stream
        when a CUDA graph is launched, otherwise the current stream.
        '''
        if getattr(obj, 'cuda_graph', None) is not None:
            return obj.stream
        return cp.cuda.get_current_stream()

    def _get_event(self, device_idx):
        pool = self._event_pool.setdefault(device_idx, [])
        if pool:
            return pool.pop()
        return cp.cuda.Event()

    def _release_event(self, device_idx, event):
        self._event_pool.setdefault(device_idx, []).append(event)

    def _flush_gpu_events(self, wait_all=False):
        '''
        Write the GPU times whose events have completed. Events older than
        GPU_EVENTS_LAG iterations (or all of them, if *wait_all* is set)
        are waited for.
        '''
        still_pending = []
        for entry in self._gpu_pending:
            iteration, sim_time, obj, t_start, ev_start, ev_end = entry
            must_wait = wait_all or (self.iteration - iteration >= GPU_EVENTS_LAG)
            if not ev_end.done:
                if not must_wait:
                    still_pending.append(entry)
                    continue
                ev_end.synchronize()
            elapsed_ms = cp.cuda.get_elapsed_time(ev_start, ev_end)
            self._record(iteration, sim_time, obj, 'trigger_gpu', t_start, int(elapsed_ms * 1e6))
            self._release_event(obj.target_device_idx, ev_start)
            self._release_event(obj.target_device_idx, ev_end)
        self._gpu_pending = still_pending

    def summary(self):
        '''
        Table of all (object, phase) pairs sorted by total time.
        Percentages are relative to the time spent in loop iterations.
        '''
        loop_ns = max(self._loop_ns, 1)
        untraced = self._loop_ns - self._toplevel_ns
        rows = sorted(self._stats.items(), key=lambda kv: kv[1][1], reverse=True)
        lines = [f'Iterations: {self._n_iter}, time in loop iterations: {self._loop_ns / 1e6:.1f} ms '
                 f'({self._loop_ns / 1e3 / max(self._n_iter, 1):.1f} us/iteration), '
                 f'device sync: {self._sync}, GPU events: {self._gpu_events}, '
                 f'skipped iterations: {self._skip}',
                 f'Loop time outside traced phases (loop overhead, speed report, ...): '
                 f'{untraced / 1e6:.1f} ms ({100 * untraced / loop_ns:.1f}%)',
                 'The setup phase runs before the loop: its %loop is only for comparison.',
                 'trigger_gpu (with GPU events) overlaps with the host phases: it is not '
                 'included in the traced loop time.',
                 '',
                 f'{"object":30s} {"class":24s} {"phase":16s} {"count":>7s} '
                 f'{"total_ms":>10s} {"mean_us":>10s} {"max_us":>10s} {"%loop":>6s}']
        for (name, cls, phase), (count, total, tmax) in rows:
            lines.append(f'{name[:30]:30s} {cls[:24]:24s} {phase:16s} {count:7d} '
                         f'{total / 1e6:10.2f} {total / count / 1e3:10.1f} {tmax / 1e3:10.1f} '
                         f'{100 * total / loop_ns:6.1f}')
        return '\n'.join(lines) + '\n'


class _Range:
    '''Context manager and decorator returned by :meth:`Tracer.__call__`'''

    def __init__(self, tracer, phase, obj, color_id):
        self._tracer = tracer
        self._phase = phase
        self._obj = obj
        self._color_id = color_id

    def __enter__(self):
        self._tracer.begin(self._obj, self._phase, self._color_id)
        return self

    def __exit__(self, *exc):
        self._tracer.end(self._obj, self._phase)
        return False

    def __call__(self, f):
        tracer, phase, obj, color_id = self._tracer, self._phase, self._obj, self._color_id

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            o = obj
            if o is None and args and isinstance(getattr(args[0], 'name', None), str):
                o = args[0]
            tracer.begin(o, phase, color_id)
            try:
                return f(*args, **kwargs)
            finally:
                tracer.end(o, phase)
        return wrapper


tracer = Tracer()
