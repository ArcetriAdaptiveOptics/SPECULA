'''
Run all displays in a separate process, so that slow drawing
does not slow down the simulation.

When enabled (``specula --async-displays``), each BaseDisplay object
in the simulation process does not draw anything: at each trigger it
copies its inputs to the CPU and sends them through a queue to a single
display process. That process holds a replica of each display, built
with the same constructor arguments, and runs the usual display code.

The queue is bounded: if the display process falls behind, new data is
dropped (before copying it) instead of piling up in memory, and the
simulation never waits for the displays. Displays that accumulate a history
(like PlotDisplay) set *skip_updates* to False: their data, usually small,
goes through a second, unbounded queue, so that no point is lost.

Only this module's top-level imports run in the display process
before specula.init(), so they must not import other SPECULA modules.
'''

import io
import types
import queue
import pickle
import importlib
import multiprocessing as mp

# Maximum number of pending messages per display
QUEUE_LEN_PER_DISPLAY = 2

_enabled = False
_displays = []
_queue = None
_history_queue = None
_process = None
_dead = False
_dropped = 0


def init(enable):
    '''If *enable* is True, displays built from now on will run in the display process'''
    global _enabled, _displays, _dropped, _dead
    _enabled = enable
    _displays = []
    _dropped = 0
    _dead = False


def enabled():
    return _enabled


def register(display):
    _displays.append(display)


class _Pickler(pickle.Pickler):
    '''Pickle modules (like the *xp* attribute of data objects) by name'''
    def reducer_override(self, obj):
        if isinstance(obj, types.ModuleType):
            return importlib.import_module, (obj.__name__,)
        return NotImplemented


def _dumps(obj):
    buf = io.BytesIO()
    _Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    return buf.getvalue()


def _to_cpu(value):
    if value is None:
        return None
    if isinstance(value, list):
        return [_to_cpu(x) for x in value]
    return value.copyTo(-1)


def start(precision, log_level):
    '''Start the display process with all registered displays'''
    global _enabled, _queue, _history_queue, _process
    _enabled = False
    if not _displays:
        return
    ctx = mp.get_context('spawn')   # fork is not safe after CUDA initialization
    _queue = ctx.Queue(maxsize=QUEUE_LEN_PER_DISPLAY * len(_displays))
    _history_queue = ctx.Queue()
    specs = [(d.name, type(d), d._init_args, d._init_kwargs) for d in _displays]
    _process = ctx.Process(target=_worker,
                           args=(_queue, _history_queue, _dumps(specs), precision, log_level),
                           daemon=True)
    _process.start()


def send(display):
    '''Send the current inputs of *display* to the display process'''
    global _dropped, _dead
    if _queue is None or _dead:
        return
    if not _process.is_alive():
        display.logger.error(f'Async displays: the display process has exited (exit code {_process.exitcode}), '
                             'displays will not be updated')
        _dead = True
        return
    if display.skip_updates and _queue.full():
        _dropped += 1
        return
    inputs = {k: _to_cpu(v) for k, v in display.local_inputs.items()}
    # Pickle here: the queue would pickle later in a background thread,
    # when a CPU simulation may have already modified the arrays
    data = _dumps((display.name, display.current_time, inputs))
    if not display.skip_updates:
        _history_queue.put(data)
        return
    try:
        _queue.put_nowait(data)
    except queue.Full:
        _dropped += 1


def stop(logger, timeout=30):
    '''Wait for the display process to draw the pending data, then stop it'''
    global _queue, _history_queue, _process
    if _process is not None:
        if _dropped:
            logger.info(f'Async displays: {_dropped} updates skipped because the display process was busy')
        if _process.is_alive():
            _history_queue.put(None)
            try:
                _queue.put(None, timeout=timeout)
            except queue.Full:
                pass
            _process.join(timeout)
            if _process.is_alive():
                _process.terminate()
        else:
            logger.error(f'Async displays: the display process exited early (exit code {_process.exitcode})')
        # Do not wait at exit for data that a dead process will never read
        for qq in [_queue, _history_queue]:
            if qq is not None:
                qq.cancel_join_thread()
                qq.close()
    _queue = None
    _history_queue = None
    _process = None


def _worker(q, history_q, specs, precision, log_level):
    try:
        _worker_loop(q, history_q, specs, precision, log_level)
    except KeyboardInterrupt:
        pass


def _worker_loop(q, history_q, specs, precision, log_level):
    import time
    import specula
    specula.init(-1, precision=precision)

    displays = {}
    for name, klass, args, kwargs in pickle.loads(specs):
        d = klass(*args, **kwargs)
        d.name = name
        d.init_logging(log_level)
        displays[name] = d

    # Each display runs its setup() before its first update,
    # when its inputs are available
    not_setup = set(displays)

    # Each queue ends with a None terminator
    open_queues = [q, history_q]
    while open_queues:
        # Get all pending updates
        msgs = []
        for qq in list(open_queues):
            while True:
                try:
                    msg = qq.get_nowait()
                except queue.Empty:
                    break
                if msg is None:
                    open_queues.remove(qq)
                    break
                msgs.append(msg)

        if not msgs:
            for d in displays.values():
                d.fig.canvas.flush_events()
            time.sleep(0.02)
            continue

        # Apply all updates, then draw each figure once
        figs = {}
        for msg in msgs:
            name, t, inputs = pickle.loads(msg)
            d = displays[name]
            d.current_time = t
            d.current_time_seconds = d.t_to_seconds(t)
            for k, v in inputs.items():
                d.inputs[k].set([] if v is None else v)
            if name in not_setup:
                d.setup()   # also gets the inputs
                not_setup.remove(name)
            else:
                d.get_all_inputs()
            d.trigger_code()
            figs[id(d.fig)] = d

        for d in figs.values():
            d._safe_draw()

    for d in displays.values():
        d.finalize()
