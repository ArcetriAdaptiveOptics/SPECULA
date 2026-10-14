'''
Run all displays in a separate process, so that slow drawing
does not slow down the simulation.

With ``specula --async-displays``, a BaseDisplay object in the simulation
does not draw: at each trigger it copies its inputs to the CPU and sends them
to a single display process, which holds a copy of each display, built with
the same constructor arguments, and runs the usual display code.

The simulation never waits for the displays. While QUEUE_LEN_PER_DISPLAY updates
per display are pending, displays with *skip_updates* (like image displays) drop
new data, before copying it, so that memory stays bounded. Displays that accumulate
a history (like PlotDisplay) set *skip_updates* to False: their data, usually small,
is always sent.

Only this module's top-level imports run in the display process
before specula.init(), so they must not import other SPECULA modules.
'''

import io
import types
import queue
import pickle
import importlib
import multiprocessing as mp

# Maximum number of pending updates per display, for displays that can skip them
QUEUE_LEN_PER_DISPLAY = 2

enabled = False     # if True, the displays being built register in *displays*
displays = []
_queue = None
_slots = None       # free queue slots for the updates that can be skipped
_process = None
_dropped = 0


def init(enable):
    '''If *enable* is True, displays built from now on will run in the display process'''
    global enabled, displays, _dropped
    enabled = enable
    displays = []
    _dropped = 0


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
    if isinstance(value, list):
        return [x.copyTo(-1) for x in value]
    return None if value is None else value.copyTo(-1)


def start(precision, log_level):
    '''Start the display process with all registered displays'''
    global enabled, _queue, _slots, _process
    enabled = False
    if not displays:
        return
    ctx = mp.get_context('spawn')   # fork is not safe after CUDA initialization
    _queue = ctx.Queue()
    _slots = ctx.Semaphore(QUEUE_LEN_PER_DISPLAY * len(displays))
    specs = _dumps([(d.name, type(d), d._init_args, d._init_kwargs) for d in displays])
    _process = ctx.Process(target=_worker, args=(_queue, _slots, specs, precision, log_level),
                           daemon=True)
    _process.start()


def send(display):
    '''Send the current inputs of *display* to the display process'''
    global _dropped
    if _queue is None:
        return
    if not _process.is_alive():
        stop(display.logger)
        return
    if display.skip_updates and not _slots.acquire(block=False):
        _dropped += 1
        return
    inputs = {k: _to_cpu(v) for k, v in display.local_inputs.items()}
    # Pickle here: the queue would pickle later in a background thread,
    # when a CPU simulation may have already modified the arrays
    _queue.put((display.skip_updates, _dumps((display.name, display.current_time, inputs))))


def stop(logger, timeout=30):
    '''Wait for the display process to draw the pending data, then stop it'''
    global _queue, _process
    if _process is None:
        return
    if _dropped:
        logger.info(f'Async displays: {_dropped} updates skipped because the display process was busy')
    if _process.is_alive():
        _queue.put(None)
        _process.join(timeout)
        _process.terminate()    # no-op if it has exited
    else:
        logger.error(f'Async displays: the display process has exited (exit code {_process.exitcode}), '
                     'displays are not updated')
    # Do not wait at exit for data that a dead process will never read
    _queue.cancel_join_thread()
    _queue.close()
    _queue = _process = None


# Runs only in the spawned display process, which coverage does not follow:
# the display loop is in _worker_loop(), tested directly
def _worker(q, slots, specs, precision, log_level):  # pragma: no cover
    try:
        import specula
        specula.init(-1, precision=precision)
        _worker_loop(q, slots, specs, log_level)
    except KeyboardInterrupt:
        pass


def _worker_loop(q, slots, specs, log_level):
    '''Display loop, run in the display process (or directly by tests)'''
    displays = {}
    for name, klass, args, kwargs in pickle.loads(specs):
        displays[name] = d = klass(*args, **kwargs)
        d.name = name
        d.init_logging(log_level)

    # setup() runs before the first update of each display, when its inputs are available
    not_setup = set(displays)

    # The queue ends with a None terminator
    while True:
        try:
            msgs = [q.get(timeout=0.02)]
        except queue.Empty:
            for d in displays.values():
                d.fig.canvas.flush_events()
            continue
        while msgs[-1] is not None:
            try:
                msgs.append(q.get_nowait())
            except queue.Empty:
                break

        # Apply all pending updates, then draw each figure once
        figs = {}
        for msg in msgs:
            if msg is None:
                break
            skip_updates, data = msg
            if skip_updates:
                slots.release()
            name, t, inputs = pickle.loads(data)
            d = displays[name]
            d.current_time = t
            d.current_time_seconds = d.t_to_seconds(t)
            for k, v in inputs.items():
                d.inputs[k].set([] if v is None else v)
            if name in not_setup:
                not_setup.remove(name)
                d.setup()   # also gets the inputs
            else:
                d.get_all_inputs()
            d.trigger_code()
            figs[d.fig] = d

        for d in figs.values():
            d._safe_draw()

        if msgs[-1] is None:
            break

    for d in displays.values():
        d.finalize()
