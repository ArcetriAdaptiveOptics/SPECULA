'''
Sends in flight between MPI processes, and waits that keep them moving.

Outputs are sent to other processes with non-blocking sends (MPI Isend). The
host copy of each output is kept here until its send completes, so that the
sending object can go on with the next steps.

MPI moves the data of a large message (above the "eager" limit of the
transport, typically some kB) in a rendezvous: the receiver must post the
receive, and then, unless the transport lets the receiver pull the data by
itself (single-copy shared memory between processes of the same node), the
sender must enter an MPI call to push it. A process that waits for its GPU
does not call MPI, so its sends would stay where they are, and the receiving
process would wait too. Therefore, when a process with sends in flight waits
for its GPU through wait_event(), wait_stream() or wait_current_stream(), it
polls the GPU and tests its sends, instead of blocking. Without sends in
flight, these waits block as usual.

The sends of at most STEPS_IN_FLIGHT steps can be pending on one channel
(destination and tag): before sending a new step, throttle() waits for the
older ones. This bounds the host memory held by the sends of a process that
runs ahead of its receivers. It cannot deadlock: the processes receive in the
same global order (steps, then trigger levels), so a receive of an older step
never waits for this send.

Environment variables (read at each use):

* ITERAS_MPI_SEND_MODE: 'nonblocking' (default) or 'buffered' (the previous
  behavior: MPI Ibsend through the buffer attached by iteras.main_simul(),
  never tested again, whose data moves only when the sender calls MPI);
* ITERAS_MPI_POLL: '0' to block in the waits even with sends in flight.
'''
import time

import iteras
from iteras import registry
from iteras.log import get_iteras_logger

POLL_INTERVAL = 50e-6       # seconds between two polls of the GPU and of the sends
STEPS_IN_FLIGHT = 2         # steps whose sends may be pending on one channel
WARN_INTERVAL = 60.0        # seconds of waiting between two warnings

_logger = get_iteras_logger(__name__)


class _Send:
    __slots__ = ('request', 'buffer', 'channel', 'step')

    def __init__(self, request, buffer, channel, step):
        self.request = request
        self.buffer = buffer        # must stay alive until the send completes
        self.channel = channel
        self.step = step


_pending = []


def buffered_sends():
    '''True if ITERAS_MPI_SEND_MODE is 'buffered' (the previous behavior)'''
    mode = registry.getenv('ITERAS_MPI_SEND_MODE', 'nonblocking')
    if mode not in ('nonblocking', 'buffered'):
        raise ValueError(f"ITERAS_MPI_SEND_MODE must be 'nonblocking' or 'buffered', got {mode!r}")
    return mode == 'buffered'


def polling_enabled():
    '''False if ITERAS_MPI_POLL is '0': waits for the GPU always block'''
    return registry.getenv('ITERAS_MPI_POLL', '1') != '0'


def pending():
    '''Number of sends in flight'''
    return len(_pending)


def isend(buffer, dest, tag, step):
    '''
    Send *buffer* (a host array, not modified afterwards) to rank *dest* with
    *tag*, as the data of time step *step*, without waiting for the receiver.
    '''
    if buffered_sends():
        iteras.process_comm.Ibsend(buffer, dest=dest, tag=tag)
        return
    throttle((dest, tag), step)
    request = iteras.process_comm.Isend(buffer, dest=dest, tag=tag)
    _pending.append(_Send(request, buffer, (dest, tag), step))


def progress():
    '''Test the sends in flight, which lets MPI move their data. Return how many are left.'''
    if _pending:
        _pending[:] = [s for s in _pending if not s.request.Test()]
    return len(_pending)


def _wait(done, what):
    '''Poll until done() is True, testing the sends meanwhile'''
    start = last_warning = time.monotonic()
    while not done():
        progress()
        time.sleep(POLL_INTERVAL)
        now = time.monotonic()
        if now - last_warning > WARN_INTERVAL:
            channels = sorted({s.channel for s in _pending})
            _logger.warning(f'Waiting since {now - start:.0f} s for {what}. '
                            f'MPI sends pending to (rank, tag): {channels}')
            last_warning = now


def throttle(channel, step):
    '''Wait until fewer than STEPS_IN_FLIGHT steps other than *step* are pending on *channel*'''
    def older():
        return {s.step for s in _pending if s.channel == channel and s.step != step}
    if len(older()) >= STEPS_IN_FLIGHT:
        _wait(lambda: len(older()) < STEPS_IN_FLIGHT,
              f'the receiver of the sends to (rank, tag) {channel}')


def wait_event(event):
    '''
    Wait for *event* (a CUDA event, or any object with "done" and "synchronize()").
    With sends in flight, poll it and test the sends meanwhile.
    '''
    if not _pending or not polling_enabled():
        event.synchronize()
        return
    while not event.done:
        if not progress():
            event.synchronize()
            return
        time.sleep(POLL_INTERVAL)


def wait_stream(stream):
    '''Wait for the work queued on a CUDA *stream*, like stream.synchronize()'''
    if not _pending or not polling_enabled():
        stream.synchronize()
        return
    wait_event(stream.record())


def wait_current_stream(device_idx):
    '''
    With sends in flight, wait (polling) for the work queued on the current
    stream of GPU *device_idx*, so that a copy to the host made after this
    call does not block for long. Without sends in flight, return at once.
    '''
    if not _pending or not polling_enabled():
        return
    with iteras.cp.cuda.Device(device_idx):
        event = iteras.cp.cuda.get_current_stream().record()
    wait_event(event)


def wait_all():
    '''Wait for all the sends in flight (end of the simulation)'''
    if _pending:
        _wait(lambda: not progress(), 'the receivers at the end of the simulation')


def reset():
    '''Forget the sends in flight (tests only)'''
    _pending.clear()
