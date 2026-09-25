
import sys
import atexit
import logging
import threading
import _thread

from specula.processing_objects.specula_input import SpeculaInput

output_list_for_help = None

PROMPT = 'specula> '


class TerminalInput(SpeculaInput):
    """
    Terminal input processing object. Handles input from a terminal.

    Commands are read in a background thread of the main process.
    When running on an interactive terminal, the prompt is managed by
    prompt_toolkit: the input line is kept at the bottom of the terminal
    and everything written to stdout/stderr (including log messages)
    is printed above it, so that simulation output never gets mixed
    with the text being typed.
    """

    # Override __new__ to make sure that
    # only one instance can be allocated.
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            raise RuntimeError("Only one instance of TerminalInput is allowed")

        cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 output_list: list,
                 target_device_idx: int=None,
                 precision: int =None):
        global output_list_for_help

        """
        output_list: list of strings
            List of output names to be generated
        target_device_idx : int, optional
            Target device index for computation (CPU/GPU). Default is None (uses global setting).
        precision : int, optional
            Precision for computation (0 for double, 1 for single). Default is None
            (uses global setting).
        """
        super().__init__(output_list,
                         target_device_idx=target_device_idx,
                         precision=precision)

        output_list_for_help = output_list
        self.reader = TerminalReader(self.put_input)
        self.reader.start()
        atexit.register(self.reader.stop)

    def finalize(self):
        super().finalize()
        self.reader.stop()
        atexit.unregister(self.reader.stop)
        # Allow a new instance, e.g. for another simulation
        # run in the same Python process
        TerminalInput._instance = None


class TerminalReader:
    """
    Reads commands from the terminal in a daemon thread and
    passes (name, value) pairs to the *put* callable, which may raise
    KeyError or ValueError to reject them.
    """
    def __init__(self, put):
        self.put = put
        self.thread = None
        self._app = None
        self._stopping = False
        self._interactive = False

    def start(self):
        # Decided here, before the thread starts, so that stop() knows
        # whether to wait for it even if called before _run() begins
        self._interactive = _isatty(sys.stdin) and _isatty(sys.stdout)
        self.thread = threading.Thread(target=self._run, name='TerminalInput', daemon=True)
        self.thread.start()

    def stop(self, timeout=1.0):
        '''
        Terminate the prompt (if active) and restore the terminal.
        Safe to call more than once.
        '''
        self._stopping = True
        app = self._app
        if app is not None and app.is_running:
            try:
                app.loop.call_soon_threadsafe(app.exit)
            except Exception:
                pass
        # In plain mode the thread is blocked reading stdin and cannot be
        # interrupted: do not wait for it (it is a daemon thread anyway)
        if self._interactive and self.thread is not None \
                and self.thread is not threading.current_thread():
            self.thread.join(timeout=timeout)

    def _run(self):
        if self._stopping:
            return
        if self._interactive:
            self._run_prompt_toolkit()
        else:
            self._run_plain()

    def _run_plain(self):
        '''
        Fallback for non-interactive stdin (e.g. piped commands):
        no prompt, one command per line.
        '''
        try:
            for line in sys.stdin:
                if self._stopping:
                    break
                self._handle_line(line)
        except (OSError, ValueError):
            # stdin closed or not readable (e.g. under pytest)
            pass

    def _run_prompt_toolkit(self):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.patch_stdout import patch_stdout

        session = PromptSession()
        self._app = session.app

        original_stderr = sys.stderr
        with patch_stdout(raw=True):
            # patch_stdout() sends stderr to the terminal (stdout) as well.
            # If stderr has been redirected away from the terminal
            # (e.g. "2> log.txt"), keep it there, since it cannot
            # interfere with the prompt.
            if not _isatty(original_stderr):
                sys.stderr = original_stderr
            redirected = _redirect_log_handlers(sys.stdout)
            try:
                while not self._stopping:
                    try:
                        line = session.prompt(PROMPT, pre_run=self._exit_if_stopping)
                    except KeyboardInterrupt:
                        # The terminal is in raw mode while the prompt is
                        # active, so Ctrl-C does not generate SIGINT:
                        # forward it to the main thread to keep the usual
                        # behaviour of interrupting the simulation.
                        _thread.interrupt_main()
                        print('Ctrl-C received: terminal input is no longer active')
                        break
                    except EOFError:
                        break
                    if line is None:     # app.exit() called by stop()
                        break
                    self._handle_line(line)
            finally:
                _restore_log_handlers(redirected)
                self._app = None

    def _exit_if_stopping(self):
        # Covers a stop() request arriving before the prompt was running
        if self._stopping:
            self._app.exit()

    def _handle_line(self, line):
        tokens = line.split()
        if len(tokens) == 0:
            return
        elif len(tokens) == 1 and tokens[0] == 'help':
            print_help()
            return
        elif len(tokens) == 1:
            name, value = tokens[0], False
        elif len(tokens) == 2:
            name, value = tokens
        else:
            print('Input not recognized')
            return
        try:
            self.put(name, value)
        except (KeyError, ValueError) as e:
            # Reject bad input immediately, at the prompt
            print(e.args[0])


def _redirect_log_handlers(stream):
    '''
    logging.StreamHandler keeps a reference to the stream it was created
    with (sys.stderr by default), so replacing sys.stdout/sys.stderr is not
    enough to capture log messages. Point all console handlers of the
    root logger to *stream* and return their previous streams.
    Handlers writing to a console stream that is not a terminal
    (e.g. stderr redirected to a file) are left alone.
    '''
    console_streams = (sys.__stdout__, sys.__stderr__)
    redirected = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        if isinstance(handler, logging.StreamHandler) and handler.stream in console_streams \
                and _isatty(handler.stream):
            redirected.append((handler, handler.setStream(stream)))
    return redirected


def _isatty(stream):
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


def _restore_log_handlers(redirected):
    for handler, old_stream in redirected:
        handler.setStream(old_stream)


def print_help():
    print(output_list_for_help)
