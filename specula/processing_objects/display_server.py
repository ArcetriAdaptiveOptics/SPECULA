
import io
import os
import socket
import threading
import time
import base64
import queue
import pickle
import typing
import multiprocessing as mp
from contextlib import contextmanager

from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room
import socketio
import socketio.exceptions

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.display.data_plotter import DataPlotter

# Use a manager to create queues so that they can be
# pickled between processes (ordinary mp.Queue cannot)
manager = mp.Manager()


class DisplayServer(BaseProcessingObj):
    '''
    Copies data objects to a separate process using multiprocessing queues.
    In this instance, the separate process is a Flask web server, but this class
    could be easily made generic to support different kinds of export processes.
    
    This object must *not* be run concurrently with any other in the simulation,
    because it can in some cases temporarily modify the data objects (removing references
    to the xp module to allow pickling)
    '''
    def __init__(self,
                 params_dict: dict,
                 input_ref_getter: typing.Callable,
                 output_ref_getter: typing.Callable,
                 info_getter: typing.Callable,
                 host: str='0.0.0.0',
                 port: int=0,    # Autoselect
    ):
        super().__init__()
        self.qin = manager.Queue()    # Queue to receive dataobj requests from the Flask webserver
        self.qout = manager.Queue()   # Queue to send regular status updates

        # Flask-SocketIO web server
        self.p = mp.Process(target=start_server, args=(params_dict, self.qout, self.qin, host, port))  # qin becomes qout for the server
        self.p.start()

        # Simulation speed calculation
        self.counter = 0
        self.t0 = time.time()
        self.c0 = self.counter
        self.speed_report = ''

        # Heuristic to detect inputs: they usually start with "in_"
        def data_obj_getter(name):
            if '.in_' in name:
                return input_ref_getter(name)
            else:
                try:
                    return output_ref_getter(name)
                except ValueError:
                    # Try inputs as well
                    return input_ref_getter(name)

        self.data_obj_getter = data_obj_getter
        self.info_getter = info_getter

    def trigger(self):
        t1 = time.time()
        self.counter += 1
        if t1 - self.t0 >= 1:
            niters = self.counter - self.c0
            speed = niters / (t1-self.t0)
            self.c0 = self.counter
            self.t0 = t1
            name, status = self.info_getter()
            status_report = f"{status} - {speed:.2f} Hz"
            self.qout.put((name, status_report))

        # Loop over data object requests
        # This loop is guaranteed to find an empty queue sooner or later,
        # thanks to the handshaking with the browser code, that will
        # avoid sending new requests until the None terminator is received
        # by the browser itself.

        while True:
            try:
                object_names, response_queue = self.qin.get(block=False)
            except queue.Empty:
                return

            for name in object_names:

                # Find the requested object, make sure it's on CPU,
                # and remove xp/np modules to prepare for pickling
                dataobj = self.data_obj_getter(name)
                if dataobj is None:
                    dataobj = BaseValue(value=None)

                if isinstance(dataobj, list):
                    dataobj_cpu = [x.copyTo(-1) for x in dataobj]
                else:
                    dataobj_cpu = dataobj.copyTo(-1)

                # Use an object copy without references to xp and np
                with remove_xp_np(dataobj_cpu) as cleaned_dataobj:

                    # Double pickle trick (we pickle, and then qout will pickle again)
                    # to avoid some problems
                    # with serialization of modules, which apparently
                    # are still present even after the xp and np removal

                    obj_bytes = pickle.dumps(cleaned_dataobj)
                    response_queue.put((name, obj_bytes))

            # Terminator
            response_queue.put((None, self.speed_report))

    def finalize(self):
        self.p.terminate()


base_dir = os.path.abspath(os.path.dirname(__file__))
templates_dir = os.path.join(base_dir, "..", "scripts", "templates")

# Global variables used by Flask-SocketIO            
app = Flask('Specula_display_server', template_folder=templates_dir)
sio = SocketIO(app)
print("Template search paths:", app.jinja_loader.searchpath)
server = None


class FlaskServer():
    '''
    Flask-SocketIO web server
    '''
    def __init__(self, params_dict: dict,
                 qin: mp.Queue,
                 qout: mp.Queue,
                 host: str='0.0.0.0',
                 port: int=5000,
                 ):
        self.params_dict = params_dict
        self.t0 = {}
        self.qin = qin
        self.qout = qout
        self.plotters = {}
        self.display_lock = threading.Lock()
        self.host = host
        self.port = port
        self.actual_port = None  # Filled in later
        self.frontend_connected = False

    def run(self):
        '''
        Run the main server and a regular status update in a separate thread
        '''
        t = threading.Thread(target=self.status_update, args=(sio,))
        t.start()

        # If port == 0 (auto), we need to know which one is selected, but Flask won't tell us.
        # Therefore, find one manually and then tell Flask to use it.
        # There is a minor race condition here (if the port is re-used in the meantime).
        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))   # Auto-select one port
                address, port = s.getsockname()
                s.close()  # Release the port
                self.actual_port = port
        else:
            self.actual_port = self.port

        sio.run(app, host=self.host, allow_unsafe_werkzeug=True, port=self.actual_port)

    def shutdown(self):
        '''Force process stop'''
        import os
        os._exit(0)

    def status_update(self, sio):
        sio_client = socketio.Client()
        def connect():
            if not self.frontend_connected:
                sio_client.connect('http://localhost:8080')  # TODO get port number from os.environ
                self.frontend_connected = True

        while True:
            try:
                name, data = self.qin.get(timeout=60)
            except queue.Empty:
                # Timeout. Problem in the processing object. We bail out
                print('No updates from simulation after 60 seconds, stopping status updates')
                self.shutdown()
            except EOFError:
                print('Display server terminated, exiting web server')
                self.shutdown()

            # Terminator
            if data is None:
                break
            # Fault-tolerant publishing
            try:
                connect()
                sio_client.emit('simul_update', data={'name': name, 'status': data, 'port': self.actual_port})
            except (socketio.exceptions.ConnectionError, socketio.exceptions.BadNamespaceError):
                # No frontend server running, will try again next time
                self.frontend_connected = False
                time.sleep(1)

    @sio.on('newdata')
    def handle_newdata(args):
        '''Request for new data from the browser.
        1) Queue all requested object names
        2) Get back all data objects, plot them, and send them back to the browser
        '''
        print(args)
        client_id = request.sid
        response_queue = manager.Queue() # Separate response queue for each client

        if client_id not in server.t0:
            server.t0[client_id] = time.time()

        # Queue data object requests to the simulation Processing object
        server.qout.put((args, response_queue))

        # Function to emit results back to the client
        def emit_results():
            while True:
                try:
                    name, obj_bytes = response_queue.get(timeout=30)
                except queue.Empty:
                    # Timeout. Problem in the processing object. We bail out
                    break

                if name is None: # Terminator
                    speed_report = obj_bytes
                    sio.emit('speed_report', speed_report)
                    break

                dataobj = pickle.loads(obj_bytes)

                # We lock because multiple clients might be requesting
                # the same plot and our plotting functions have a global state.
                with server.display_lock:
                    fig = DataPlotter.plot_best_effort(name, dataobj)

                sio.emit('plot', {'name': name, 'imgdata': encode(fig) }, room=client_id)
            done()

        def done():
            t1 = time.time()
            t0 = server.t0[client_id]
            freq = 1.0 / (t1 - t0) if t1 != t0 else 0
            sio.emit('done', f'Display rate: {freq:.2f} Hz', room=client_id)
            server.t0[client_id] = t1

        # Emit results in a separate thread so it doesn't block the event loop
        join_room(client_id)
        if len(args) > 0:
            threading.Thread(target=emit_results).start()
        else:
            done()

    @sio.on('connect')
    def handle_connect(*args):
        '''On connection, send the entire parameter dictionary
        so that the browser can refresh the view'''
        client_id = request.sid

        # Exclude DataStore since its input_list has a different format
        # and cannot be displayed at the moment

        # TODO .inf values cannot be parsed by the Javascript client
        # For the moment, these are only present in the Source object,
        # which is not a processing object and so is skipped.
        display_params = {}
        for k, v in server.params_dict.items():
            if 'class' in v:
                if v['class'] == 'DataStore':
                    continue
                if 'inputs' not in v and 'outputs' not in v:
                    continue
            display_params[k] = v
        sio.emit('params', display_params, room=client_id)

    @app.route('/')
    def index():
        return render_template('specula_display.html')


def start_server(params_dict, qin, qout, host, port):
    global server
    server = FlaskServer(params_dict, qin, qout, host=host, port=port)
    server.run()


@contextmanager
def remove_xp_np(obj):
    '''Temporarily remove any instance of xp and np modules
    The removed modules are put back when exiting the context manager.
 
    Works recursively on object lists
    '''
    def _remove(obj):
        attrnames = ['xp', 'np']
        # Recurse into lists
        if isinstance(obj, list):
            return list(map(_remove, obj))

        # Remove xp and np and return the deleted ones
        deleted = {}
        for attrname in attrnames:
            if hasattr(obj, attrname):
                deleted[attrname] = getattr(obj, attrname)
                delattr(obj, attrname)
        return deleted

    def _putback(args):
        obj, deleted = args

        # Recurse into lists
        if isinstance(obj, list):
            _ = list(map(_putback, zip(obj, deleted)))
            return
        for k, v in deleted.items():
            setattr(obj, k, v)

    deleted =_remove(obj)    
    yield obj
    _putback((obj, deleted))


def encode(fig):
    '''
    Encode a PNG image for web display
    '''
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    imgB64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return imgB64
