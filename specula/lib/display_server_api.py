import io
import os
import socket
import threading
import time
import queue
import pickle
import base64
import multiprocessing as mp

from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room
import socketio
import socketio.exceptions
from specula.display.data_plotter import DataPlotter

if os.name == 'nt':
    async_mode = 'threading'
else:
    try:
        import eventlet
        eventlet.monkey_patch()
        async_mode = 'eventlet'
    except ImportError:
        try:
            import gevent
            from gevent import monkey
            monkey.patch_all()
            async_mode = 'gevent'
        except ImportError:
            async_mode = 'threading'

base_dir = os.path.abspath(os.path.dirname(__file__))
templates_dir = os.path.join(base_dir, "..", "scripts", "templates")

app = Flask('Specula_display_server', template_folder=templates_dir)

if async_mode:
    sio = SocketIO(app, cors_allowed_origins="*", async_mode=async_mode)
else:
    sio = SocketIO(app, cors_allowed_origins="*")

server = None

def encode(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    imgB64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return imgB64

def filter_params_for_display(params_dict):
    display_params = {}
    for k, v in params_dict.items():
        if 'class' in v:
            if v['class'] == 'DataStore':
                continue
            if 'inputs' not in v and 'outputs' not in v:
                continue
        display_params[k] = v
    return display_params

def setup_image_mode_handlers():
    @sio.on('connect')
    def handle_image_connect(*args):        
        client_id = request.sid
        server.client_types[client_id] = 'web'
        
        display_params = filter_params_for_display(server.params_dict)
        
        sio.emit('params', display_params, room=client_id)

    @sio.on('newdata')
    def handle_image_newdata(args):        
        client_id = request.sid

        if client_id not in server.t0:
            server.t0[client_id] = time.time()

        server.request_queue.put((client_id, args))
        join_room(client_id)

    @sio.on('disconnect')
    def handle_image_disconnect():        
        client_id = request.sid
        if client_id in server.client_types:
            del server.client_types[client_id]
        if client_id in server.t0:
            del server.t0[client_id]

def setup_data_mode_handlers():
    @sio.on('connect')
    def handle_data_connect(*args):        
        client_id = request.sid
        
        server.client_types[client_id] = 'dpg'
        server.client_subscriptions[client_id] = set()
        
        display_params = filter_params_for_display(server.params_dict)
        
        sio.emit('params', display_params, room=client_id)

    @sio.on('newdata')
    def handle_data_newdata(args):        
        client_id = request.sid
        
        server.client_subscriptions[client_id] = set(args)
        
        if client_id not in server.t0:
            server.t0[client_id] = time.time()

        server.request_queue.put((client_id, args))
        server.last_request_time = time.time()
        server.last_request_outputs = args
        join_room(client_id)

    @sio.on('disconnect')
    def handle_data_disconnect():        
        client_id = request.sid
        if client_id in server.client_types:
            del server.client_types[client_id]
        if client_id in server.t0:
            del server.t0[client_id]
        if client_id in server.client_subscriptions:
            del server.client_subscriptions[client_id]

    @sio.on('get_params')
    def handle_get_params():        
        client_id = request.sid
        
        display_params = {}
        for k, v in server.params_dict.items():
            if 'class' in v:
                if v['class'] == 'DataStore':
                    continue
                if 'inputs' not in v and 'outputs' not in v:
                    continue
            display_params[k] = v
        
        sio.emit('params', display_params, room=client_id)

    @sio.on('set_client_type')
    def handle_set_client_type(data):        
        client_id = request.sid
        client_type = data.get('type', 'dpg')
        server.client_types[client_id] = client_type

    @sio.on('unsubscribe')
    def handle_unsubscribe(data):        
        client_id = request.sid
        output_to_unsubscribe = data.get('output')
        
        if client_id in server.client_subscriptions and output_to_unsubscribe in server.client_subscriptions[client_id]:
            server.client_subscriptions[client_id].remove(output_to_unsubscribe)

class ImageFlaskServer:
    def __init__(self, params_dict: dict,
                 status_queue: mp.Queue,
                 request_queue: mp.Queue,
                 host: str = '0.0.0.0',
                 port: int = 5000,
                 ):
        self.params_dict = params_dict
        self.t0 = {}
        self.status_queue = status_queue
        self.request_queue = request_queue
        self.plotters = {}
        self.display_lock = threading.Lock()
        self.host = host
        self.port = port
        self.actual_port = None
        self.frontend_connected = False
        self.client_types = {}
        
        self.response_handler_thread = threading.Thread(target=self.handle_image_responses, daemon=True)
        self.response_handler_thread.start()

    def run(self):
        t = threading.Thread(target=self.status_update, args=(sio,))
        t.start()

        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))
                address, port = s.getsockname()
                s.close()
                self.actual_port = port
        else:
            self.actual_port = self.port

        sio.run(app, host=self.host, allow_unsafe_werkzeug=True, port=self.actual_port)

    def shutdown(self):
        import os
        os._exit(0)

    def status_update(self, sio):
        sio_client = socketio.Client()
        
        def connect():
            if not self.frontend_connected:
                try:
                    sio_client.connect(f'http://{self.host}:{self.port}')
                    self.frontend_connected = True
                except:
                    pass

        while True:
            try:
                item = self.status_queue.get(timeout=60)
                
                if isinstance(item, tuple) and len(item) == 2:
                    name, data = item
                    
                    if data is None:
                        break
                    
                    try:
                        connect()
                        sio_client.emit('simul_update', data={'name': name, 'status': data, 'port': self.actual_port})
                    except (socketio.exceptions.ConnectionError, socketio.exceptions.BadNamespaceError):
                        self.frontend_connected = False
                        time.sleep(1)
                else:
                    self.status_queue.put(item)
                    time.sleep(0.001)
                    
            except queue.Empty:
                self.shutdown()
            except EOFError:
                self.shutdown()
            except Exception as e:
                self.shutdown()

    def handle_image_responses(self):
        while True:
            try:
                item = self.status_queue.get(timeout=1)
                
                if isinstance(item, tuple) and len(item) == 4:
                    response_type, client_id, name, data = item
                    
                    if response_type == 'image_data':
                        if client_id in self.client_types:
                            try:
                                dataobj = pickle.loads(data)
                                
                                with self.display_lock:
                                    fig = DataPlotter.plot_best_effort(name, dataobj)
                                
                                sio.emit('plot', {'name': name, 'imgdata': encode(fig)}, room=client_id)
                            except Exception as e:
                                print(f"[SERVER][ImageMode] Error processing image data for {name}: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    elif response_type == 'image_terminator':
                        if client_id in self.client_types:
                            try:
                                sio.emit('speed_report', data, room=client_id)
                                t1 = time.time()
                                t0 = self.t0.get(client_id, t1)
                                freq = 1.0 / (t1 - t0) if t1 != t0 else 0
                                sio.emit('done', f'Display rate: {freq:.2f} Hz', room=client_id)
                                self.t0[client_id] = t1
                            except Exception as e:
                                print(f"[SERVER][ImageMode] Error sending terminator to {client_id}: {e}")
                else:
                    self.status_queue.put(item)
                    time.sleep(0.001)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SERVER][ImageMode] Error in image response handler: {e}")
                import traceback
                traceback.print_exc()

class DataFlaskServer:
    def __init__(self, params_dict: dict,
                 status_queue: mp.Queue,
                 request_queue: mp.Queue,
                 host: str = '0.0.0.0',
                 port: int = 5000,
                 ):
        self.params_dict = params_dict
        self.t0 = {}
        self.status_queue = status_queue
        self.request_queue = request_queue
        self.display_lock = threading.Lock()
        self.host = host
        self.port = port
        self.actual_port = None
        self.frontend_connected = False
        self.client_types = {}
        self.last_request_time = 0
        self.last_request_outputs = []
        self.client_subscriptions = {}

        self.response_handler_thread = threading.Thread(target=self.handle_responses, daemon=True)
        self.response_handler_thread.start()

    def run(self):
        t = threading.Thread(target=self.status_update, args=(sio,))
        t.start()

        if self.port == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', 0))
                address, port = s.getsockname()
                s.close()
                self.actual_port = port
        else:
            self.actual_port = self.port

        sio.run(app, host=self.host, allow_unsafe_werkzeug=True, port=self.actual_port)

    def shutdown(self):
        import os
        os._exit(0)

    def status_update(self, sio):
        sio_client = socketio.Client()
        
        def connect():
            if not self.frontend_connected:
                try:
                    sio_client.connect(f'http://{self.host}:{self.port}')                    
                    self.frontend_connected = True
                except:
                    pass

        while True:
            try:
                item = self.status_queue.get(timeout=60)
                
                if isinstance(item, tuple) and len(item) == 2:
                    name, data = item
                    
                    if name == 'data_response' or name == 'terminator':
                        self.status_queue.put(item)
                        continue
                    
                    if data is None:
                        break
                    
                    try:
                        connect()
                        sio_client.emit('simul_update', data={'name': name, 'status': data, 'port': self.actual_port})
                    except (socketio.exceptions.ConnectionError, socketio.exceptions.BadNamespaceError):
                        self.frontend_connected = False
                        time.sleep(1)
                else:
                    self.status_queue.put(item)
                    time.sleep(0.001)
                    
            except queue.Empty:
                self.shutdown()
            except EOFError:
                self.shutdown()
            except Exception as e:
                self.shutdown()

    def handle_responses(self):
        while True:
            try:
                item = self.status_queue.get(timeout=1)
                
                if isinstance(item, tuple) and len(item) == 2:
                    self.status_queue.put(item)
                    continue
                elif isinstance(item, tuple) and len(item) == 4:
                    response_type, client_id, name, data = item
                    
                    if response_type == 'data_response':
                        if client_id in self.client_types:
                            try:
                                sio.emit('data_update', {
                                    'name': name,
                                    'data': data
                                }, room=client_id)
                            except Exception as e:
                                print(f"[SERVER][DataMode] Error emitting to {client_id}: {e}")
                    
                    elif response_type == 'terminator':
                        if client_id in self.client_types:
                            try:
                                sio.emit('speed_report', data, room=client_id)
                                t1 = time.time()
                                t0 = self.t0.get(client_id, t1)
                                freq = 1.0 / (t1 - t0) if t1 != t0 else 0
                                sio.emit('done', f'Display rate: {freq:.2f} Hz', room=client_id)
                                self.t0[client_id] = t1
                            except Exception as e:
                                print(f"[SERVER][DataMode] Error sending terminator to {client_id}: {e}")
                else:
                    print(f"[SERVER][DataMode] Unknown item format: {item}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SERVER][DataMode] Error in response handler: {e}")
                import traceback
                traceback.print_exc()

def start_server(params_dict, status_queue, request_queue, host, port, mode):
    global server
    
    safe_params_dict = {}
    for k, v in params_dict.items():
        if isinstance(v, dict):
            safe_v = {}
            for key, val in v.items():
                if isinstance(val, (str, int, float, bool, type(None), list, dict)):
                    safe_v[key] = val
            safe_params_dict[k] = safe_v
        else:
            safe_params_dict[k] = v
    
    sio.handlers = {}
    sio.namespace_handlers = {'/': {}}
    
    if mode == 'image':
        server = ImageFlaskServer(
            params_dict=safe_params_dict,
            status_queue=status_queue,
            request_queue=request_queue,
            host=host,
            port=port
        )
        setup_image_mode_handlers()
    else:
        server = DataFlaskServer(
            params_dict=safe_params_dict,
            status_queue=status_queue,
            request_queue=request_queue,
            host=host,
            port=port
        )
        setup_data_mode_handlers()
    
    server.run()

@app.route('/')
def index():
    return render_template('specula_display.html')

@app.route('/status')
def status():    
    if not server:
        return {'status': 'not_initialized'}
    
    return {
        'status': 'running',
        'mode': 'image' if isinstance(server, ImageFlaskServer) else 'data',
        'port': server.actual_port if server else 'unknown',
        'connected_clients': len(server.client_types) if server else 0,
        'clients': list(server.client_types.items()) if server else [],
        'status_queue_size': server.status_queue.qsize() if server else 0,
        'request_queue_size': server.request_queue.qsize() if server else 0
    }