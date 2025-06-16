
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from specula.lib.process_utils import daemonize, killProcessByName


app = Flask('Specula_frontend', template_folder = os.path.join(os.path.dirname(__file__), 'templates'))
socketio = SocketIO(app)
try:
    port = os.environ['SPECULA_PORT']
except KeyError:
    port = 8080

'''
From clients:

import socketio
sio_client = socketio.Client()
sio_client.connect('http://localhost:8080')
sio_client.emit('msg', {"key": "value"})
'''

@socketio.on('simul_update')
def simul_update(data):
    '''
    Message received from running simulations with: (name, data, port)
    '''
    client_id = request.sid
    print(client_id, data)
    socketio.emit('simul_update', data)

@app.route('/')
def index():
    return render_template('specula_frontend.html')

def start():
    print(f'Starting SPECULA frontend server on port {port}')
    daemonize()
    socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)    

def stop():
    killProcessByName('specula_frontend_start')
   
if __name__ == '__main__':
    start()

