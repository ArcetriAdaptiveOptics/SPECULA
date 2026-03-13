import io
import time
import base64
import queue
import pickle
import typing
import multiprocessing as mp
import numpy as np
from contextlib import contextmanager

from flask_socketio import SocketIO

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj

from specula.lib.display_server_api import start_server

class DisplayServer(BaseProcessingObj):
    """
    Display server processing object.
    Copies data objects to a separate process using multiprocessing queues.
    In this instance, the separate process is a Flask web server, but this class
    could be easily made generic to support different kinds of export processes.
    
    This object must *not* be run concurrently with any other in the simulation,
    because it can in some cases temporarily modify the data objects (removing references
    to the xp module to allow pickling)
    It has two modes of operation: 'image' and 'data'. In 'image' mode, it renders and 
    sends the images correspoinding to the displayed data objects. In 'data' mode, it sends
    the raw data to the client, which is responsible for rendering.
    
    """
    def __init__(self,
                 params_dict: dict,
                 input_ref_getter: typing.Callable,
                 output_ref_getter: typing.Callable,
                 info_getter: typing.Callable,
                 host: str = '0.0.0.0',
                 port: int = 0,
                 mode: str = 'image',
    ):
        """
        Note
        ----
            
        This object must *not* be run concurrently with any other in the simulation,
        because it can in some cases temporarily modify the data objects (removing references
        to the xp module to allow pickling)
        """
        super().__init__()
        self.mode = mode
        
        self.qin = mp.Queue()
        self.qout = mp.Queue()

        self.params_dict = params_dict

        self.p = mp.Process(
            target=start_server, 
            args=(
                params_dict, 
                self.qout, 
                self.qin, 
                host, 
                port,
                self.mode
            )
        )
        
        self.p.start()

        # Simulation speed calculation
        self.counter = 0
        self.t0 = time.time()
        self.c0 = self.counter
        self.speed_report = ''

        # Heuristic to detect inputs: they usually start with "in_"
        def data_obj_getter(name):
            try:
                if '.in_' in name:
                    return input_ref_getter(name)
                else:
                    try:
                        return output_ref_getter(name)
                    except ValueError:
                        return input_ref_getter(name)
            except Exception as e:
                print(f"[DisplayServer] Could not get object '{name}': {e}")
                return None

        self.data_obj_getter = data_obj_getter
        self.info_getter = info_getter

        print(f"[DisplayServer] Initialized in {self.mode} mode")

    def _trigger_image_mode(self):

        # Loop over data object requests
        # This loop is guaranteed to find an empty queue sooner or later,
        # thanks to the handshaking with the browser code, that will
        # avoid sending new requests until the None terminator is received
        # by the browser itself.

        while True:
            try:
                request_data = self.qin.get(block=False)
                client_id, object_names = request_data
                
                for name in object_names:
                    dataobj = self.data_obj_getter(name)
                    if dataobj is None:                        
                        continue

                    try:
                        if isinstance(dataobj, list):
                            dataobj_cpu = [x.copyTo(-1) for x in dataobj]
                        else:
                            dataobj_cpu = dataobj.copyTo(-1)

                        with remove_xp_np(dataobj_cpu) as cleaned_dataobj:
                            obj_bytes = pickle.dumps(cleaned_dataobj)
                            self.qout.put(('image_data', client_id, name, obj_bytes))
                    except Exception as e:
                        print(f"[DisplayServer][ImageMode] Error processing {name}: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.qout.put(('image_terminator', client_id, None, self.speed_report))
                
            except queue.Empty:
                return
            except Exception as e:
                print(f"[DisplayServer][ImageMode] Error processing request: {e}")
                import traceback
                traceback.print_exc()

    def trigger(self):
        t1 = time.time()
        self.counter += 1
        if t1 - self.t0 >= 1:
            niters = self.counter - self.c0
            speed = niters / (t1 - self.t0)
            self.c0 = self.counter
            self.t0 = t1
            name, status = self.info_getter()
            status_report = f"{status} - {speed:.2f} Hz"
            try:
                self.qout.put((name, status_report))
            except Exception as e:
                print(f"[SIMULATION][{self.__class__.__name__}] Error putting status: {e}")

        if self.mode == 'image':
            self._trigger_image_mode()
        else:
            self._trigger_data_mode()

    def _trigger_data_mode(self):

        processed = 0
        while True:
            try:
                request_data = self.qin.get(block=False)
                processed += 1
                
                client_id, object_names = request_data
                
                responses = []
                for i, name in enumerate(object_names):
                    dataobj = self.data_obj_getter(name)
                    if dataobj is None:
                        dataobj = BaseValue(value=None)

                    try:
                        if isinstance(dataobj, list):
                            dataobj_cpu = [x.copyTo(-1) for x in dataobj]
                        else:
                            dataobj_cpu = dataobj.copyTo(-1)
                        
                        def _set_xp_attribute(obj):
                            if not hasattr(obj, 'xp'):
                                obj.xp = np
                        
                        if isinstance(dataobj_cpu, list):
                            for obj in dataobj_cpu:
                                _set_xp_attribute(obj)
                        else:
                            _set_xp_attribute(dataobj_cpu)
                        
                        processed_data = self._process_for_dpg(dataobj_cpu, name)
                        responses.append((name, processed_data))
                        
                    except Exception as e:
                        print(f"[SIMULATION][DataMode] Error preparing data for {name}: {e}")
                        import traceback
                        traceback.print_exc()
                        responses.append((name, {
                            'type': 'error',
                            'data': str(e),
                            'name': name
                        }))

                for name, data in responses:
                    try:
                        self.qout.put(('data_response', client_id, name, data))
                    except Exception as e:
                        print(f"[SIMULATION][DataMode] Error putting response: {e}")
                
                try:
                    self.qout.put(('terminator', client_id, None, self.speed_report))
                except Exception as e:
                    print(f"[SIMULATION][DataMode] Error putting terminator: {e}")
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"[SIMULATION][DataMode] Error processing request: {e}")
                import traceback
                traceback.print_exc()
                break


    def _process_for_dpg(self, dataobj, name):
        """Process data object for DPG plotting."""
        try:
            def _safe_extract(obj):
                try:
                    # Here we rely on the fact that all data objects have an array_for_display method that 
                    # returns a CPU array or a list of CPU arrays, and that this method is implemented in a 
                    # way that it can be called safely even if the object has some non-picklable attributes (like xp)
                    if hasattr(obj, 'array_for_display'):
                        return obj.array_for_display()
                        
                except Exception as e:
                    print(f"Error extracting array from {type(obj).__name__}: {e}")
                    return None
                return None
            
            if isinstance(dataobj, list):
                arrays = []
                for obj in dataobj:
                    array = _safe_extract(obj)
                    if array is not None:
                        if not isinstance(array, np.ndarray):
                            array = np.array(array)
                        # Ensure it's float for plotting
                        if np.issubdtype(array.dtype, np.integer):
                            array = array.astype(np.float32)
                        arrays.append(array)
                
                if arrays:
                    # For 1D lists, check if we should combine them
                    all_1d = all(arr.ndim == 1 for arr in arrays)
                    same_length = all(arr.shape[0] == arrays[0].shape[0] for arr in arrays) if len(arrays) > 1 else True
                    
                    if all_1d and same_length and len(arrays) == 1:
                        # Single 1D array
                        array = arrays[0]
                        return {
                            'type': '1d_array',
                            'data': array.tolist(),
                            'shape': array.shape,
                            'dtype': str(array.dtype),
                            'name': name
                        }
                    elif all_1d and same_length:
                        # Multiple 1D arrays of same length - stack them
                        stacked = np.stack(arrays, axis=-1)
                        return {
                            'type': '2d_array',
                            'data': stacked.tolist(),
                            'shape': stacked.shape,
                            'dtype': str(stacked.dtype),
                            'name': name
                        }
                    else:
                        # Mixed dimensions
                        arrays_as_lists = [arr.tolist() for arr in arrays]
                        return {
                            'type': 'multi_data',
                            'data': arrays_as_lists,
                            'shapes': [arr.shape for arr in arrays],
                            'dtypes': [str(arr.dtype) for arr in arrays],
                            'name': name
                        }
            else:
                array = _safe_extract(dataobj)
                if array is not None:
                    if not isinstance(array, np.ndarray):
                        array = np.array(array)
                    
                    # Ensure it's float for plotting
                    if np.issubdtype(array.dtype, np.integer):
                        array = array.astype(np.float32)
                    
                    # Determine the type
                    if array.ndim == 0:
                        data_type = 'scalar'
                    elif array.ndim == 1:
                        data_type = '1d_array'
                    elif array.ndim == 2:
                        data_type = '2d_array'
                    else:
                        data_type = 'nd_array'
                    
                    return {
                        'type': data_type,
                        'data': array.tolist(),
                        'shape': array.shape,
                        'dtype': str(array.dtype),
                        'name': name
                    }
            
            return {
                'type': 'unknown',
                'data': None,
                'name': name
            }
            
        except Exception as e:
            print(f"Error processing data for DPG ({name}): {e}")
            import traceback
            traceback.print_exc()
            return {
                'type': 'error',
                'data': str(e),
                'name': name
            }


    def finalize(self):
        if hasattr(self, 'p') and self.p.is_alive():
            self.p.terminate()
            self.p.join()
            print(f"[DisplayServer] Server process terminated")


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

    deleted = _remove(obj)    
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
