from astropy.io import fits

from specula.base_time_obj import BaseTimeObj
from specula import default_target_device, cp
from specula import show_in_profiler
from specula.connections import InputValue, InputList

class BaseProcessingObj(BaseTimeObj):
    
    _streams = {}
    
    def __init__(self, target_device_idx=None, precision=None):
        """
        Initialize the base processing object.

        Parameters:
        precision (int, optional): if None will use the global_precision, otherwise pass 0 for double, 1 for single
        target_device_idx (int, optional): if None will use the default_target_device_idx, otherwise pass -1 for cpu, i for GPU of index i
        """
        BaseTimeObj.__init__(self, target_device_idx=target_device_idx, precision=precision)

        self.current_time = 0
        self.current_time_seconds = 0

        self._verbose = 0
        
        # Stream/input management
        self.stream  = None
        self.ready = False
        self.cuda_graph = None

        # Will be populated by derived class
        self.inputs = {}
        self.local_inputs = {}
        self.last_seen = {}
        self.outputs = {}

    def checkInputTimes(self):        
        if len(self.inputs)==0:
            return True
        for input_name, input_obj in self.inputs.items():
            if type(input_obj) is InputValue:
                if input_name not in self.last_seen and input_obj.get_time() is not None and input_obj.get_time() >= 0:  # First time
                    return True
                if input_name in self.last_seen and input_obj.get_time() > self.last_seen[input_name]:
                    return True
            elif type(input_obj) is InputList:
                if input_name not in self.last_seen:
                    for tt in input_obj.get_time():
                        if tt >= 0:
                            return True
#                if input_name not in self.last_seen and input_obj.get_time() >= 0:  # First time
#                    return True
                else:
                    for tt, last in zip(input_obj.get_time(), self.last_seen[input_name]):
                        if tt > last:
                            return True
        return False

    def prepare_trigger(self, t):
        self.current_time_seconds = self.t_to_seconds(self.current_time)
        for input_name, input_obj in self.inputs.items():
            if type(input_obj) is InputValue:
                self.local_inputs[input_name] = input_obj.get(self.target_device_idx)
                if self.local_inputs[input_name] is not None:
                    self.last_seen[input_name] = self.local_inputs[input_name].generation_time
            elif type(input_obj) is InputList:
                self.local_inputs[input_name] = []
                self.last_seen[input_name] = []
                input_list = input_obj.get(self.target_device_idx)
                if input_list is not None:
                    for tt in input_list:
                        self.local_inputs[input_name].append(tt)
                        if self.local_inputs[input_name] is not None:
                            self.last_seen[input_name].append(tt.generation_time)

    def trigger_code(self):
        '''
        Any code implemented by derived classes must:
        1) only perform GPU operations using the xp module
           on arrays allocated with self.xp
        2) avoid any explicity numpy or normal python operation.
        3) NOT use any value in variables that are reallocated by prepare_trigger() or post_trigger(),
           and in general avoid any value defined outside this class (like object inputs)
        
        because if stream capture is used, a CUDA graph will be generated that will skip
        over any non-GPU operation and re-use GPU memory addresses of its first run.
        
        Defining local variables inside this function is OK, they will persist in GPU memory.
        '''
        pass

    def post_trigger(self):
        if self.target_device_idx>=0 and self.cuda_graph:
            self._target_device.use()
            self.stream.synchronize()

#        if self.checkInputTimes():
#         if self.target_device_idx>=0 and self.cuda_graph:
#             self.stream.synchronize()
#             self._target_device.synchronize()
#             self.xp.cuda.runtime.deviceSynchronize()
## at the end of the derevide method should call this?
#            default_target_device.use()
#            self.xp.cuda.runtime.deviceSynchronize()                
#            cp.cuda.Stream.null.synchronize()

    @classmethod
    def device_stream(cls, target_device_idx):
        if not target_device_idx in cls._streams:
            cls._streams[target_device_idx] = cp.cuda.Stream(non_blocking=False)
        return cls._streams[target_device_idx]
        
    def build_stream(self, allow_parallel=True):
        if self.target_device_idx>=0:
            self._target_device.use()
            if allow_parallel:
                self.stream = cp.cuda.Stream(non_blocking=False)
            else:
                self.stream = self.device_stream(self.target_device_idx)
            self.capture_stream()
            default_target_device.use()

    def capture_stream(self):
        with self.stream:
            # First execution is needed to build the FFT plan cache
            # See for example https://github.com/cupy/cupy/issues/7559
            self.trigger_code()
            self.stream.begin_capture()
            self.trigger_code()
            self.cuda_graph = self.stream.end_capture()

    def check_ready(self, t):
        self.current_time = t
        if self.checkInputTimes():
            if self.target_device_idx>=0:
                self._target_device.use()
            self.prepare_trigger(t)
            self.ready = True
        else:
            if self.verbose:
                print(f'No inputs have been refreshed, skipping trigger')
        return self.ready
    
    def trigger(self):        
        if self.ready:
            with show_in_profiler(self.__class__.__name__+'.trigger'):
                if self.target_device_idx>=0:
                    self._target_device.use()
                if self.target_device_idx>=0 and self.cuda_graph:
                    self.cuda_graph.launch(stream=self.stream)
                else:
                    self.trigger_code()
            self.ready = False
                    
    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def setup(self):
        """
        Override this method to perform any setup
        just before the simulation is started.

        The base class implementation also checks that
        all non-optional inputs have been set.
        
        """
        if self.target_device_idx >= 0:
            self._target_device.use()
        for name, input in self.inputs.items():
            if input.get(self.target_device_idx) is None and not input.optional:
                raise ValueError(f'Input {name} for object {self} has not been set')

    def finalize(self):
        '''
        Override this method to perform any actions after
        the simulation is completed
        '''
        pass

    def save(self, filename):
        with fits.open(filename, mode='update') as hdul:
            hdr = hdul[0].header
            hdr['VERBOSE'] = self._verbose
            hdul.flush()

    def read(self, filename):        
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            self._verbose = hdr.get('VERBOSE', 0)

