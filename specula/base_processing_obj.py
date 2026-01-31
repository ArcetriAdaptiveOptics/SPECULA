from collections import defaultdict

from specula import cpuArray, default_target_device, cp, MPI_DBG, MPI_SEND_DBG
from specula import show_in_profiler
from specula import process_comm, process_rank
from specula.base_time_obj import BaseTimeObj
from specula.data_objects.layer import Layer


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

        self.verbose = 0

        # Stream/input management
        self.stream  = None
        self.inputs_changed = False
        self.cuda_graph = None

        # Will be populated by derived class
        self.inputs = {}
        self.local_inputs = {}
        self.outputs = {}
        self.remote_outputs = defaultdict(list)
        self.sent_valid = {}

        # Use the correct CUDA device for allocations in derived classes'  __init__
        if self.target_device_idx >= 0:
            self._target_device.use()

        # Default name is none is given externally
        self.name = self.__class__.__name__

    # Use the correct CUDA device for allocations in derived classes' prepare_trigger()
    def prepare_trigger(self, t):
        self.current_time_seconds = self.t_to_seconds(self.current_time)
        if self.target_device_idx >= 0:
            self._target_device.use()

    def addRemoteOutput(self, name, remote_output):
        self.remote_outputs[name].append(remote_output)

    def checkInputTimes(self):
        if len(self.inputs)==0:
            return True
        self.get_all_inputs()
        for input_name, input_obj in self.local_inputs.items():
            if type(input_obj) is not list:
                input_obj = [input_obj]

            tt_list = [x.generation_time for x in input_obj if x is not None]
            for tt in tt_list:
                if tt is not None and tt >= self.current_time:
                    return True
        else:
            return False

    def get_all_inputs(self):
        '''
        Perform get() on all inputs.
        Remote inputs, if any, are received via MPI.
        Data is transferred between devices if necessary.
        '''
        for input_name, input_obj in self.inputs.items():
            if MPI_DBG: print(process_rank, 'get_all_inputs(): getting InputValue:',
                              input_name, flush=True)
            # Set additional info for better error messages
            input_obj.requesting_obj_name = self.name
            input_obj.input_name = input_name
            self.local_inputs[input_name] = input_obj.get(self.target_device_idx)

        if MPI_DBG:
            print(process_rank, self.name, 'My inputs are:')
            for in_name, in_value in self.local_inputs.items():
                if type(in_value) is list:
                    if len(in_value) > 0 and type(in_value[0]) is Layer:
                        print(process_rank, in_name,
                              [(x.generation_time, x.phaseInNm) for x in in_value],
                              flush=True)
                    else:
                        print(process_rank, in_name,
                              [(x.generation_time, x) for x in in_value],
                              flush=True)
                else:
                    print(process_rank, in_name,
                          in_value.generation_time if in_value is not None else None,
                          in_value, type(in_value), flush=True)

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
        '''
        Make sure we are using the correct device and that any previous
        CUDA graph has been synchronized
        '''
        # Double check that we can execute
        if not self.inputs_changed:
            raise RuntimeError("trigger() called when the object's inputs have not changed")

        # Reset inputs flag
        self.inputs_changed = False

        if self.target_device_idx>=0:
            self._target_device.use()
            if self.cuda_graph:
                self.stream.synchronize()


    def send_remote_output(self, item, dest_rank, dest_tag, first_mpi_send=True, out_name=''):
        if MPI_SEND_DBG: print(process_rank, f'SEND to rank {dest_rank} {dest_tag=} {(dest_tag in self.sent_valid)=} (from {self.name}.{out_name})', flush=True)
        if first_mpi_send or not dest_tag in self.sent_valid:
            if MPI_SEND_DBG: print(process_rank, 'SEND with Pickle', dest_tag, flush=True)
            xp_orig = item.xp
            item.xp = 0            
            process_comm.ibsend(item, dest=dest_rank, tag=dest_tag)
            item.xp = xp_orig
        else:            
            buffer = item.get_value()
            if MPI_SEND_DBG:  print(process_rank, dest_tag, 'SEND .device', buffer.device)
            if MPI_SEND_DBG: print(process_rank, 'SEND with Buffer', dest_tag, type(buffer), buffer, flush=True)
            if MPI_SEND_DBG: print(process_rank, 'SEND with Buffer type', dest_tag, buffer.dtype, flush=True)

            process_comm.Ibsend(cpuArray(buffer), dest=dest_rank, tag=dest_tag)

            process_comm.ibsend(item.generation_time, dest=dest_rank, tag=dest_tag+1)
        if item.get_value() is not None:
            self.sent_valid[dest_tag] = True


    # this method implements the mpi send call of the outputs connected to remote inputs
    def send_outputs(self, skip_delayed=False, delayed_only=False, first_mpi_send=True):
        '''
        Send all remote outputs via MPI.
        If *skip_delayed* is True, skip sending all delayed outputs.
            Used during the last iteration when the simulation is ending and
            no one would receive the delayed inputs.
        If *delayed_only* is True, only send the delayed outputs.
            Used while setting up the simulation, to initialize outputs
            that are delayed and thus would not be received otherwise.
        '''
        if MPI_DBG:
            print(process_rank, self.name, 'My outputs are:')
            for out_name, out_value in self.outputs.items():
                print(process_rank, out_name, out_value, flush=True)

        if MPI_DBG: print(process_rank, 'send_outputs', flush=True)
        for out_name, remote_specs in self.remote_outputs.items():
            for remote_spec in remote_specs:
                dest_rank, dest_tag, delay = remote_spec
                # avoid sending outputs that will not be received
                # because the simulation is ending
                if delay < 0 and skip_delayed:
                    if MPI_SEND_DBG: print(process_rank, f'SKIPPED SEND to rank {dest_rank} {dest_tag=} due to delay={delay}', flush=True)
                    continue
                if delay >= 0 and delayed_only:
                    if MPI_SEND_DBG: print(process_rank, f'SKIPPED SEND to rank {dest_rank} {dest_tag=} due to delay={delay}', flush=True)
                    continue
                if MPI_DBG: print(process_rank, 'Sending ', out_name, 'to ', dest_rank, 'with tag',  dest_tag, type(self.outputs[out_name]), flush=True)
                # workaround because module objects cannot be pickled
                for item in self.outputs[out_name] if isinstance(self.outputs[out_name], list) else [self.outputs[out_name]]:
                    self.send_remote_output(item, dest_rank, dest_tag, first_mpi_send, out_name)

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
        if self.target_device_idx >= 0:
            self._target_device.use()
        if self.checkInputTimes():
            self.inputs_changed = True  # Signal ready for trigger and post_trigger()
            self.prepare_trigger(t)
        else:
            self.inputs_changed = False
            if self.verbose:
                print('No inputs have been refreshed, skipping trigger')
        return self.inputs_changed

    def trigger(self):
        # Double check that we can execute
        if not self.inputs_changed:
            raise RuntimeError("trigger() called when the object's inputs have not changed")

        with show_in_profiler(self.__class__.__name__+'.trigger'):
            if self.target_device_idx>=0:
                self._target_device.use()
            if self.target_device_idx>=0 and self.cuda_graph:
                self.cuda_graph.launch(stream=self.stream)
            else:
                self.trigger_code()
             
    def setup(self):
        """
        Override this method to perform any setup
        just before the simulation is started.

        The base class implementation also checks that
        all non-optional inputs have been set.
        
        """
        if self.target_device_idx >= 0:
            self._target_device.use()

        self.get_all_inputs()
        for input_name, input in self.inputs.items():
            if self.local_inputs[input_name] is None and not input.optional:
                raise ValueError(f'Input {input_name} for object {self} has not been set')

    def finalize(self):
        '''
        Override this method to perform any actions after
        the simulation is completed
        '''
        pass



