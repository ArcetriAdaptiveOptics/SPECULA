import logging
from collections import defaultdict, namedtuple
import fnmatch
import re

from specula import cpuArray, default_target_device, cp
from specula import show_in_profiler
from specula import process_comm
from specula.base_time_obj import BaseTimeObj
from specula.connections import InputList, InputValue
from specula.data_objects.layer import Layer


InputDesc = namedtuple('InputDesc', 'type desc')
OutputDesc = namedtuple('OutputDesc', 'type desc')

_PLACEHOLDER_PATTERN = re.compile(r'\{[^{}]+\}')


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

        # Stream/input management
        self.stream = None
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

        # Default name is the class name, a more specific one
        # can be given externally by the Simul class
        self.name = self.__class__.__name__

    # Use the correct CUDA device for allocations in derived classes' prepare_trigger()
    def prepare_trigger(self, t):
        self.current_time_seconds = self.t_to_seconds(self.current_time)
        if self.target_device_idx >= 0:
            self._target_device.use()

    def addRemoteOutput(self, name, remote_output):
        self.remote_outputs[name].append(remote_output)

    def checkInputTimes(self):
        '''
        Determine whether this processing object needs to execute
        the trigger method, based on the input states
        '''
        # No inputs: always trigger
        if len(self.inputs) == 0:
            return True

        self.get_all_inputs()

        # Inputs are all optional, and none of them is set: always trigger
        if all((self.inputs[k].optional is True and
                self.local_inputs[k] is None)
               for k in self.inputs):
            return True

        # Otherwise, only trigger if at least one input has been refreshed.
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
            self.logger.mpi_debug(f'- get_all_inputs(): '
                                  f'getting InputValue: {input_name}')
            # Set additional info for better error messages
            input_obj.requesting_obj_name = self.name
            input_obj.input_name = input_name
            self.local_inputs[input_name] = input_obj.get(self.target_device_idx)

        if self.logger.level <= logging.DEBUG:
            self.logger.mpi_debug(f'My inputs are:')
            for in_name, in_value in self.local_inputs.items():
                if type(in_value) is list:
                    if len(in_value) > 0 and type(in_value[0]) is Layer:
                        self.logger.mpi_debug(f'- {in_name}' + 
                                    str([(x.generation_time, x.phaseInNm) for x in in_value]))
                    else:
                        self.logger.mpi_debug(f'- {in_name}' + 
                                    str([(x.generation_time, x) for x in in_value]))
                else:
                    self.logger.mpi_debug(f'- {in_name}' + 
                            str(in_value.generation_time if in_value is not None else None) +
                            f'{in_value} type: {type(in_value)}')

    def trigger_code(self):
        '''
                Implementations in derived classes should run GPU operations using
                the xp module on arrays allocated with self.xp.

                Avoid explicit numpy or pure-Python operations and avoid using values
                from variables that are reallocated by prepare_trigger() or
                post_trigger().

                When stream capture is enabled, a CUDA graph is generated, non-GPU
                operations are skipped, and GPU memory addresses from the first run
                are reused.
        '''
        pass

    def post_trigger(self):
        '''
        Make sure we are using the correct device and that any previous
        CUDA graph has been synchronized
        '''
        # Double check that we can execute
        if not self.inputs_changed:
            raise RuntimeError("post_trigger() called when the object's inputs have not changed")

        # Reset inputs flag
        self.inputs_changed = False

        if self.target_device_idx>=0:
            self._target_device.use()
            if self.cuda_graph:
                self.stream.synchronize()

    def send_remote_output(self, item, dest_rank, dest_tag, first_mpi_send=True, out_name=''):
        self.logger.mpi_send_debug(f'SEND to rank {dest_rank} {dest_tag=} {(dest_tag in self.sent_valid)=} (from {self.name}.{out_name})')
        if first_mpi_send or not dest_tag in self.sent_valid:
            self.logger.mpi_send_debug(f'SEND with Pickle: {dest_tag=}')
            xp_orig = item.xp
            item.xp = 0            
            process_comm.ibsend(item, dest=dest_rank, tag=dest_tag)
            item.xp = xp_orig
        else:
            buffer = item.get_value()
            self.logger.mpi_send_debug(f'{dest_tag=}, SEND .device {buffer.device}')
            self.logger.mpi_send_debug(f'SEND with Buffe {dest_tag=}, {type(buffer)=}, {buffer=}')
            self.logger.mpi_send_debug(f'SEND with Buffer type {dest_tag=} {buffer.dtype=}')

            process_comm.Ibsend(cpuArray(buffer), dest=dest_rank, tag=dest_tag)

            process_comm.ibsend(item.generation_time, dest=dest_rank, tag=dest_tag+1)
        if item.get_value() is not None:
            self.sent_valid[dest_tag] = True

    # this method implements the mpi send call of the outputs connected to remote inputs
    def send_outputs(self, skip_delayed=False, delayed_only=False, first_mpi_send=True):
        '''
        Send all remote outputs via MPI.
        If *skip_delayed* is True, skip sending delayed outputs.
        This is used during the last iteration when the simulation is ending
        and no one would receive delayed inputs.

        If *delayed_only* is True, only send delayed outputs.
        This is used while setting up the simulation to initialize outputs
        that are delayed and would not be received otherwise.
        '''
        self.logger.mpi_debug(f'My outputs are:')
        for out_name, out_value in self.outputs.items():
            self.logger.mpi_debug(f'{out_name=}, {out_value=}')

        self.logger.mpi_debug(f'send_outputs')
        for out_name, remote_specs in self.remote_outputs.items():
            for remote_spec in remote_specs:
                dest_rank, dest_tag, delay = remote_spec
                # avoid sending outputs that will not be received
                # because the simulation is ending
                if delay < 0 and skip_delayed:
                    self.logger.mpi_send_debug(f'SKIPPED SEND to rank {dest_rank} {dest_tag=} due to delay={delay}')
                    continue
                if delay >= 0 and delayed_only:
                    self.logger.mpi_send_debug(f'SKIPPED SEND to rank {dest_rank} {dest_tag=} due to delay={delay}')
                    continue
                self.logger.mpi_debug(f'Sending {out_name} to {dest_rank} with tag {dest_tag} {type(self.outputs[out_name])}')
                # workaround because module objects cannot be pickled
                for item in self.outputs[out_name] if isinstance(self.outputs[out_name], list) else [self.outputs[out_name]]:
                    self.send_remote_output(item, dest_rank, dest_tag, first_mpi_send, out_name)

    @classmethod
    def device_stream(cls, target_device_idx):
        if not target_device_idx in cls._streams:
            cls._streams[target_device_idx] = cp.cuda.Stream(non_blocking=False)
        return cls._streams[target_device_idx]

    def build_stream(self, allow_parallel=True):
        if self.target_device_idx >= 0:
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
            self.logger.debug('No inputs have been refreshed, skipping trigger')
        return self.inputs_changed

    def trigger(self):
        # Double check that we can execute
        if not self.inputs_changed:
            raise RuntimeError("trigger() called when the object's inputs have not changed")

        with show_in_profiler(self.__class__.__name__+'.trigger'):
            if self.target_device_idx >= 0:
                self._target_device.use()
            if self.target_device_idx >= 0 and self.cuda_graph:
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

    def sanity_check(self):
        '''
        Check that all inputs and outputs have been setup correctly.
        '''
        self.check_input_names()
        self.check_output_names()

    @staticmethod
    def _normalize_declared_pattern(pattern):
        """Normalize declared I/O names to a fnmatch-compatible pattern.

        Declaration grammar accepted by sanity checks:
        - Exact names: ``out_comm``
                - Placeholder-like names: ``out_modes_{sensor_idx}``
                    (used in ``ModalrecMultirate.output_names``)
                - Placeholder-like names: ``out_{source_name_}ef``
                    (used in ``AtmoRandomPhase.output_names``)

                Placeholder segments delimited by ``{...}`` are treated as wildcards and
                internally converted to ``*``. Matching is delegated to
        :func:`fnmatch.fnmatchcase`.
        """
        normalized = _PLACEHOLDER_PATTERN.sub('*', str(pattern))
        return normalized

    @classmethod
    def _match_declared_name(cls, actual_name, declared_name):
        normalized = cls._normalize_declared_pattern(declared_name)
        return fnmatch.fnmatchcase(actual_name, normalized)

    @classmethod
    def _best_declared_match(cls, actual_name, declared_map):
        # Prefer exact match when available, then fallback to first pattern match.
        if actual_name in declared_map:
            return actual_name, declared_map[actual_name]

        for declared_name, declared_desc in declared_map.items():
            if cls._match_declared_name(actual_name, declared_name):
                return declared_name, declared_desc

        return None, None

    def check_input_names(self):
        '''
        Check that all input names declared in self.input_names are present in self.inputs
                with the correct type (InputValue or InputList).

                Supported declaration grammar for input names:
                - Exact names, e.g. ``in_ef``
                - Placeholder patterns, e.g. ``in_sensor_{idx}``

                Notes
                -----
                - Placeholder segments ``{...}`` are treated as wildcards.
                                - In project classes, placeholder-style declarations are mandatory;
                                    do not use raw ``*`` declarations in ``input_names()``.
                - Optional declarations are identified by descriptions ending with
                    ``(optional)``.
                - Exact-key matches take precedence over pattern matches.
        '''
        if not hasattr(self, 'input_names'):
            return
        input_dict = self.input_names()
        for declared_name, declared_desc in input_dict.items():
            optional_decl = declared_desc[1].endswith("(optional)")
            declared_matches = [
                name for name in self.inputs
                if self._match_declared_name(name, declared_name)
            ]
            if not optional_decl and not declared_matches:
                raise ValueError(
                    f"Input {declared_name} declared in input_names but not present in inputs"
                )

        for input_name, input_obj in self.inputs.items():
            declared_name, declared_desc = self._best_declared_match(input_name, input_dict)
            if declared_name is None:
                raise ValueError(f"Input {input_name} present in inputs but not declared in input_names")

            if not isinstance(input_obj, (InputValue, InputList)):
                raise TypeError(f"Input {input_name} must be an InputValue or an InputList")

            expected_type = declared_desc[0]
            if input_obj.output_ref_type is not expected_type:
                raise TypeError(
                    f"Input {input_name} must be of type {expected_type}, "
                    f"but got {input_obj.type}"
                )

    def check_output_names(self):
        '''
        Check that all output names declared in self.output_names are present in self.outputs

        Supported declaration grammar for output names:
        - Exact names, e.g. ``out_ef``
                - Placeholder patterns, e.g. ``out_modes_{sensor_idx}``
                    (``ModalrecMultirate``)
                - Placeholder patterns, e.g. ``out_{source_name_}layer``
                    (``AtmoRandomPhase``)

        Notes
        -----
        - Placeholder segments ``{...}`` are treated as wildcards.
                                - In project classes, placeholder-style declarations are mandatory;
                                    do not use raw ``*`` declarations in ``output_names()``.
        - Exact-key matches take precedence over pattern matches.
        '''
        if not hasattr(self, 'output_names'):
            return
        output_list = self.output_names()
        for declared_name, declared_desc in output_list.items():
            declared_matches = [
                name for name in self.outputs
                if self._match_declared_name(name, declared_name)
            ]
            if not declared_matches:
                raise ValueError(
                    f"Output {declared_name} declared in output_names but not present in outputs"
                )

            expected_type = declared_desc[0]
            for output_name in declared_matches:
                if not isinstance(self.outputs[output_name], expected_type):
                    raise TypeError(
                        f"Output {output_name} must be of type {expected_type}, "
                        f"but got {type(self.outputs[output_name])}"
                    )

        for output_name in self.outputs:
            declared_name, _ = self._best_declared_match(output_name, output_list)
            if declared_name is None:
                raise ValueError(
                    f"Output {output_name} present in outputs but not declared in output_names"
                )
