from __future__ import annotations

import re
import inspect
import itertools
from copy import deepcopy
from pathlib import Path
from specula import process_rank
from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj

from specula.log import get_specula_logger
from specula.loop_control import LoopControl
from specula.lib.utils import import_class, get_type_hints, remove_suffix, resolve_type
from specula.calib_manager import CalibManager
from specula.processing_objects.data_store import DataStore
from specula.connections import InputList, InputValue, split_output
from specula.simul_diagram import SimulDiagram

import yaml
import hashlib


def computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name):
    s = output_obj_name + '%' + dest_object + '%' + str(output_attr_name) + '%' + str(input_attr_name)
    rr = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**6
    return rr


class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self,
                 *param_files,
                 simul_idx=0,
                 overrides: str | None = None,
                 stepping=False,
                 diagram=False,
                 diagram_title=None,
                 diagram_filename=None,
                 diagram_colors_on=False,
                 speed_report=True,
                 log_level: str='info',
                 ):
        if len(param_files) < 1:
            raise ValueError('At least one Yaml parameter file must be present')

        self.is_dataobj = {}
        self.all_objs_ranks = {}
        self.all_target_device_idxs = {}
        self.remote_objs_ranks = {}
        self.param_files = param_files
        self.objs = {}
        self.simul_idx = simul_idx
        self.mainParams = None
        if overrides is None:
            self.overrides = ""
        else:
            self.overrides = overrides
        self.stepping = stepping
        self.speed_report = speed_report
        self.logger = get_specula_logger(__name__)
        self.logger.setLevel(log_level.upper())
        if diagram or diagram_title or diagram_filename or diagram_colors_on:
            if diagram_filename is None:
                diagram_filename = str(Path(self.param_files[0]).with_suffix('.png'))
            if diagram_title is None:
                diagram_title = str(Path(self.param_files[0]).with_suffix(''))

            self.diagram = SimulDiagram(param_file=self.param_files[0],
                                        title=diagram_title,
                                        filename=diagram_filename,
                                        colors_on=diagram_colors_on)
        else:
            self.diagram = None

    def output_owner(self, output_name):
        return split_output(output_name).obj_name

    def split_output(self, output_name, get_ref=False, use_inputs=False):
        output = split_output(output_name)
        if get_ref:
            obj_name, output_key, *_ = output

            if not obj_name in self.objs:
                if obj_name in self.remote_objs_ranks:
                    ref = None
                else:
                    raise ValueError(f'Object {obj_name} does not exist anywhere')
            elif output_key is None:
                ref = self.objs[obj_name]
            else:
                if use_inputs:
                    array_to_check, display_str = self.objs[obj_name].local_inputs, 'input'
                else:
                    array_to_check, display_str = self.objs[obj_name].outputs, 'output'
                if not output_key in array_to_check:
                    raise ValueError(f'Object {obj_name} does not define an {display_str} with name {output_key}')
                else:
                    ref = array_to_check[output_key]
            output = output._replace(ref = ref)
        return output

    def output_ref(self, output_name):
        return self.split_output(output_name, get_ref=True).ref

    def input_ref(self, input_name):
        return self.split_output(input_name, get_ref=True, use_inputs=True).ref

    def output_delay(self, output_name):
        return split_output(output_name).delay

    def is_leaf(self, p):
        '''
        Returns True if the passed object parameter dictionary
        does not specify any inputs for the current iterations.
        Inputs coming from previous iterations (:-1 syntax) are ignored.
        '''
        if 'inputs' not in p:
            return True

        for input_name, output_name in p['inputs'].items():
            if isinstance(output_name, str):
                maxdelay = self.output_delay(output_name)
            elif isinstance(output_name, list):
                maxdelay = -1
                if len(output_name) > 0:
                    maxdelay = max([self.output_delay(x) for x in output_name])
            if maxdelay == 0:
                return False
        return True

    def has_delayed_output(self, obj_name, params):
        '''
        Find out if an object has an output
        that is used as a delayed input for another
        object in the pars dictionary
        '''
        for name, pars in params.items():
            if 'inputs' not in pars:
                continue
            for input_name, output_name in pars['inputs'].items():
                if isinstance(output_name, str):
                    outputs_list = [output_name]
                elif isinstance(output_name, list):
                    outputs_list = output_name
                else:
                    raise ValueError('Malformed output: must be either str or list: '+str(output_name))

                for x in outputs_list:
                    owner = self.output_owner(x)
                    delay = self.output_delay(x)
                    if owner == obj_name and delay < 0:
                        # Delayed input detected
                        return True
        return False

    def build_trigger_order(self, params_orig):
        '''
        Work on a copy of the parameter file.
        1. Find leaves, add them to trigger
        2. Remove leaves, remove their inputs from other objects
          2a. Objects will become a leaf when all their inputs have been removed
        3. Repeat from step 1. until there is no change
        4. Check if any objects have been skipped
        '''
        order = []
        order_index = []
        params = deepcopy(params_orig)
        for index in itertools.count():
            leaves = [name for name, pars in params.items() if self.is_leaf(pars)]
            if len(leaves) == 0:
                break
            start = len(params)
            for leaf in leaves:
                if self.has_delayed_output(leaf, params):
                    continue
                order.append(leaf)
                order_index.append(index)
                del params[leaf]
                self.remove_inputs(params, leaf, log=False)
            end = len(params)
            if start == end:
                raise ValueError('Cannot determine trigger order: circular loop detected in {leaves}')
        if len(params) > 0:
            self.logger.warning(f'the following objects will not be triggered: {params.keys()}')
        return order, order_index

    def validate_section_names(self, params):
        '''
        Reject section (object) names that use characters reserved by the
        output-reference syntax parsed in split_output(): '.', '-' and ':'.

        These characters separate an optional alias, the object name, the
        output key and an optional delay/type suffix (e.g. "alias-obj.output:1").
        An object name containing one of them makes output references
        ambiguous: e.g. an object named "my-control" cannot be distinguished
        from an alias "my" applied to an object named "control".
        '''
        reserved = {
            '.': "separates 'obj_name' from 'output_key'",
            '-': "separates an optional alias from 'obj_name.output_key'",
            ':': "introduces an optional delay/type suffix",
        }
        for key in params.keys():
            for ch, purpose in reserved.items():
                if ch in key:
                    raise ValueError(f"Invalid section name '{key}' in parameter file: "
                                      f"character '{ch}' is reserved ({purpose}) and cannot "
                                      "be used in object names")

    def setSimulParams(self, params):
        for key, pars in params.items():
            classname = pars['class']
            if classname == 'SimulParams':
                self.mainParams = pars

    def build_order(self, params):
        '''
        Return the correct object build order, taking into account
        dependencies specified by _ref and _dict_ref parameters
        '''
        build_order = []

        def add_to_build_order(key):
            if key in build_order:
                return

            pars = params[key]
            for name, value in pars.items():
                if name.endswith('_ref'):
                    objlist = value if type(value) is list else [value]
                    for output in objlist:
                        owner = self.output_owner(output)
                        if owner not in build_order:
                            add_to_build_order(owner)

            build_order.append(key)

        for key in params.keys():
            add_to_build_order(key)

        return build_order

    def create_input_list_inputs(self, params):
        '''
        Create inputs for objects that use input_list parameter.
        Currently supported: DataStore, DataBuffer
        '''
        supported_classes = ['DataBuffer','DataStore']

        for key, pars in params.items():
            if ('class' in pars and 
                pars['class'] in supported_classes and
                'inputs' in pars and 
                'input_list' in pars['inputs']):

                for single_output_name in pars['inputs']['input_list']:
                    output = self.split_output(single_output_name, get_ref=True)
                    if key in self.objs:
                        if type(output.ref) is list:
                            self.objs[key].inputs[output.input_name] = InputList(type=type(output.ref[0]))
                        else:
                            self.objs[key].inputs[output.input_name] = InputValue(type=type(output.ref))
                    params[key]['inputs'][output.input_name] = single_output_name
                del params[key]['inputs']['input_list']

            if pars['class'] == 'DataBuffer':
                self.objs[key].setOutputs()

    def build_objects(self, params):

        self.setSimulParams(params)

        cm = CalibManager(self.mainParams['root_dir'])
        skip_pars = 'class inputs outputs gui_pos'.split()
        if 'add_modules' in self.mainParams:
            additional_modules = self.mainParams['add_modules']
        else:
            additional_modules = []

        self.logger.mpi_debug(f'building objects')

        for key in self.build_order(params):

            pars = params[key]
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            klass = import_class(classname, additional_modules)
            args = inspect.getfullargspec(getattr(klass, '__init__')).args
            hints = get_type_hints(klass)
            target_device_idx = pars.get('target_device_idx', None)
            self.all_target_device_idxs[key] = target_device_idx
 
            par_target_rank = pars.pop('target_rank', None)
            if par_target_rank is None:
                target_rank = 0
            else:
                target_rank = par_target_rank
            self.all_objs_ranks[key] = target_rank

            # create the simulations objects for this process. Data Objects are created
            # on all ranks (processes) by default, unless a specific rank has been specified.
            self.is_dataobj[key] = issubclass(klass, BaseDataObj)

            build_this_object = (process_rank == target_rank) or \
                                (issubclass(klass, BaseDataObj) and (par_target_rank == None)) or \
                                (issubclass(klass, BaseDataObj) and (par_target_rank == process_rank)) or \
                                (process_rank == None)

            # If not build, remember the remote rank of this object (needed for connections setup)
            if not build_this_object:
                self.remote_objs_ranks[key] = target_rank

            if 'tag' in pars and build_this_object:
                if 'target_device_idx' in pars:
                    del pars['target_device_idx']
                if len(pars) > 2:
                    raise ValueError('Extra parameters with "tag" are not allowed')
                filename = cm.filename(classname, pars['tag'])
                # tags are restored into each process (multiple copies), target_rank is not checked
                self.logger.info(f'Restoring: {filename}')
                self.objs[key] = klass.restore(filename, target_device_idx=target_device_idx)
                self.objs[key].name = key
                self.objs[key].init_logging(self.logger.getEffectiveLevel())
                self.objs[key].printMemUsage()
                self.objs[key].tag = pars['tag']
                continue

            pars2 = {}
            for name, value in pars.items():

                # Skip special parameters, unless explictly present in __init__
                # e.g. "outputs" in DataSource
                if name in skip_pars and name not in args:
                    continue

                # Check that each parameter name is expected by the constructor of the class, after removing possible suffixes
                parname = name
                if parname not in args:
                    for ending in ['_ref', '_data', '_object']:
                        candidate = remove_suffix(parname, ending)
                        if candidate in args:
                            parname = candidate
                            break
                if parname not in args:
                    raise ValueError(f'Parameter {parname} is not expected by class {classname}')

                # list_ref field contains an ordered list of associated data objects
                # (defined in the same yml file).
                elif name.endswith('_list_ref') and parname != name:
                    if not isinstance(value, (list, tuple)):
                        raise ValueError(f'Parameter {name} must be a list of object names')
                    if build_this_object:
                        pars2[parname] = [self.objs[x] for x in value]
                    if self.diagram:
                        for x in value:
                            self.diagram.add_reference(start=key, end=x)

                # dict_ref field contains a dictionary of names and associated data objects (defined in the same yml file)
                elif name.endswith('_dict_ref') and parname != name:
                    if build_this_object:
                        data = {x : self.objs[x] for x in value}
                        pars2[parname] = data
                    if self.diagram:
                        for x in value:
                            self.diagram.add_reference(start=key, end=x)

                # list_object fields contain an ordered list of tags to be restored as data objects.
                elif name.endswith('_list_object') and parname != name and build_this_object:
                    if value is None:
                        pars2[parname] = None
                    elif not isinstance(value, (list, tuple)):
                        raise ValueError(f'Parameter {name} must be a list of tags')
                    elif parname in hints:
                        try:
                            partype = resolve_type(hints[parname], require_list=True)
                        except TypeError:
                            raise ValueError(f'Parameter {parname} must be typed as List[DataObjType]')

                        loaded = []
                        for tag in value:
                            filename = cm.filename(partype.__name__, tag)
                            self.logger.info(f'Restoring: {filename}')
                            obj = partype.restore(filename, target_device_idx=target_device_idx)
                            obj.printMemUsage()
                            obj.tag = tag
                            loaded.append(obj)

                        pars2[parname] = loaded
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                # dict_object fields contain a dictionary of tags to be restored as data objects.
                elif name.endswith('_dict_object') and parname != name and build_this_object:
                    if value is None:
                        pars2[parname] = None
                    elif not isinstance(value, dict):
                        raise ValueError(f'Parameter {name} must be a dictionary of tags')
                    elif parname in hints:
                        try:
                            partype = resolve_type(hints[parname], require_dict=True)
                        except TypeError:
                            raise ValueError(f'Parameter {parname} must be typed as Dict[str, DataObjType]')

                        loaded = {}
                        for dict_key, tag in value.items():
                            filename = cm.filename(partype.__name__, tag)
                            self.logger.info(f'Restoring: {filename}')
                            obj = partype.restore(filename, target_device_idx=target_device_idx)
                            obj.printMemUsage()
                            obj.tag = tag
                            loaded[dict_key] = obj

                        pars2[parname] = loaded
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')
                elif name.endswith('_ref') and parname != name:
                    if build_this_object:
                        data = self.objs[value]
                        pars2[parname] = data
                    if self.diagram:
                        self.diagram.add_reference(start=key, end=value)

                # data fields are read from a fits file
                elif name.endswith('_data') and parname != name and build_this_object:
                    if value is None:
                        pars2[parname] = None
                    else:
                        data = cm.read_data(value)
                        pars2[parname] = data

                # object fields are data objects which are loaded from a fits file
                # the name of the object is the string preceeding the "_object" suffix,
                # while its type is inferred from the constructor of the current class
                elif name.endswith('_object') and parname != name and build_this_object:
                    if value is None:
                        pars2[parname] = None
                    elif parname in hints:
                        partype = resolve_type(hints[parname])

                        # data objects are restored into each process (multiple copies), target_rank is not checked
                        filename = cm.filename(parname, value)  # TODO use partype instead of parname?
                        self.logger.info(f'Restoring: {filename}')
                        parobj = partype.restore(filename, target_device_idx=target_device_idx)
                        parobj.init_logging(self.logger.getEffectiveLevel())
                        parobj.printMemUsage()

                        # Set data_tag
                        parobj.tag = value
                        pars2[parname] = parobj
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                else:
                    if build_this_object:
                        pars2[name] = value

            if not build_this_object:
                continue

            # Add global and class-specific params if needed
            my_params = {}

            if 'data_dir' in args and 'data_dir' not in my_params:  # TODO special case
                my_params['data_dir'] = cm.root_subdir(classname)

            if 'params_dict' in args:
                my_params['params_dict'] = params

            if 'input_ref_getter' in args:
                my_params['input_ref_getter'] = self.input_ref

            if 'output_ref_getter' in args:
                my_params['output_ref_getter'] = self.output_ref

            if 'info_getter' in args:
                my_params['info_getter'] = self.get_info

            my_params.update(pars2)
            try:
                self.objs[key] = klass(**my_params)
                self.objs[key].name = key
                self.objs[key].init_logging(self.logger.getEffectiveLevel())
            except Exception:
                self.logger.error(f'Exception building {key}')
                raise
            if classname != 'SimulParams':
                self.objs[key].stopMemUsageCount()

            # TODO this could be more general like the getters above
            if type(self.objs[key]) is DataStore:
                self.objs[key].setParams(params)

    def connect(self, output_name, input_name, dest_object):
        '''
        Connect the output *output_name*, defined by object *output_obj_name*,
        and whose reference is *output_ref*, which might be None if the object is remote,
        to the input *input_name* of the object *dest_object*, which might be local or remote.

        This routine handles the three cases:
        1. local output to local input - use Python references
        2. local output to remote input - use addRemoteOutput() to send the output to the remote object
        3. remote output to local input - use set_remote_rank() to set the remote rank of the input
        '''
        output = self.split_output(output_name, get_ref=True)
        local_dest_object = dest_object in self.objs.keys()

        send = output.ref is not None and local_dest_object is False
        recv = output.ref is None and local_dest_object is True
        local = output.ref is not None and local_dest_object is True
        if send or recv:
            tag = computeTag(output.obj_name, dest_object, output.output_key, input_name)

        self.logger.mpi_debug(f'{output.obj_name}.{output.output_key} -> {dest_object} : {send=} {recv=} {local=}')

        if recv:
            self.logger.mpi_debug(f'CONNECT Connecting remote output {output.obj_name}.{output.output_key} to local input {dest_object}.{input_name} with tag {tag}')
            self.objs[dest_object].inputs[input_name].append(None,
                                                            remote_rank = self.remote_objs_ranks[output.obj_name],
                                                            tag=tag)
        if local:
            self.logger.mpi_debug(f'CONNECT Connecting local output {output.obj_name}.{output.output_key} to local input {dest_object}.{input_name}')
            self.objs[dest_object].inputs[input_name].append(output.ref)

        if send:
            self.objs[output.obj_name].addRemoteOutput(output.output_key, (self.remote_objs_ranks[dest_object], 
                                                                            tag,
                                                                            output.delay))
                
    def connect_objects(self, params):
        
        for dest_object, pars in params.items():

            self.logger.mpi_debug(f'connect_objects for {dest_object}')

            local_dest_object = dest_object in self.objs.keys()

            # Check that outputs exist (or for remote objects, that they are defined in the params)
            if 'outputs' in pars:
                for output_name in pars['outputs']:
                    if local_dest_object:
                        # check that this output was actually created by this dest_object
                        if not output_name in self.objs[dest_object].outputs:
                            raise ValueError(f'Object {dest_object} does not have an output called {output_name}')
                    else:
                        # remote object case
                        # TODO these checks are almost all redundant
                        if not ( self.all_objs_ranks[dest_object] != process_rank \
                             and 'outputs' in params[dest_object] \
                             and output_name in params[dest_object]['outputs'] ):
                            raise ValueError(f'Remote Object {dest_object} does not have an output called {output_name}')

            if 'inputs' not in pars:
                continue

            for input_name, output_name in pars['inputs'].items():

                self.logger.mpi_debug(f'ASSIGNMENT of input_name: {input_name}')
                self.logger.mpi_debug(f'{output_name=}')

                if local_dest_object and input_name != 'input_list':
                    if not input_name in self.objs[dest_object].inputs:
                        raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')

                if not isinstance(output_name, (str, list)):
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')

                for single_output_name in output_name if isinstance(output_name, list) else [output_name]:
                    self.logger.mpi_debug(f'List input')

                    output = self.split_output(single_output_name, get_ref=True)

                    if self.diagram:
                        self.diagram.add_connection(start = output.obj_name,
                                                    end= dest_object,
                                                    start_label= output.output_key,
                                                    end_label = input_name)

                    # Remote-to-remote: nothing to do
                    if not local_dest_object and output.ref is None:
                        continue
                    
                    try:
                        self.connect(single_output_name, input_name, dest_object)
                    except ValueError:
                        self.logger.error(f'Exception while connecting {single_output_name} {dest_object}.{input_name}')
                        raise


    def isReplay(self, params):
        return 'data_source' in params

    def data_store_to_data_source(self, datastore_pars, set_store_dir=None):
        '''
        Convert data store parameters to data source.

        Returns a tuple (pars, refs), where:
        - pars is a parameter dictionary for a DataSource object
        - objnames is a list of objects referenced by original DataStore inpus
        '''
        data_source_pars = {}
        data_source_outputs = {}
        data_source_pars['class'] = 'DataSource'
        data_source_pars['outputs'] = []
        if 'data_format' in datastore_pars:
            data_source_pars['data_format'] = datastore_pars['data_format']
        if set_store_dir:
            data_source_pars['store_dir'] = set_store_dir
        else:
            data_source_pars['store_dir'] = datastore_pars['store_dir']

        objnames = []
        for _, fullname in self.iterate_inputs(datastore_pars):
            output = self.split_output(fullname)
            data_source_pars['outputs'].append(output.input_name)
            data_source_outputs[output.obj_name+'.'+output.output_key] = output.input_name
            objnames.append(output.obj_name)

        return data_source_pars, objnames, data_source_outputs

    def build_replay(self, params):
        replay_params = deepcopy(params)
        obj_to_remove = []
        data_source_objname =''
        data_source_outputs = {}
        for key, pars in params.items():
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            if classname=='DataStore':
                data_source_pars, obj_to_remove, data_source_outputs = self.data_store_to_data_source(pars)
                replay_params['data_source'] = data_source_pars
                data_source_objname = key
                obj_to_remove.append(data_source_objname)

        for obj_name in set(obj_to_remove):
            del replay_params[obj_name]

        for key, pars in replay_params.items():
            if not key=='data_source':
                if 'inputs' in pars.keys():
                    for input_name, output_name_full in pars['inputs'].items():
                        if type(output_name_full) is list:
                            self.logger.warning('TODO: list of inputs is not handled in output replay')
                            continue
                        self.logger.debug(f'{output_name_full=}')
                        if output_name_full in data_source_outputs.keys():
                            replay_params[key]['inputs'][input_name] = 'data_source.' + data_source_outputs[output_name_full]

        return replay_params

    def inject_recorded_seeds(self, target_params, recorded_seeds):
        '''
        Mutate target_params in place: for every (key, seed) in recorded_seeds,
        if key is present in target_params and it does not already declare an
        explicit 'seed', inject the recorded value. This lets a replay reproduce
        the exact values of a RandomGenerator (or similar) that had no explicit
        seed in the original run, without changing the behavior of fresh runs
        (which never have recorded_seeds to inject).
        '''
        for key, seed in recorded_seeds.items():
            if key in target_params and 'seed' not in target_params[key]:
                target_params[key]['seed'] = seed

    def _produces_ef_or_layer(self, class_name):
        '''
        True if the given class declares an ElectricField or Layer output.
        Used to detect objects whose silent omission from a replay would drop
        a phase-additive contribution (e.g. an ElectricFieldCombinator or
        PhaseScreenCube feeding a WFS downstream of a captured AtmoPropagation).
        '''
        from specula.data_objects.electric_field import ElectricField
        from specula.data_objects.layer import Layer
        try:
            outputs = import_class(class_name).output_names()
        except Exception:
            return False
        return any(issubclass(desc.type, (ElectricField, Layer)) for desc in outputs.values())

    def _find_dropped_ef_layer_consumers(self, params, replay_params):
        '''
        Objects present in the original params but not captured into replay_params,
        that (a) consume an output of an object that IS captured, and (b) themselves
        declare an ElectricField or Layer output -- i.e. objects whose omission
        silently drops a phase-additive contribution from the replayed graph
        (SPECULA #696: a PhaseScreenCube summed via ElectricFieldCombinator onto an
        AtmoPropagation source's output, downstream of the propagation object, is
        invisible to a replay targeted at that propagation object).
        '''
        dropped = {}
        for key, pars in params.items():
            if key in replay_params or not self._produces_ef_or_layer(pars.get('class')):
                continue
            hits = [(input_name, out) for input_name, out in self.iterate_inputs(pars)
                    if self.output_owner(out) in replay_params]
            if hits:
                dropped[key] = hits
        return dropped

    def build_targeted_replay(self, params, *target_object_names, set_store_dir=None,
                               on_missing_downstream_consumers='error'):
        '''
        Build a replay file making sure that the target objects
        still exist, and therefore all their inputs are either loaded
        from disk or computed, recursively.

        SimulParams parameters are replicated unchanged.
        DataStore parameters are converted to DataSource

        on_missing_downstream_consumers : str
            What to do when an object present in params, that consumes a captured
            object's output and itself declares an ElectricField or Layer output,
            was not captured into the replay (see _find_dropped_ef_layer_consumers):
            'error' (default) raises ValueError, 'warn' logs a warning, 'ignore'
            does nothing.
        '''
        # Create new parameter dict and copy SimulParams without changes
        replay_params = {}
        datastore_outputs = {}

        for key, pars in params.items():
            if pars['class'] == 'SimulParams':
                main_pars = pars
                break
        else:
            raise ValueError('Parameter file does not contain a SimulParams class')

        replay_params[key] = main_pars.copy()

        # Copy DataStore params and convert it to DataSource
        for key, pars in params.items():
            if pars['class'] == 'DataStore':
                data_source_pars, _, datastore_mapping = self.data_store_to_data_source(pars, set_store_dir=set_store_dir)
                replay_params['data_source'] = data_source_pars

                # Merge all datastore outputs using the complete key (obj_name.output_key)
                datastore_outputs.update(datastore_mapping)

        def add_key(key):
            if key in replay_params:
                return

            replay_params[key] = params[key].copy()

            # Add all inputs
            for k, _input in self.iterate_inputs(params[key]):
                desc = self.split_output(_input)
                # Use the complete key for lookup
                complete_key = f"{desc.obj_name}.{desc.output_key}"
                if complete_key in datastore_outputs:
                    replay_params[key]['inputs'][k] = 'data_source.' + datastore_outputs[complete_key]
                    continue
                else:
                    add_key(desc.obj_name)
            # Add all references to other objects
            for k, v in params[key].items():
                if k.endswith('_list_ref'):
                    for objname in v:
                        add_key(objname)
                elif k.endswith('_dict_ref'):
                    for objname in v:
                        add_key(objname)
                elif k.endswith('_ref'):
                    add_key(v)

        for key in target_object_names:
            add_key(key)

        dropped = self._find_dropped_ef_layer_consumers(params, replay_params)
        if dropped and on_missing_downstream_consumers != 'ignore':
            message = (
                "build_targeted_replay: the following objects produce an ElectricField/Layer "
                "output and consume a captured object's output, but were not themselves captured "
                "-- their contribution is silently missing from the replayed graph: "
                + '; '.join(f"'{k}' (input '{inp}' -> '{out}')"
                            for k, hits in dropped.items() for inp, out in hits)
                + ". Pass on_missing_downstream_consumers='warn' or 'ignore' if this is expected "
                  "(e.g. a diagnostics-only branch), otherwise add the missing object(s) to "
                  "target_object_names."
            )
            if on_missing_downstream_consumers == 'error':
                raise ValueError(message)
            self.logger.warning(message)

        return replay_params

    def iterate_inputs(self, pars):
        '''
        Iterate over all inputs of a parameter dictionary.
        Yields a series of (key, value) tuples suitable
        for dictionary-like iteration.
        '''
        if 'inputs' not in pars:
            return
        inputs = pars['inputs']
        if 'input_list' in inputs:
            for x in inputs['input_list']:
                yield ('input_list', x)
        else:
            for k, v in inputs.items():
                if type(v) is list:
                    for xx in v:
                        yield (k, xx)
                else:
                    yield (k, v)

    def remove_inputs(self, params, obj_to_remove, log=True):
        '''
        Modify params removing all references to the specified object name
        '''
        for objname, obj in params.items():
            for key in ['inputs']:
                if key not in obj:
                    continue
                obj_inputs_copy = deepcopy(obj[key])
                for input_name, output_name in obj[key].items():
                    if isinstance(output_name, str):
                        owner = self.output_owner(output_name)
                        if owner == obj_to_remove:
                            del obj_inputs_copy[input_name]
                            if log:
                                self.logger.info(f'Deleted {input_name} from {obj[key]}')
                    elif isinstance(output_name, list):
                        newlist = [x for x in output_name if self.output_owner(x) != obj_to_remove]
                        diff = set(output_name).difference(set(newlist))
                        obj_inputs_copy[input_name] = newlist
                        if len(diff) > 0 and log:
                            self.logger.info(f'Deleted {diff} from {obj[key]}')
                obj[key] = obj_inputs_copy
        return params

    def combine_params(self, params, additional_params):
        '''
        Add/update/remove params with additional_params
        '''
        for name, values in additional_params.items():
            # Check if "name" ends with _ followed by a number, in that case 
            # the number is a simulation index and we skip these parameters
            # if our simul_idx is not equal to the number.
            # e.g. dm_override_2: { ... } or remove_3: ['atmo', 'rec', 'dm2']
            match = re.search(r'^(.*)_(\d+)$', name)
            if match:
                idx = int(match.group(2))
                if idx != self.simul_idx:
                    continue
                else:
                    name = match.group(1)

            if name == 'remove':
                for objname in values:
                    if objname not in params:
                        raise ValueError(f'Parameter file has no object named {objname}')
                    del params[objname]
                    self.logger.info(f'Removed {objname}')
                    # Remove corresponding inputs
                    params = self.remove_inputs(params, objname)
            elif name.endswith('_override'):
                objname = name[:-9]
                if objname not in params:
                    raise ValueError(f'Parameter file has no object named {objname}')
                params[objname].update(values)
            else:
                if name in params:
                    raise ValueError(f'Parameter file already has an object named {name}')
                params[name] = values

    def apply_overrides(self, params):
        self.logger.info('overrides: ' + str(self.overrides))
        if len(self.overrides) > 0:
            for k, v in yaml.full_load(self.overrides).items():
                parts = k.split('.')
                if len(parts) == 2:
                    params[parts[0]][parts[1]] = v
                    self.logger.debug(f'{parts} {v}')
                elif len(parts) == 3:
                    params[parts[0]][parts[1]][parts[2]] = v
                    self.logger.debug(f'{parts} {v}')
                else:
                    raise ValueError(f"Invalid number of parts detected in override: {parts}. Did you add/forget a '.'?")
    
    def run(self, start_time=0, end_time=None):
        params = {}
        # Read YAML file(s)
        self.logger.info('Reading parameters from ' + self.param_files[0])
        with open(self.param_files[0], 'r') as stream:
            params = yaml.safe_load(stream)

        for filename in self.param_files[1:]:
            self.logger.info('Reading additional parameters from ' + filename)
            with open(filename, 'r') as stream:
                additional_params = yaml.safe_load(stream)                
                self.combine_params(params, additional_params)

        # Actual creation code
        self.apply_overrides(params)

        self.validate_section_names(params)

        self.trigger_order, self.trigger_order_idx = self.build_trigger_order(params)
        self.logger.info(f'{self.trigger_order=}')
        self.logger.info(f'{self.trigger_order_idx=}')

        if not self.isReplay(params):
            replay_params = self.build_replay(params)
        else:
            recorded_seeds = (params.get('data_source') or {}).get('random_seeds') or {}
            self.inject_recorded_seeds(params, recorded_seeds)
            replay_params = None

        self.build_objects(params)
        self.create_input_list_inputs(params)
        self.connect_objects(params)

        if (process_rank == 0 or process_rank is None) and self.diagram:
            self.diagram.build(trigger_order = self.trigger_order,
                               trigger_order_idx = self.trigger_order_idx,
                               all_target_device_idxs=self.all_target_device_idxs,
                               all_objs_ranks = self.all_objs_ranks,
                               is_dataobj = self.is_dataobj)

        if replay_params is not None:
            recorded_seeds = {
                key: obj.get_resolved_seed()
                for key, obj in self.objs.items()
                if isinstance(obj, BaseProcessingObj) and obj.get_resolved_seed() is not None
            }
            if recorded_seeds:
                replay_params.setdefault('data_source', {})['random_seeds'] = recorded_seeds
            for obj in self.objs.values():
                if type(obj) is DataStore:
                    obj.setReplayParams(replay_params)

        # Initialize housekeeping objects
        self.loop = LoopControl(stepping=self.stepping)

        # Build loop
        for name, idx in zip(self.trigger_order, self.trigger_order_idx):
            if name not in self.remote_objs_ranks:
                obj = self.objs[name]
                if isinstance(obj, BaseProcessingObj):
                    self.loop.add(obj, idx)
        
        self.loop.max_global_order = max(self.trigger_order_idx)
        self.logger.debug(f'{self.loop.max_global_order=}')

        # Default display web server
        if 'display_server' in self.mainParams and self.mainParams['display_server'] and process_rank in [0, None]:
            from specula.processing_objects.display_server import DisplayServer
            disp = DisplayServer(params, self.input_ref, self.output_ref, self.get_info)
            disp.name = 'display_server'
            self.objs['display_server'] = disp
            self.loop.add(disp, idx+1)

        # Run simulation loop
        total_time = self.mainParams['total_time']
        run_time = (end_time if end_time is not None else total_time) - start_time
        self.loop.run(run_time=run_time,
                      dt=self.mainParams['time_step'],
                      t0=start_time,
                      speed_report=self.speed_report)

        self.logger.debug(f'Simulation finished')
#        if data_store.has_key('sr'):
#            self.logger.info(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * self.mainParams['total_time'] / self.mainParams['time_step']])) * 100.}")

    def get_info(self):
        '''Quick info string intended for web interfaces'''
        name= f'{self.param_files[0]}'
        curtime= f'{self.loop.t / self.loop._time_resolution:.3f}'
        stoptime= f'{self.loop.run_time / self.loop._time_resolution:.3f}'

        info = f'{curtime}/{stoptime}s'
        return name, info
