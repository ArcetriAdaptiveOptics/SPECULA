import sys
import typing
import inspect
import itertools
from copy import deepcopy
from pathlib import Path
from collections import Counter
from specula import process_comm, process_rank, MPI_DBG
from specula.base_processing_obj import BaseProcessingObj
from specula.base_data_obj import BaseDataObj

from specula.loop_control import LoopControl
from specula.lib.flatten import flatten
from specula.lib.utils import import_class, get_type_hints
from specula.calib_manager import CalibManager
from specula.processing_objects.data_store import DataStore
from specula.connections import InputValue, InputList

import yaml
import hashlib
 
def computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name, index=None):
    s = output_obj_name + '%' + dest_object + '%' + str(output_attr_name) + '%' + str(input_attr_name) + '%'  + str(index)

    rr = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**6   
    return rr, s

class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self,
                 *param_files,
                 overrides=None,
                 diagram=False,
                 diagram_title=None,
                 diagram_filename=None
                 ):
        if len(param_files) < 1:
            raise ValueError('At least one Yaml parameter file must be present')
        self.all_objs_ranks = {}
        self.remote_objs_ranks = {}
        self.remote_objs_types = {}
        self.remote_objs_params = {}
        self.param_files = param_files
        self.objs = {}
        self.verbose = False  #TODO
        self.isReplay = False
        self.mainParams = None
        self.mainParamsKeyName = None
        if overrides is None:
            self.overrides = []
        else:
            self.overrides = overrides
        self.diagram = diagram
        self.diagram_title = diagram_title
        self.diagram_filename = diagram_filename

    def output_owner(self, output_name):
        if ':' in output_name:
            output_name = output_name.split(':')[0]
        if '-' in output_name:
            output_name = output_name.split('-')[1]
        try:
            obj_name, _ = output_name.split('.')
        except ValueError:
            raise ValueError(f'Invalid output name {output_name}, must be in the form "object_name.output_name"')
        return obj_name

    def output_key(self, output_name):
        if ':' in output_name:
            output_name = output_name.split(':')[0]
        if '-' in output_name:
            output_name = output_name.split('-')[1]
        try:
            _, output_key = output_name.split('.')
        except ValueError:
            raise ValueError(f'Invalid output name {output_name}, must be in the form "object_name.output_name"')
        return output_key
        
    def output_ref(self, output_name):
        '''
        return a tuple with:
           - reference to the output, or None if the object is remote.
           - name of the object that defines the output
        '''
        if ':' in output_name:
            output_name = output_name.split(':')[0]
        if '-' in output_name:
            output_name = output_name.split('-')[1]
        try:
            obj_name, output_name = output_name.split('.')
        except ValueError:
            raise ValueError(f'Invalid output name {output_name}, must be in the form "object_name.output_name"')
            
        if not obj_name in self.objs:
            if obj_name in self.remote_objs_ranks:
                return None, obj_name
            else:
                raise ValueError(f'Object {obj_name} does not exist anywhere')
        if not output_name in self.objs[obj_name].outputs:
            raise ValueError(f'Object {obj_name} does not define an output with name {output_name}')
        output_ref = self.objs[obj_name].outputs[output_name]

        return output_ref, obj_name

    def input_ref(self, input_name):
        '''
        return a tuple with:
           - reference to the input, or None if the object is remote.
           - name of the object that defines the input
        '''
        if ':' in input_name:
            input_name = input_name.split(':')[0]
        if '-' in input_name:
            input_name = input_name.split('-')[1]
        try:
            obj_name, attr_name = input_name.split('.')
        except ValueError:
            raise ValueError(f'Invalid output name {input_name}, must be in the form "object_name.attr_name"')

        if not obj_name in self.objs:
            if obj_name in self.remote_objs_ranks:
                return None, obj_name
            else:                
                raise ValueError(f'Object {obj_name} does not exist anywhere')
        if not attr_name in self.objs[obj_name].inputs:
            raise ValueError(f'Object {obj_name} does not define an input with name {attr_name}')
        input_ref = self.objs[obj_name].local_inputs[attr_name]
        return input_ref, obj_name

    def output_delay(self, output_name):
        if ':' in output_name:
            return int(output_name.split(':')[1])
        else:
            return 0

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
                    raise ValueError('Malformed output: must be either str or list')

                for x in outputs_list:
                    owner = self.output_owner(x)
                    delay = self.output_delay(x)
                    if owner == obj_name and delay < 0:
                        # Delayed input detected
                        return True
        return False

    def trigger_order(self, params_orig):
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
                self.remove_inputs(params, leaf)
            end = len(params)
            if start == end:
                raise ValueError('Cannot determine trigger order: circular loop detected in {leaves}')
        if len(params) > 0:
            print('Warning: the following objects will not be triggered:', params.keys())
        return order, order_index

    def setSimulParams(self, params):
        for key, pars in params.items():
            classname = pars['class']
            if classname == 'SimulParams':
                self.mainParams = pars
                self.mainParamsKeyName = key

    def build_objects(self, params):

        self.setSimulParams(params)

        cm = CalibManager(self.mainParams['root_dir'])
        skip_pars = 'class inputs outputs'.split()

        if MPI_DBG: print(process_rank, 'building objects')

        for key, pars in params.items():
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            klass = import_class(classname)
            args = inspect.getfullargspec(getattr(klass, '__init__')).args
            hints = get_type_hints(klass)

            target_device_idx = pars.get('target_device_idx', None)
                        
            target_rank = pars.get('target_rank', None)
            if target_rank is None:
                target_rank = 0
                self.all_objs_ranks[key] = 0
            else:                          
                self.all_objs_ranks[key] = target_rank
                del pars['target_rank']        

            if 'tag' in pars:
                if len(pars) > 2:
                    raise ValueError('Extra parameters with "tag" are not allowed')
                filename = cm.filename(classname, pars['tag'])
                # tags are restored into each process (multiple copies), target_rank is not checked
                print('Restoring:', filename)
                self.objs[key] = klass.restore(filename, target_device_idx=target_device_idx)
                self.objs[key].stopMemUsageCount()
                self.objs[key].printMemUsage()

                continue

            pars2 = {}
            for name, value in pars.items():
                if key == 'data_source':
                    self.isReplay = True

                if key != 'data_source' and name in skip_pars:
                    continue

                if key == 'data_source' and name in ['class']:
                    continue

                # dict_ref field contains a dictionary of names and associated data objects (defined in the same yml file)
                elif name.endswith('_dict_ref'):
                    data = {x : self.objs[x] for x in value}
                    pars2[name[:-4]] = data

                elif name.endswith('_ref'):
                    data = self.objs[value]
                    pars2[name[:-4]] = data

                # data fields are read from a fits file
                elif name.endswith('_data'):
                    data = cm.read_data(value)
                    pars2[name[:-5]] = data

                # object fields are data objects which are loaded from a fits file
                # the name of the object is the string preceeding the "_object" suffix,
                # while its type is inferred from the constructor of the current class
                elif name.endswith('_object'):
                    parname = name[:-7]
                    if value is None:
                        pars2[parname] = None
                    elif parname in hints:
                        partype = hints[parname]

                        # Handle Optional and Union types (for python <3.11)
                        if hasattr(partype, "__origin__") and partype.__origin__ is typing.Union:
                            # Extract actual class type from Optional/Union
                            # (first non-None type argument)
                            for arg in partype.__args__:
                                if arg is not type(None):  # Skip NoneType
                                    partype = arg
                                    break
                        # data objects are restored into each process (multiple copies), target_rank is not checked
                        filename = cm.filename(parname, value)  # TODO use partype instead of parname?
                        print('Restoring:', filename)
                        parobj = partype.restore(filename, target_device_idx=target_device_idx)
                        parobj.stopMemUsageCount()
                        parobj.printMemUsage()

                        pars2[parname] = parobj
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                else:
                    pars2[name] = value

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
            # create the simulations objects for this process. Data Object and SimulParams are always
            # created, no matter what their rank (assigned process) is.
            if process_rank==target_rank or issubclass(klass, BaseDataObj) or classname=='SimulParams':
                try:
                    self.objs[key] = klass(**my_params)
                except Exception:
                    print(f'Exception building', key)
                    raise
                if classname != 'SimulParams':
                    self.objs[key].stopMemUsageCount()

                self.objs[key].name = key

                # TODO this could be more general like the getters above
                if type(self.objs[key]) is DataStore:
                    self.objs[key].setParams(params)
            else:
                self.remote_objs_ranks[key] = target_rank
                self.remote_objs_types[key] = klass
                self.remote_objs_params[key] = my_params

    def connect_objects(self, params):
        self.connections = []
        
        def _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object,
                     index=None, set_list=False):
    
            if output_ref is None:
                if local_dest_object:
                    print(process_rank, f'{output_ref} -> {dest_object} : remote to local connection', flush=True)
                    # receiving input from a remote object
                    if index is not None:
                        self.objs[dest_object].inputs[input_name].set_item_remote_rank(self.remote_objs_ranks[output_obj_name], index=index)
                    else:
                        self.objs[dest_object].inputs[input_name].set_remote_rank(self.remote_objs_ranks[output_obj_name])
                    tag, s = computeTag(output_obj_name, dest_object, output_attr_name, input_name, index)
                    if MPI_DBG: print(process_rank, 'Input side, Computed tag (B):', tag, s, flush=True)
                    if index is not None:
                        self.objs[dest_object].inputs[input_name].set_item_tag(tag, index=index)
                    else:
                        self.objs[dest_object].inputs[input_name].set_tag(tag)

                else:
                #   nothing to do, both the sender and the receiver are remote, 
                #   some other processe will take care of this case
                    print(process_rank, f'{output_ref} -> {dest_object} : remote to remote connection, nothing to do', flush=True)
            else:
                # local connection
                if local_dest_object:
                    print(process_rank, f'{output_ref} -> {dest_object} : local to local connection, simple case', flush=True)
                    if index is not None:
                        print(process_rank, f'{dest_object=} setting input {input_name} to {output_ref} with index {index}', flush=True)
                        self.objs[dest_object].inputs[input_name].set_item(output_ref, index=index)
                    else:
                        if set_list:
                            self.objs[dest_object].inputs[input_name].set_list(output_ref)
                        else:
                            self.objs[dest_object].inputs[input_name].set(output_ref)
                else:
                    # sending output to a remote object
                    print(process_rank, f'{output_ref} -> {dest_object} : local to remote connection, calling addRemoteOutput()', flush=True)
                    if dest_object in self.remote_objs_ranks and output_obj_name in self.objs:
                        print(process_rank, 'Adding remote output to ', output_obj_name, flush=True)
                        tag, s = computeTag(output_obj_name, dest_object, output_attr_name, input_name, index=index)
                        if MPI_DBG: print(process_rank, 'Output side, Computed tag (B):', tag, s, flush=True)
                        self.objs[output_obj_name].addRemoteOutput(output_attr_name, (self.remote_objs_ranks[dest_object], tag))

        for dest_object, pars in params.items():

            if MPI_DBG: print(process_rank, 'connect_objects for', dest_object, flush=True)

            classname = pars['class']
            local_dest_object = dest_object in self.objs.keys()

            if 'outputs' in pars:
                for output_name in pars['outputs']:
                    if local_dest_object:
                        # check that this output was actually created by this dest_object
                        if not output_name in self.objs[dest_object].outputs:
                            raise ValueError(f'Object {dest_object} does not have an output called {output_name}')
                    else:
                        # remote object case
                        if not ( self.all_objs_ranks[dest_object]!=process_rank \
                             and 'outputs' in params[dest_object] \
                             and output_name in params[dest_object]['outputs'] ):
                            raise ValueError(f'Remote Object {dest_object} does not have an output called {output_name}')

            if 'inputs' not in pars:
                continue

            for input_name, output_name in pars['inputs'].items():

                if MPI_DBG: print(process_rank, 'ASSIGNMENT of input_name:', input_name, flush=True)
                if MPI_DBG: print(process_rank, 'output_name', output_name, flush=True)

                # Special case for DataStore
                if isinstance(output_name, list) and input_name=='input_list':
                    inputs = [x.split('-')[0] for x in output_name]
                    output_names = [x.split('-')[1].split('.')[1] for x in output_name]
                    outputs = [self.output_ref(x.split('-')[1])[0] for x in output_name]
                    outputs_obj_names = [self.output_ref(x.split('-')[1])[1] for x in output_name]                    
                    # if MPI_DBG: print(process_rank, 'output_names:', output_names, flush=True)
                    # if MPI_DBG: print(process_rank, 'inputs:', inputs, flush=True)
                    # if MPI_DBG: print(process_rank, 'outputs:', outputs, flush=True)
                    # if MPI_DBG: print(process_rank, 'outputs_obj_names:', outputs_obj_names, flush=True)
                    for input_attr_name, oo, output_attr_name, output_obj_name in zip(inputs, outputs, output_names, outputs_obj_names):
                        a_connection = {}                            
                        if oo is None:
                            if local_dest_object:
                                # remote input case
                                a_connection['remote'] = True
                                self.objs[dest_object].inputs[input_attr_name] = InputValue(type = self.remote_objs_types[output_obj_name])
                                self.objs[dest_object].inputs[input_attr_name].set_remote_rank(self.remote_objs_ranks[output_obj_name])
                                tag, s = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)
                                if MPI_DBG: print(process_rank, 'Input side, Computed tag (A):', s, tag, flush=True)
                                self.objs[dest_object].inputs[input_attr_name].set_tag(tag)
                            # else: nothing to do, both the sender and the reciver are remote, some other process will take care of this case
                        else:
                            if local_dest_object:
                                a_connection['remote'] = False
                                self.objs[dest_object].inputs[input_attr_name] = InputValue(type = type(oo))
                                self.objs[dest_object].inputs[input_attr_name].set(oo)
                            else:
                                # the sender is local, but the receiver is not
                                if MPI_DBG: print('Adding remote output to ', output_attr_name, flush=True)
                                tag, s = computeTag(output_obj_name, dest_object, output_attr_name, input_attr_name)
                                if MPI_DBG: print(process_rank, 'Output side, Computed tag (A):', s, tag, flush=True)
                                self.objs[output_obj_name].addRemoteOutput(output_attr_name, (self.remote_objs_ranks[dest_object], tag))

                        a_connection['start'] = output_attr_name.split('.')[0].split('-')[-1]
                        a_connection['end'] = dest_object
                        a_connection['start_label'] = input_attr_name
                        # a_connection['middle_label'] = self.objs[dest_object].inputs[input_attr_name]
                        a_connection['end_label'] = output_attr_name
                        self.connections.append(a_connection)
                        print(a_connection)
                    continue

                if local_dest_object:
                    if not input_name in self.objs[dest_object].inputs:
                        raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')
                    if not isinstance(output_name, (str, list)):
                        raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')
                    
                    wanted_type = self.objs[dest_object].inputs[input_name].type()
                else:
                    # TODO cannot check the type for remote object
                    wanted_type = None

                print(process_rank, 'known objects: ', self.objs.keys())
                if local_dest_object:
                    local_list = type(self.objs[dest_object].inputs[input_name]) == InputList
                else:
                    local_list = False

                if isinstance(output_name, str):
                    if MPI_DBG: print(process_rank, 'Simple input', flush=True)

                    # Here we add the input, we can create the local or remote connections
                    output_ref= self.output_ref(output_name)[0]
                    output_obj_name = self.output_ref(output_name)[1]
                    output_attr_name = self.output_key(output_name)

                    print(process_rank, 'output_obj_name', output_obj_name, 'output_attr_name', output_attr_name,flush=True)                    
                    print(process_rank, f'output_ref: {output_ref}', flush=True)
    
                    if not local_dest_object and output_ref is None:
                        continue
                    
                    output_ref_is_list = type(output_ref) is list
                    
                    if local_list:
                        if output_ref_is_list:
                            # List - List (example: atmo.layer_list to propagation.atmo_layer_list)
                            _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object, set_list=True)
                        else:
                            # Single remote object into a local list (not tested)
                            _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object, index=0)

                    else:
                        if output_ref_is_list:
                            # TODO this works but I do not understand why (test: set atmo to rank 1 and prop to rank 0)
                            _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object, index=0)
                        else:
                            _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object)

                elif isinstance(output_name, list):
                    if MPI_DBG: print(process_rank, 'List input', flush=True)

                    output_refs = [self.output_ref(x)[0] for x in output_name]
                    output_obj_names = [self.output_ref(x)[1] for x in output_name]
                    output_attr_names = [self.output_key(x) for x in output_name]
                    
                    print(output_name, flush=True)
                    for x in output_name:
                        print(process_rank, f'output_ref: {x}',  flush=True)

                    if local_dest_object and not local_list:
                        raise ValueError(f'Cannot set a list of inputs to a non-list input {dest_object}.{input_name}')
                    
                    for i, (output_ref, output_obj_name, output_attr_name) in \
                        enumerate(zip(output_refs, output_obj_names, output_attr_names)):

                        if not local_dest_object and output_ref is None:
                            continue
                        
                        _connect(output_ref, output_obj_name, output_attr_name, local_dest_object, input_name, dest_object, index=i)

                else:
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')
                    

                try:
                    pass
                    #if output_ref is not None and local_dest_object:
                    #    if MPI_DBG: print(process_rank, "setting input", dest_object, input_name, output_ref, flush=True)
                    #    self.objs[dest_object].inputs[input_name].set(output_ref)
                    # TODO Note this! is it necessary or useful?
                    #else:
                    #    if dest_object in self.remote_objs_ranks and output_obj_name in self.objs:
                    #        print(process_comm, '2 Adding remote output to ', output_obj_name)
                    #        tag, s = computeTag(output_obj_name, dest_object, output_attr_name, input_name)
                    #        if MPI_DBG: print(process_rank, 'Computed tag (G):', tag, s, flush=True)
                    #        self.objs[output_obj_name].remote_outputs[output_attr_name] = (self.remote_objs_ranks[dest_object], tag)
                    # self.objs[output_obj_name].remote_outputs[output_attr_name] = (self.remote_objs_ranks[dest_object], tag)            
                    #    # self.objs[dest_object].inputs[input_name].set(None)
                except ValueError:
                    print(f'Error connecting {output_name} to {dest_object}.{input_name}')
                    raise
                # else:
                #     # do nothing??? TODO
                                        
            if local_dest_object:             
                if not type(output_name) is list:
                    a_connection = {}
                    a_connection['start'] = output_name.split('.')[0].split('-')[-1]
                    a_connection['end'] = dest_object
                    a_connection['start_label'] = output_name.split('.')[-1]
                    a_connection['middle_label'] = self.objs[dest_object].inputs[input_name]
                    a_connection['end_label'] = self.objs[dest_object].inputs[input_name]

                    self.connections.append(a_connection)
                else:
                    for oo in output_name:
                        a_connection = {}
                        a_connection['start'] = oo.split('.')[0].split('-')[-1]
                        a_connection['end'] = dest_object
                        a_connection['start_label'] = oo.split('.')[-1]
                        # a_connection['middle_label'] = self.objs[dest_object].inputs[input_name]
                        # a_connection['end_label'] = self.objs[dest_object].inputs[input_name]
                        if output_ref is not None:
                            a_connection['remote'] = False
                        else:
                            a_connection['remote'] = True
                        print(a_connection)
                        self.connections.append(a_connection)

    def build_replay(self, params):
        self.replay_params = deepcopy(params)
        obj_to_remove = []
        data_source_outputs = {}
        for key, pars in params.items():
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            if classname=='DataStore':
                self.replay_params['data_source'] = self.replay_params[key]
                self.replay_params['data_source']['class'] = 'DataSource'
                del self.replay_params[key]
                for output_name_full in pars['inputs']['input_list']:
                    input_name, output_name = output_name_full.split('-')
                    output_obj, output_name_small = output_name.split('.')                     
                    data_source_outputs[output_name] = 'data_source.' + input_name # 'source.' + output_obj + '-' + output_name_small                    
                    obj_to_remove.append(output_obj)

        for obj_name in set(obj_to_remove):
            del self.replay_params[obj_name]

        for key, pars in self.replay_params.items():
            if not key=='data_source':
                if 'inputs' in pars.keys():
                    for input_name, output_name_full in pars['inputs'].items():
                        if type(output_name_full) is list:
                            print('TODO: list of inputs is not handled in output replay')
                            continue
                        if output_name_full in data_source_outputs.keys():
                            self.replay_params[key]['inputs'][input_name] = data_source_outputs[output_name_full]

            if key=='data_source':
                self.replay_params[key]['outputs'] = []
                for v in self.replay_params[key]['inputs']['input_list']:
                    kk, vv = v.split('-')
                    self.replay_params[key]['outputs'].append(kk)
                del self.replay_params[key]['inputs']

        for obj in self.objs.values():
            if type(obj) is DataStore:
                obj.setReplayParams(self.replay_params)

    def remove_inputs(self, params, obj_to_remove):
        '''
        Modify params removing all references to the specificed object name
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
                            if self.verbose:
                                print(f'Deleted {input_name} from {obj[key]}')
                    elif isinstance(output_name, list):
                        newlist = [x for x in output_name if self.output_owner(x) != obj_to_remove]
                        diff = set(output_name).difference(set(newlist))
                        obj_inputs_copy[input_name] = newlist
                        if len(diff) > 0:
                            if self.verbose:
                                print(f'Deleted {diff} from {obj[key]}')
                obj[key] = obj_inputs_copy
        return params

    def combine_params(self, params, additional_params):
        '''
        Add/update/remove params with additional_params
        '''
        for name, values in additional_params.items():
            if name == 'remove':
                for objname in values:
                    if objname not in params:
                        raise ValueError(f'Parameter file has no object named {objname}')
                    del params[objname]
                    print(f'Removed {objname}')

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
        print('overrides:', self.overrides)
        if len(self.overrides) > 0:
            for k, v in yaml.full_load(self.overrides).items():
                obj_name, param_name = k.split('.')
                params[obj_name][param_name] = v
                print(obj_name, param_name, v)


    def arrangeInGrid(self, trigger_order, trigger_order_idx):
        rows = []
        n_cols = max(trigger_order_idx) + 1                
        n_rows = max( list(dict(Counter(trigger_order_idx)).values()))        
        # names_to_orders = dict(zip(trigger_order, trigger_order_idx))
        orders_to_namelists = {}
        for order in range(n_cols):
            orders_to_namelists[order] = []
        for name, order in zip(trigger_order, trigger_order_idx):
            orders_to_namelists[order].append(name)

        for ri in range(n_rows):
            r = []
            for ci in range(n_cols):
                col_elements = len(orders_to_namelists[ci])
                if ri<col_elements:
                    block_name = orders_to_namelists[ci][ri]
                else:
                    block_name = ""                
                r.append(block_name)
            rows.append(r)
        return rows

    def buildDiagram(self):
        from orthogram import Color, DiagramDef, write_png, Side, FontWeight, TextOrientation

        print('Building diagram...')

        d = DiagramDef(label=self.diagram_title, text_fill=Color(0, 0, 1), scale=2.0, collapse_connections=True)
        rows = self.arrangeInGrid(self.trigger_order, self.trigger_order_idx)
        # a row is a list of strings, which are labels for the cells
        for r in rows:
            d.add_row(r)        
        for c in self.connections:
            aconn = d.add_connection(c['start'], c['end'], buffer_fill=Color(1.0,1.0,1.0), buffer_width=1, 
                             exits=[Side.RIGHT], entrances=[Side.LEFT, Side.BOTTOM, Side.TOP])
            aconn.set_start_label(c['middle_label'],font_weight=FontWeight.BOLD, text_fill=Color(0, 0.5, 0), text_orientation=TextOrientation.HORIZONTAL)
        write_png(d, self.diagram_filename)
        print('Diagram saved.')

    def run(self):
        params = {}
        # Read YAML file(s)
        print('Reading parameters from', self.param_files[0])
        with open(self.param_files[0], 'r') as stream:
            params = yaml.safe_load(stream)

        for filename in self.param_files[1:]:
            print('Reading additional parameters from', filename)
            with open(filename, 'r') as stream:
                additional_params = yaml.safe_load(stream)
                self.combine_params(params, additional_params)

        # Actual creation code
        self.apply_overrides(params)

        self.trigger_order, self.trigger_order_idx = self.trigger_order(params)
        print(f'{self.trigger_order=}')
        print(f'{self.trigger_order_idx=}')

        self.build_objects(params)
        self.connect_objects(params)

        # Initialize housekeeping objects
        self.loop = LoopControl()

        if not self.isReplay:
            self.build_replay(params)

        if self.diagram or self.diagram_filename or self.diagram_title:
            if self.diagram_filename is None:
                self.diagram_filename = str(Path(self.param_files[0]).with_suffix('.png'))
            if self.diagram_title is None:
                self.diagram_title = str(Path(self.param_files[0]).with_suffix(''))
            self.buildDiagram()

        # Build loop
        for name, idx in zip(self.trigger_order, self.trigger_order_idx):
            if name not in self.remote_objs_ranks:
                obj = self.objs[name]
                if isinstance(obj, BaseProcessingObj):
                    self.loop.add(obj, idx)
        
        self.loop.max_global_order = max(self.trigger_order_idx)
        print('self.loop.max_global_order', self.loop.max_global_order, flush=True)


        # Default display web server
        if 'display_server' in self.mainParams and self.mainParams['display_server']:
            from specula.processing_objects.display_server import DisplayServer
            disp = DisplayServer(params, self.input_ref, self.output_ref, self.get_info)
            self.objs['display_server'] = disp
            self.loop.add(disp, idx+1)
            disp.name = 'display_server'

        if MPI_DBG: print(process_rank, 'at run barrier')
        
        sys.stdout.flush()
        if process_comm is not None:
            process_comm.barrier()

        # Run simulation loop
        self.loop.run(run_time=self.mainParams['total_time'], dt=self.mainParams['time_step'], speed_report=True)

#        if data_store.has_key('sr'):
#            print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * self.mainParams['total_time'] / self.mainParams['time_step']])) * 100.}")

    def get_info(self):
        '''Quick info string intended for web interfaces'''
        name= f'{self.param_files[0]}'
        curtime= f'{self.loop._t / self.loop._time_resolution:.3f}'
        stoptime= f'{self.loop._run_time / self.loop._time_resolution:.3f}'

        info = f'{curtime}/{stoptime}s'
        return name, info
