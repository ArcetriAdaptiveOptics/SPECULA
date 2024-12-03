
import re
import typing
import inspect
import importlib
from copy import deepcopy
from specula.base_processing_obj import BaseProcessingObj

from specula.loop_control import LoopControl
from specula.lib.flatten import flatten
from specula.calib_manager import CalibManager
from specula.processing_objects.data_store import DataStore
from specula.connections import InputValue, InputList

import yaml
import io


class Simul():
    '''
    Simulation organizer
    '''
    def __init__(self, *param_files, overrides=None):
        if len(param_files) < 1:
            raise ValueError('At least one Yaml parameter file must be present')
        self.param_files = param_files
        self.objs = {}
        self.verbose = False  #TODO
        self.isReplay = False
        if overrides is None:
            self.overrides = []
        else:
            self.overrides = overrides

    def _camelcase_to_snakecase(self, s):
        tokens = re.findall('[A-Z]+[0-9a-z]*', s)
        return '_'.join([x.lower() for x in tokens])

    def _import_class(self, classname):
        modulename = self._camelcase_to_snakecase(classname)
        try:
            try:
                mod = importlib.import_module(f'specula.processing_objects.{modulename}')
            except ModuleNotFoundError:
                try:
                    mod = importlib.import_module(f'specula.data_objects.{modulename}')
                except ModuleNotFoundError:
                    mod = importlib.import_module(f'specula.display.{modulename}')
        except ModuleNotFoundError:
            raise ImportError(f'Class {classname} must be defined in a file called {modulename}.py but it cannot be found')

        try:
            return getattr(mod, classname)
        except AttributeError:
            raise AttributeError(f'Class {classname} not found in file {modulename}.py')

    def _get_type_hints(self, type):
        hints ={}
        for x in type.__mro__:
            hints.update(typing.get_type_hints(getattr(x, '__init__')))
        return hints
    
    def output_owner(self, output_name):
        if '-' in output_name:
            output_name = output_name.split('-')[1]
        if '.' in output_name:
            obj_name, attr_name = output_name.split('.')
            return obj_name
        else:
            return output_name

    def output_ref(self, output_name):
        if ':' in output_name:
            output_name = output_name.split(':')[0]
        if '.' in output_name:
            obj_name, attr_name = output_name.split('.')
            if not obj_name in self.objs:
                raise ValueError(f'Object {obj_name} does not exist')
            if not attr_name in self.objs[obj_name].outputs:
                raise ValueError(f'Object {obj_name} does not define an output with name {attr_name}')
            output_ref = self.objs[obj_name].outputs[attr_name]
        else:
            output_ref = self.objs[output_name]
        return output_ref

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
        ii = 0
        params = deepcopy(params_orig)
        del params['main']
        while True:
            start = len(params)
            leaves = [name for name, pars in params.items() if self.is_leaf(pars)]
            if len(leaves) == 0:
                break
            for leaf in leaves:
                order.append(leaf)
                order_index.append(ii)
                del params[leaf]
                self.remove_inputs(params, leaf)
            ii+=1
        if len(params) > 0:
            print('Warning: the following objects will not be triggered:', params.keys())
        return order, order_index


    def build_objects(self, params):
        main = params['main']
        cm = CalibManager(main['root_dir'])
        skip_pars = 'class inputs outputs'.split()

        for key, pars in params.items():
            if key == 'main':
                continue
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            klass = self._import_class(classname)
            args = inspect.getfullargspec(getattr(klass, '__init__')).args
            hints = self._get_type_hints(klass)

            target_device_idx = pars.get('target_device_idx', None)

            if 'tag' in pars:
                if len(pars) > 2:
                    raise ValueError('Extra parameters with "tag" are not allowed')
                filename = cm.filename(classname, pars['tag'])
                self.objs[key] = klass.restore(filename, target_device_idx=target_device_idx)
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
                    data = {x : self.output_ref(x) for x in value}
                    pars2[name[:-4]] = data

                # data fields are read from a fits file
                elif name.endswith('_data'):
                    data = cm.read_data(value)
                    pars2[name[:-5]] = data

                # object fields are data objects which are loaded from a fits file
                # the name of the object is the string preceeding th e_, 
                # while its type is infered from the constructor of the current class                
                elif name.endswith('_object'):
                    parname = name[:-7]
                    if parname in hints:
                        partype = hints[parname]
                        filename = cm.filename(parname, value)  # TODO use partype instead of parname?
                        parobj = partype.restore(filename, target_device_idx=target_device_idx)
                        pars2[parname] = parobj
                    else:
                        raise ValueError(f'No type hint for parameter {parname} of class {classname}')

                else:
                    pars2[name] = value

            # Add global and class-specific params if needed
            my_params = {k: main[k] for k in args if k in main}
            if 'data_dir' in args and 'data_dir' not in my_params:  # TODO special case
                my_params['data_dir'] = cm.root_subdir(classname)

            my_params.update(pars2)
            self.objs[key] = klass(**my_params)

            if type(self.objs[key]) is DataStore:
                self.objs[key].setParams(params)

    def connect_objects(self, params):
        for dest_object, pars in params.items():

            if 'outputs' in pars:
                for output_name in pars['outputs']:
                    if not output_name in self.objs[dest_object].outputs:
                        raise ValueError(f'Object {dest_object} does does not have an output called {output_name}')

            if 'inputs' not in pars:
                continue
            
            for input_name, output_name in pars['inputs'].items():

                if isinstance(output_name, list) and input_name=='input_list':
                    inputs = [x.split('-')[0] for x in output_name]
                    outputs = [self.output_ref(x.split('-')[1]) for x in output_name]
                    for ii, oo in zip(inputs, outputs):
                        self.objs[dest_object].inputs[ii] = InputValue(type = type(oo) )
                        self.objs[dest_object].inputs[ii].set(oo)
                        self.objs[dest_object].add(oo, name=ii)
                    continue

                if not input_name in self.objs[dest_object].inputs:
                    raise ValueError(f'Object {dest_object} does does not have an input called {input_name}')
                if not isinstance(output_name, (str, list)):
                    raise ValueError(f'Object {dest_object}: invalid input definition type {type(output_name)}')
                
                wanted_type = self.objs[dest_object].inputs[input_name].type()
                
                if isinstance(output_name, str):
                    output_ref = self.output_ref(output_name)
                    if not isinstance(output_ref, wanted_type):
                        raise ValueError(f'Input {input_name}: output {output_ref} is not of type {wanted_type}')

                elif isinstance(output_name, list):
                    outputs = [self.output_ref(x) for x in output_name]
                    output_ref = flatten(outputs)
                    for output in output_ref:
                        if not isinstance(output, wanted_type):
                            raise ValueError(f'Input {input_name}: output {output} is not of type {wanted_type}')

                self.objs[dest_object].inputs[input_name].set(output_ref)


    def build_replay(self, params):
        self.replay_params = deepcopy(params)
        skip_pars = 'class inputs outputs'.split()
        obj_to_remove = []
        data_source_outputs = {}
        for key, pars in params.items():
            if key == 'main':
                continue
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            if classname=='DataStore':
                self.replay_params['data_source'] = self.replay_params[key]
                self.replay_params['data_source']['class'] = 'DataSource'
                del self.replay_params['data_store']
                for output_name_full in pars['inputs']['input_list']:
                    input_name, output_name = output_name_full.split('-')
                    output_obj, output_name_small = output_name.split('.')                     
                    data_source_outputs[output_name] = 'data_source.' + input_name # 'source.' + output_obj + '-' + output_name_small                    
                    obj_to_remove.append(output_obj)

        for obj_name in set(obj_to_remove):
            del self.replay_params[obj_name]
        
        for key, pars in self.replay_params.items():            
            if key == 'main':
                continue
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

        for key, pars in params.items():
            if key == 'main':
                continue
            try:
                classname = pars['class']
            except KeyError:
                raise KeyError(f'Object {key} does not define the "class" parameter')

            if type(self.objs[key]) is DataStore:
                self.objs[key].setReplayParams(self.replay_params)


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

    def run(self):
        params = {}
        # Read YAML file(s)
        print('Reading parameters from', self.param_files[0])
        with open(self.param_files[0], 'r') as stream:
            params = yaml.safe_load(stream)
                
        for filename in self.param_files[1:]:
            print('Reading additional parameters from', self.param_files[0])
            with open(filename, 'r') as stream:
                additional_params = yaml.safe_load(stream)
                self.combine_params(params, additional_params)

        # Initialize housekeeping objects
        loop = LoopControl(run_time=params['main']['total_time'], dt=params['main']['time_step'])        

        # Actual creation code
        self.apply_overrides(params)
        self.build_objects(params)
        self.connect_objects(params)                
        
        if not self.isReplay:
            self.build_replay(params)

        trigger_order, trigger_order_idx = self.trigger_order(params)
        print(f'{trigger_order=}')
        print(f'{trigger_order_idx=}')

        # Build loop
        for name, idx in zip(trigger_order, trigger_order_idx):
            obj = self.objs[name]
            if isinstance(obj, BaseProcessingObj):
                loop.add(obj, idx)

        # Run simulation loop
        loop.run(run_time=params['main']['total_time'], dt=params['main']['time_step'], speed_report=True)

#        if data_store.has_key('sr'):
#            print(f"Mean Strehl Ratio (@{params['psf']['wavelengthInNm']}nm) : {store.mean('sr', init=min([50, 0.1 * params['main']['total_time'] / params['main']['time_step']])) * 100.}")

        for obj in self.objs.values():
            obj.finalize()
