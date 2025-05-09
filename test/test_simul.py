

import specula
specula.init(0)  # Default target device

import unittest

import yaml
from specula.simul import Simul

class DummyObj:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

class DummyInput:
    def __init__(self, type_):
        self._type = type_
        self.value = None

    def type(self):
        return self._type

    def set(self, value):
        self.value = value

class DummyOutput:
    pass

class DummyOutputDerived(DummyOutput):
    pass
  

class TestSimul(unittest.TestCase):

    def test_none_object_in_parameter_dict_is_none(self):
        '''
        Test that an "_object" directive in the YAML file
        with a "null" value results in a None value.
        
        We use one of our simplest objects setting
        a harmless parameter to " _object: null"
        '''
        yml = '''
        main:
          class: 'SimulParams'
          root_dir: dummy
          
        test:
          class: 'FuncGenerator'
          nmodes_object: null
        '''
        simul = Simul([])
        params = yaml.safe_load(yml)
        simul.build_objects(params)
        
        assert simul.objs['test'].nmodes is None

    def test_scalar_input_reference(self):
        '''Test that an input is correctly connected'''
        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = DummyInput(DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': 'a.out'
                }
            }
        })

        assert isinstance(simul.objs['b'].inputs['in'].value, DummyOutputDerived)
        
    def test_list_input_reference(self):
        '''Test that a list of inputs is correctly connected'''
        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out1'] = DummyOutputDerived()
        simul.objs['a'].outputs['out2'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = DummyInput(DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': ['a.out1', 'a.out2']
                }
            }
        })

        val = simul.objs['b'].inputs['in'].value
        assert isinstance(val, list)
        assert all(isinstance(x, DummyOutputDerived) for x in val)
        
    def test_missing_output_raises(self):
        simul = Simul([])
        simul.objs = {'a': DummyObj()}
        simul.objs['a'].outputs = {}

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'a': {'outputs': ['missing']}
            })
        
    def test_invalid_input_type(self):
        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = DummyInput(DummyOutput)

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'b': {
                    'inputs': {
                        'in': 42
                    }
                }
            })

    def test_type_mismatch(self):
        class WrongType:
            pass

        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out'] = WrongType()
        simul.objs['b'].inputs['in'] = DummyInput(DummyOutput)

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'b': {'inputs': {'in': 'a.out'}}
            })
