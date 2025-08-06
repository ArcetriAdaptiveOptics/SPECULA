

import specula
specula.init(0)  # Default target device

import unittest

import yaml
from specula.simul import Simul
from specula.connections import InputValue, InputList

class DummyObj:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}

class DummyOutput:
    target_device_idx = -1

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
          class: 'Source'
          polar_coordinates: [1, 2]
          magnitude: null
          wavelengthInNm: null
        '''
        simul = Simul([])
        params = yaml.safe_load(yml)
        simul.build_objects(params)

        assert simul.objs['test'].magnitude is None
        assert simul.objs['test'].wavelengthInNm is None

    def test_scalar_input_reference(self):
        '''Test that an input is correctly connected'''
        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': 'a.out'
                }
            }
        })

        assert isinstance(simul.objs['b'].inputs['in'].get(-1), DummyOutputDerived)
        
    def test_list_input_reference(self):
        '''Test that a list of inputs is correctly connected'''
        simul = Simul([])
        simul.objs = {
            'a': DummyObj(),
            'b': DummyObj()
        }
        simul.objs['a'].outputs['out1'] = DummyOutputDerived()
        simul.objs['a'].outputs['out2'] = DummyOutputDerived()
        simul.objs['b'].inputs['in'] = InputList(type=DummyOutput)

        simul.connect_objects({
            'b': {
                'inputs': {
                    'in': ['a.out1', 'a.out2']
                }
            }
        })

        val = simul.objs['b'].inputs['in'].get(-1)
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
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

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
        simul.objs['b'].inputs['in'] = InputValue(type=DummyOutput)

        with self.assertRaises(ValueError):
            simul.connect_objects({
                'b': {'inputs': {'in': 'a.out'}}
            })


    def test_delayed_input(self):
        '''This test checks that the has_delayed_input method of
        Simul returns True if any object has a delayed input with
        the -1 syntax.
        '''
        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
            'obj3': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output'
                }
            }      
        }

        simul = Simul([])
        assert simul.has_delayed_output('obj1', pars) == True
        assert simul.has_delayed_output('obj2', pars) == False

    def test_delayed_input_detects_circular_loop(self):

        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
            'obj3': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output'
                }
            }      
        }
        simul = Simul([])

        # Does not raise
        _ = simul.trigger_order(pars)

        # These outputs depend on each other
        pars = {
            'obj1': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in1': 'obj2.output:-1'
                },
                'outputs': ['output']
            },
            'obj2': {
                'class': 'WaveGenerator',
                'inputs': {
                    'in2': 'obj1.output:-1'
                }
            },
        }
        # Raises ValueError
        with self.assertRaises(ValueError):
            _ = simul.trigger_order(pars)