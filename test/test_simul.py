

import specula
specula.init(0)  # Default target device

import unittest

import yaml
from specula.simul import Simul

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
          root_dir: dummy
          
        test:
          class: 'FuncGenerator'
          nmodes_object: null
        '''
        simul = Simul([])
        params = yaml.safe_load(yml)
        simul.build_objects(params)
        
        assert simul.objs['test'].nmodes is None

