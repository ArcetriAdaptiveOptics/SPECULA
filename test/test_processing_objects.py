import unittest

import specula
specula.init(0)  # Default target device

from specula.base_processing_obj import InputDesc, OutputDesc
from test.specula_testlib import iter_processing_object_classes


class TestProcessingObjects(unittest.TestCase):

    def test_input_names(self):
        '''
        Test that all data objects define an input_names dictionary
        where each pair is (string, InputDesc namedtuple)
        
        '''
        test = ['ModulatedPyramid', 'ExtSourcePyramid']

        for klass in iter_processing_object_classes():
            if klass.__name__ not in test:
                continue
            
            assert type(klass.input_names()) is dict
            for k, v in klass.input_names().items():
                assert type(k) is str
                assert type(v) is InputDesc

    def test_output_names(self):
        '''
        Test that all data objects define an output_names dictionary
        where each pair is (string, OutputDesc namedtuple)

        '''
        test = ['ModulatedPyramid', 'ExtSourcePyramid']

        for klass in iter_processing_object_classes():
            if klass.__name__ not in test:
                continue
            
            assert type(klass.output_names()) is dict
            for k, v in klass.output_names().items():
                assert type(k) is str
                assert type(v) is OutputDesc

