import unittest

import specula
specula.init(0)  # Default target device

from test.specula_testlib import iter_data_object_classes

class TestDataObjects(unittest.TestCase):

    def test_all_data_objects(self):
        '''
        Test that all data objects have the mandatory methods
        
        get_value, set_value, save, restore, from_header and get_fits_header
        '''
        skip = ['InfinitePhaseScreen', 'SimulParams', 'SubapData']

        for klass in iter_data_object_classes(skip=skip):
            assert hasattr(klass, 'get_value')
            assert hasattr(klass, 'set_value')
            assert hasattr(klass, 'save')
            assert hasattr(klass, 'restore')
            assert hasattr(klass, 'from_header')
            assert hasattr(klass, 'get_fits_header')
