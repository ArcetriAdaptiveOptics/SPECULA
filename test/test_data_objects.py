
import unittest
import importlib
import pkgutil
import inspect

import specula.data_objects

from specula.base_data_obj import BaseDataObj


class TestDataObjects(unittest.TestCase):

    def test_all_data_objects(self):
        '''
        Test that all data objects have the mandatory methods
        
        get_value, set_value, save, restore, from_header and get_fits_header
        '''
        def generate_data_classes():
            # Iterate over all modules in the package path
            for finder, name, ispkg in pkgutil.iter_modules(specula.data_objects.__path__):

                # Import submodule
                full_name = f"{specula.data_objects.__name__}.{name}"
                module = importlib.import_module(full_name)

                skip = ['InfinitePhaseScreen', 'SimulParams', 'SubapData', 'TimeHistory']
                # List all classes defined in that module
                classes = [value for name, value in inspect.getmembers(module, inspect.isclass) if name not in skip]

                # Filter: only classes whose __module__ matches the submodule (not external ones)
                classes = [cls for cls in classes if cls.__module__ == module.__name__]

                for c in classes:
                    yield c

        for klass in generate_data_classes():
            assert hasattr(klass, 'get_value')
            assert hasattr(klass, 'set_value')
            assert hasattr(klass, 'save')
            assert hasattr(klass, 'restore')
            assert hasattr(klass, 'from_header')
            assert hasattr(klass, 'get_fits_header')
