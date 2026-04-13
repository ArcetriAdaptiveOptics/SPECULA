import inspect
import unittest

import specula
specula.init(0)  # Default target device

from specula.base_processing_obj import InputDesc, OutputDesc
from test.specula_testlib import iter_processing_object_classes


class TestInputOutputNames(unittest.TestCase):

    def test_all_classmethods_input_names(self):
        '''
        Test that all processing object classes with a classmethod input_names
        return a dict where each key is str and each value is an InputDesc namedtuple.
        '''
        for klass in iter_processing_object_classes():
            static = inspect.getattr_static(klass, 'input_names', None)
            if static is None or not isinstance(static, classmethod):
                continue
            result = klass.input_names()
            self.assertIsInstance(
                result, dict,
                f"{klass.__name__}.input_names() must return dict"
            )
            for k, v in result.items():
                self.assertIsInstance(
                    k, str,
                    f"{klass.__name__} input key must be str"
                )
                self.assertIsInstance(
                    v, InputDesc,
                    f"{klass.__name__} input value must be InputDesc"
                )

    def test_all_classmethods_output_names(self):
        '''
        Test that all processing object classes with a classmethod output_names
        return a dict where each key is str and each value is an OutputDesc namedtuple.
        '''
        for klass in iter_processing_object_classes():
            static = inspect.getattr_static(klass, 'output_names', None)
            if static is None or not isinstance(static, classmethod):
                continue
            result = klass.output_names()
            self.assertIsInstance(
                result, dict,
                f"{klass.__name__}.output_names() must return dict"
            )
            for k, v in result.items():
                self.assertIsInstance(
                    k, str,
                    f"{klass.__name__} output key must be str"
                )
                self.assertIsInstance(
                    v, OutputDesc,
                    f"{klass.__name__} output value must be OutputDesc"
                )


if __name__ == '__main__':
    unittest.main()