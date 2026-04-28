import specula
specula.init(0)  # Default target device

import time
import unittest
import sys
import types
from typing import Union, Dict, List

from specula import np
from specula import cpuArray

from specula.lib.utils import unravel_index_2d
from specula.lib.utils import camelcase_to_snakecase
from specula.lib.utils import get_type_hints
from specula.lib.utils import remove_suffix
from specula.lib.utils import make_tn
from specula.lib.utils import resolve_type

from test.specula_testlib import cpu_and_gpu

class TestUtils(unittest.TestCase):
   
    @cpu_and_gpu
    def test_unravel_index_square_shape(self, target_device_idx, xp):
        
        idxs = xp.array([1,2,3])
        shape = (3,3)
        y, x = unravel_index_2d(idxs, shape, xp) 
        ytest, xtest = xp.unravel_index(idxs, shape)
        np.testing.assert_array_almost_equal(cpuArray(x), cpuArray(xtest))
        np.testing.assert_array_almost_equal(cpuArray(y), cpuArray(ytest))

    @cpu_and_gpu
    def test_unravel_index_rectangular_shape(self, target_device_idx, xp):
       
        idxs = xp.array([2,6,13])
        shape = (4,8)
        y, x = unravel_index_2d(idxs, shape, xp) 
        ytest, xtest = xp.unravel_index(idxs, shape)
        np.testing.assert_array_almost_equal(cpuArray(x), cpuArray(xtest))
        np.testing.assert_array_almost_equal(cpuArray(y), cpuArray(ytest))

    @cpu_and_gpu
    def test_unravel_index_wrong_shape(self, target_device_idx, xp):

        with self.assertRaises(ValueError):
            _ = unravel_index_2d([1,2,3], (1,2,3), xp)

    def test_camelcase_to_snakecase(self):
        assert camelcase_to_snakecase('IFunc') == 'ifunc'
        assert camelcase_to_snakecase('M2C') == 'm2c'
        assert camelcase_to_snakecase('BaseValue') == 'base_value'
        assert camelcase_to_snakecase('CCD') == 'ccd'


class TestGetTypeHints(unittest.TestCase):

    def test_simple_class(self):
        class A:
            def __init__(self, x: int, y: str):
                pass

        hints = get_type_hints(A)
        self.assertEqual(hints, {'x': int, 'y': str})

    def test_inherited_class_merges_hints(self):
        class A:
            def __init__(self, x: int):
                pass

        class B(A):
            def __init__(self, y: float):
                pass

        hints = get_type_hints(B)
        # Should merge parent's and child's
        self.assertEqual(hints, {'x': int, 'y': float})

    def test_class_with_no_annotations(self):
        class A:
            def __init__(self, x, y):
                pass

        hints = get_type_hints(A)
        self.assertEqual(hints, {})

    def test_multiple_inheritance(self):
        class A:
            def __init__(self, x: int):
                pass

        class B:
            def __init__(self, y: str):
                pass

        class C(A, B):
            def __init__(self, z: float):
                pass

        hints = get_type_hints(C)
        # Collects all hints from C, A, and B
        self.assertEqual(hints, {'x': int, 'y': str, 'z': float})

    def test_child_overrides_parent_hint(self):
        class A:
            def __init__(self, value: int):
                pass

        class B(A):
            def __init__(self, value: str):
                pass

        hints = get_type_hints(B)
        # Child's annotation overrides parent's
        self.assertEqual(hints, {'value': str})

    def test_class_without_init(self):
        class A:
            pass

        hints = get_type_hints(A)
        # Default __init__ has no annotations
        self.assertEqual(hints, {})

    def test_remove_suffix(self):
        self.assertEqual(remove_suffix('parameter_ref', '_ref'), 'parameter')
        self.assertEqual(remove_suffix('parameter_data', '_data'), 'parameter')
        self.assertEqual(remove_suffix('parameter_object', '_object'), 'parameter')
        self.assertEqual(remove_suffix('parameter', '_ref'), 'parameter')  # No suffix to remove

    def test_make_tn_format(self):
        """Output should match YYYYMMDD_HHMMSS format"""
        tn = make_tn()
        pattern = r"^\d{8}_\d{6}$"
        self.assertRegex(tn, pattern)

    def test_make_tn_changes_over_time(self):
        """Two calls separated by time should produce different values"""
        tn1 = make_tn()
        time.sleep(2)
        tn2 = make_tn()
        self.assertNotEqual(tn1, tn2)


class Recmat:
    pass


class TestResolveType(unittest.TestCase):

    # --- Basic behavior ---

    def test_dict_type(self):
        self.assertEqual(resolve_type(Dict[str, float]), float)

    def test_list_type(self):
        self.assertEqual(resolve_type(List[float]), float)

    def test_union_type(self):
        self.assertEqual(resolve_type(Union[float, None]), float)

    def test_plain_type(self):
        self.assertEqual(resolve_type(float), float)

    def test_custom_type(self):
        self.assertEqual(resolve_type(Recmat), Recmat)

    # --- Custom types inside containers ---

    def test_dict_custom(self):
        self.assertEqual(resolve_type(Dict[str, Recmat]), Recmat)

    def test_list_custom(self):
        self.assertEqual(resolve_type(List[Recmat]), Recmat)

    def test_union_custom(self):
        self.assertEqual(resolve_type(Union[Recmat, None]), Recmat)

    # --- PEP 604 unions (Python 3.10+) ---

    @unittest.skipIf(sys.version_info < (3, 10), "Requires Python 3.10+")
    def test_pep604_union(self):
        self.assertEqual(resolve_type(Recmat | None), Recmat)

    # --- require_list flag ---

    def test_require_list_success(self):
        self.assertEqual(resolve_type(List[int], require_list=True), int)

    def test_require_list_failure_on_plain(self):
        with self.assertRaises(TypeError):
            resolve_type(int, require_list=True)

    def test_require_list_failure_on_dict(self):
        with self.assertRaises(TypeError):
            resolve_type(Dict[str, int], require_list=True)

    # --- require_dict flag ---

    def test_require_dict_success(self):
        self.assertEqual(resolve_type(Dict[str, int], require_dict=True), int)

    def test_require_dict_failure_on_plain(self):
        with self.assertRaises(TypeError):
            resolve_type(int, require_dict=True)

    def test_require_dict_failure_on_list(self):
        with self.assertRaises(TypeError):
            resolve_type(List[int], require_dict=True)

    # --- Combined flags (edge cases) ---

    def test_require_both_flags_fail(self):
        # Cannot be both list and dict
        with self.assertRaises(TypeError):
            resolve_type(List[int], require_list=True, require_dict=True)

    def test_require_list_with_union_fails(self):
        with self.assertRaises(TypeError):
            resolve_type(Union[int, None], require_list=True)

    def test_require_dict_with_union_fails(self):
        with self.assertRaises(TypeError):
            resolve_type(Union[int, None], require_dict=True)

    # --- Nested structures (documented behavior: one level only) ---

    def test_nested_list(self):
        self.assertEqual(resolve_type(List[List[int]]), List[int])

    def test_nested_dict(self):
        self.assertEqual(resolve_type(Dict[str, Dict[str, int]]), Dict[str, int])

    # --- Union ordering behavior ---

    def test_union_multiple_types(self):
        self.assertEqual(resolve_type(Union[int, float, None]), int)



