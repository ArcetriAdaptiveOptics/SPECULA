import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.utils import unravel_index_2d
from specula.lib.utils import camelcase_to_snakecase
from specula.lib.utils import get_type_hints  # Adjust import path

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

