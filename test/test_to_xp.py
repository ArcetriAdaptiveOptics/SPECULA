import specula
specula.init(0)

import unittest
import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

from specula import to_xp  # Adjust import


class TestToXp(unittest.TestCase):

    def test_numpy_array_no_copy(self):
        xp = np
        arr = np.array([1, 2, 3])
        result = to_xp(xp, arr)
        self.assertIs(result, arr)  # no copy
        self.assertTrue(np.all(result == arr))

    def test_numpy_array_force_copy(self):
        xp = np
        arr = np.array([1, 2, 3])
        result = to_xp(xp, arr, force_copy=True)
        self.assertIsNot(result, arr)
        self.assertTrue(np.all(result == arr))

    def test_numpy_list_input(self):
        xp = np
        lst = [1, 2, 3]
        result = to_xp(xp, lst)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(np.all(result == np.array(lst)))

    def test_numpy_with_dtype(self):
        xp = np
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = to_xp(xp, arr, dtype=np.float64)
        self.assertEqual(result.dtype, np.float64)

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_cupy_array_no_copy(self):
        xp = cp
        arr = cp.array([1, 2, 3])
        result = to_xp(xp, arr)
        self.assertIs(result, arr)  # no copy
        self.assertTrue(cp.all(result == arr))

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_cupy_array_force_copy(self):
        xp = cp
        arr = cp.array([1, 2, 3])
        result = to_xp(xp, arr, force_copy=True)
        self.assertIsNot(result, arr)
        self.assertTrue(cp.all(result == arr))

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_cupy_list_input(self):
        xp = cp
        lst = [1, 2, 3]
        result = to_xp(xp, lst)
        self.assertTrue(isinstance(result, cp.ndarray))
        self.assertTrue(cp.all(result == cp.array(lst)))

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_numpy_array_to_cupy(self):
        xp = cp
        arr = np.array([1, 2, 3])
        result = to_xp(xp, arr)
        self.assertTrue(isinstance(result, cp.ndarray))
        self.assertTrue(cp.all(result == cp.array(arr)))

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_cupy_array_to_numpy(self):
        xp = np
        arr = cp.array([1, 2, 3])
        result = to_xp(xp, arr)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(np.all(result == arr.get()))

    def test_dtype_and_force_copy_numpy(self):
        xp = np
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = to_xp(xp, arr, dtype=np.float64, force_copy=True)
        self.assertEqual(result.dtype, np.float64)
        self.assertIsNot(result, arr)

    @unittest.skipUnless(CUPY_AVAILABLE, "CuPy not installed")
    def test_dtype_and_force_copy_cupy(self):
        xp = cp
        arr = cp.array([1, 2, 3], dtype=cp.int32)
        result = to_xp(xp, arr, dtype=cp.float64, force_copy=True)
        self.assertEqual(result.dtype, cp.float64)
        self.assertIsNot(result, arr)



