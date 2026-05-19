'''
Test file for functions in specula/__init__.py
'''

import os
import importlib
import sys
import specula

import unittest

from specula import np, cp, array_types
from specula import cpuArray


def reload_specula():
    import specula
    importlib.reload(specula)
    return specula


class TestInit(unittest.TestCase):
   
    def test_cpuArray_from_cpu_to_cpu_without_copy(self):
        data = np.arange(3)
        assert id(data) == id(cpuArray(data))

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_cpuArray_from_gpu_to_cpu(self):
        data = cp.arange(3)
        data_cpu = cpuArray(data)
        assert isinstance(data_cpu, np.ndarray)
        np.testing.assert_array_equal(np.arange(3), data_cpu)

    @unittest.skipIf(cp is None, 'GPU not available')
    def test_array_types_with_gpu(self):
        '''Test that the array_types list contains both numpy and cupy arrays'''
        assert len(array_types) == 2
        assert np.ndarray in array_types
        assert cp.ndarray in array_types

    @unittest.skipIf(cp is not None, 'Test for non-GPU configurations')
    def test_array_types_no_gpu(self):
        '''Test that the array_types list contains numpy arrays only'''
        assert array_types == [np.ndarray]


class TestSpeculaDisableGPU(unittest.TestCase):

    def setUp(self):
        # Save original environment state
        self.original_value = os.environ.get("SPECULA_DISABLE_GPU")
        if 'specula' in sys.modules:    # pragma: no cover
            self.orig_specula = sys.modules['specula']
        else:                           # pragma: no cover
            self.orig_specula = None

    def tearDown(self):
        # Restore environment exactly as it was
        if self.original_value is None:     # pragma: no cover
            os.environ.pop("SPECULA_DISABLE_GPU", None)
        else:                               # pragma: no cover
            os.environ["SPECULA_DISABLE_GPU"] = self.original_value

        # Reset imported module state
        if self.orig_specula:     # pragma: no cover
            sys.modules['specula'] = self.orig_specula
        else:                     # pragma: no cover
            if "specula" in sys.modules:
                del sys.modules["specula"]

    def test_gpu_disabled_when_env_not_false(self):
        os.environ["SPECULA_DISABLE_GPU"] = "TRUE"

        specula = reload_specula()
        specula.init(device_idx=0)

        self.assertEqual(specula.default_target_device_idx, -1)
        self.assertEqual(specula.xp.__name__, "numpy")

    def test_gpu_enabled_when_env_false(self):
        os.environ["SPECULA_DISABLE_GPU"] = "FALSE"

        specula = reload_specula()
        specula.init(device_idx=0)

        # If cupy is installed, GPU path is active
        if specula.cp is not None:     # pragma: no cover
            self.assertTrue(specula.gpuEnabled)
        else:
            # fallback if cupy is missing
            self.assertEqual(specula.xp.__name__, "numpy")