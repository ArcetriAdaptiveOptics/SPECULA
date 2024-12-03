

import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.processing_objects.func_generator import FuncGenerator


class TestFuncGenerator(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_func_generator_constant_gpu(self):
        self._test_func_generator_constant(target_device_idx=0)

    def test_func_generator_constant_cpu(self):
        self._test_func_generator_constant(target_device_idx=-1)

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_func_generator_sin_gpu(self):
        self._test_func_generator_sin(target_device_idx=0)

    def test_func_generator_sin_cpu(self):
        self._test_func_generator_sin(target_device_idx=-1)

    def _test_func_generator_constant(self, target_device_idx):
        constant = 4
        f = FuncGenerator('SIN', target_device_idx=target_device_idx, constant=constant)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        value = cpuArray(f.outputs['output'].value)
        np.testing.assert_almost_equal(value, constant)

    def _test_func_generator_sin(self, target_device_idx):
        amp = 1
        freq = 2
        offset = 3
        constant = 4
        f = FuncGenerator('SIN', target_device_idx=target_device_idx, amp=amp, freq=freq, offset=offset, constant=constant)
        f.run_check(self)

        # Test twice in order to test streams capture, if enabled
        for t in [f.seconds_to_t(x) for x in [0.1, 0.2, 0.3]]:
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            np.testing.assert_almost_equal(value, amp * np.sin(freq*2 * np.pi*f.t_to_seconds(t) + offset) + constant)
        

