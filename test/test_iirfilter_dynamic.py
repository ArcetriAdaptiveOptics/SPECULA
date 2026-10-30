'''
DynamicIntegrator and DynamicIirFilter: gain changed and state reset by
their optional inputs during the simulation.
'''
import specula
specula.init(0)  # Default target device

import unittest

import numpy as np

from iteras import cpuArray
from iteras.base_value import BaseValue
from iteras.scalar_values import FloatValue, IntValue
from iteras.data_objects.iir_filter_data import IirFilterData
from iteras.processing_objects.dynamic_integrator import DynamicIntegrator
from iteras.processing_objects.dynamic_iir_filter import DynamicIirFilter

from test.specula_testlib import cpu_and_gpu


class TestDynamicFilters(unittest.TestCase):

    def _run(self, filt, target_device_idx, xp):
        '''
        Integrate a constant input of 1 with gains [0.5, 0.3]: the gain becomes
        1 at the second step, and the state is reset at the third step.
        '''
        self.assertIn('int_gain', type(filt).input_names())
        self.assertIn('reset', type(filt).input_names())
        self.assertIn('out_comm', type(filt).output_names())

        constant = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                             target_device_idx=target_device_idx)
        gain = FloatValue(value=1.0)
        reset = IntValue(value=1)
        filt.inputs['delta_comm'].set(constant)
        filt.inputs['int_gain'].set(gain)
        filt.inputs['reset'].set(reset)
        filt.setup()

        outputs = []
        for step in range(3):
            t = filt.seconds_to_t(0.001 * step)
            constant.generation_time = t
            gain.generation_time = t if step == 1 else -1
            reset.generation_time = t if step == 2 else -1
            filt.check_ready(t)
            filt.trigger()
            filt.post_trigger()
            outputs.append(cpuArray(filt.outputs['out_comm'].value).copy())

        np.testing.assert_allclose(outputs[0], [0.5, 0.3], rtol=1e-5)
        np.testing.assert_allclose(outputs[1], [1.5, 1.3], rtol=1e-5)  # new gain 1
        np.testing.assert_allclose(outputs[2], [1.0, 1.0], rtol=1e-5)  # state reset

    @cpu_and_gpu
    def test_dynamic_integrator(self, target_device_idx, xp):
        filt = DynamicIntegrator(int_gain=[0.5, 0.3], target_device_idx=target_device_idx)
        self._run(filt, target_device_idx, xp)

    @cpu_and_gpu
    def test_dynamic_iir_filter(self, target_device_idx, xp):
        data = IirFilterData.from_gain_and_ff([0.5, 0.3], target_device_idx=target_device_idx)
        filt = DynamicIirFilter(data, target_device_idx=target_device_idx)
        self._run(filt, target_device_idx, xp)


@unittest.skipIf(specula.cp is None, 'Cupy not found')
class TestDynamicFiltersAccelerated(unittest.TestCase):
    '''
    A new gain replaces the coefficient arrays of the filter: its CUDA graph is
    captured again (invalidate_graph()), and it stays accelerated
    '''

    def _run(self, accelerate, make_filter):
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
        constant = BaseValue(value=np.array([1.0, 1.0], dtype=np.float32), target_device_idx=0)
        gain = FloatValue(value=1.0)
        reset = IntValue(value=1)
        filt = make_filter()
        filt.inputs['delta_comm'].set(constant)
        filt.inputs['int_gain'].set(gain)
        filt.inputs['reset'].set(reset)
        accelerator = None
        if accelerate:
            accelerator = Accelerator(AcceleratorOptions())
            accelerator.apply([filt])
        filt.setup()
        outputs = []
        for step in range(7):
            t = filt.seconds_to_t(0.001 * step)
            constant.generation_time = t
            gain.value = 1.0 + 0.5 * step
            gain.generation_time = t if step in (3, 5) else -1     # after the capture (step 2)
            reset.generation_time = t if step == 4 else -1
            filt.check_ready(t)
            filt.trigger()
            filt.post_trigger()
            outputs.append(cpuArray(filt.outputs['out_comm'].value).copy())
        return np.array(outputs), filt, accelerator

    def check(self, make_filter):
        ref, _, _ = self._run(False, make_filter)
        with self.assertNoLogs('iteras.accelerator', level='WARNING'):
            out, filt, accelerator = self._run(True, make_filter)
        np.testing.assert_allclose(out, ref, rtol=1e-6)
        self.assertIsNotNone(filt._accel)
        self.assertEqual(filt._accel.status, 'accelerated')

    def test_dynamic_integrator(self):
        self.check(lambda: DynamicIntegrator(int_gain=[0.5, 0.3], target_device_idx=0))

    def test_dynamic_iir_filter(self):
        self.check(lambda: DynamicIirFilter(IirFilterData.from_gain_and_ff([0.5, 0.3], target_device_idx=0),
                                            target_device_idx=0))


if __name__ == '__main__':
    unittest.main()
