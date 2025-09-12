
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray, np
from specula.processing_objects.base_operation import BaseOperation
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu

class TestBaseOperation(unittest.TestCase):

    @cpu_and_gpu
    def test_sum(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([1]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 3

    @cpu_and_gpu
    def test_sub(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([1]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(sub=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == -1

    @cpu_and_gpu
    def test_mul(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([2]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(mul=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 6

    @cpu_and_gpu
    def test_div(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(div=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 2

    @cpu_and_gpu
    def test_concat(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([1, 2]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(concat=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        output_value = cpuArray(op.outputs['out_value'].value)
                                
        np.testing.assert_array_almost_equal(output_value, [1,2,3])
        
    @cpu_and_gpu
    def test_const_sum(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_sum=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 8

    @cpu_and_gpu
    def test_const_sub(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_sub=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 4

    @cpu_and_gpu
    def test_const_mul(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_mul=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 12
        
    @cpu_and_gpu
    def test_const_div(self, target_device_idx, xp):
        
        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_div=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 3

    @cpu_and_gpu
    def test_missing_value2(self, target_device_idx, xp):
        '''Test that setup() raises ValueError when input2 has not been set'''

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        # All these must raise an exception in setup() with a single input
        ops = []
        ops.append(BaseOperation(sum=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(sub=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(mul=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(div=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(concat=True, target_device_idx=target_device_idx))
        
        for op in ops:
            op.inputs['in_value1'].set(value1)
            with self.assertRaises(ValueError):
                op.setup()

        # constant mul/div do not raise any exception in setup() with a single input
        ops = []
        ops.append(BaseOperation(constant_mul=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(constant_div=True, target_device_idx=target_device_idx))
          
        for op in ops:
            op.inputs['in_value1'].set(value1)
            # Does not raise
            op.setup()  


    @cpu_and_gpu
    def test_that_value1_is_not_overwritten(self, target_device_idx, xp):
        '''Test that value1 is not overwritten'''

        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value2.generation_time = 1
        
        ops = []
        ops.append(BaseOperation(sum=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(sub=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(mul=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(div=True, target_device_idx=target_device_idx))
        ops.append(BaseOperation(constant_mul=2, target_device_idx=target_device_idx))
        ops.append(BaseOperation(constant_div=3, target_device_idx=target_device_idx))
        
        for op in ops:
            op.inputs['in_value1'].set(value1)
            op.inputs['in_value2'].set(value2)
            op.setup()
            op.check_ready(1)
            op.prepare_trigger(1)
            op.trigger()
            op.post_trigger()
            assert op.inputs['in_value1'].get(target_device_idx=target_device_idx).value == 1.0

        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value2.generation_time = 1
        
        op = BaseOperation(concat=True, target_device_idx=target_device_idx)
        
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()
        assert op.inputs['in_value1'].get(target_device_idx=target_device_idx).value == 1.0
