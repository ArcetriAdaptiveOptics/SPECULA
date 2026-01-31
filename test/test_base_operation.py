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

    @cpu_and_gpu
    def test_const_mul_vector(self, target_device_idx, xp):
        """Test constant multiplication with vector"""

        value1 = BaseValue(value=xp.array([2.0, 3.0, 4.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        # Test with list
        op = BaseOperation(constant_mul=[2, 3, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        expected = xp.array([4.0, 9.0, 2.0])  # [2*2, 3*3, 4*0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_mul_numpy_array(self, target_device_idx, xp):
        """Test constant multiplication with numpy array"""

        value1 = BaseValue(value=xp.array([2.0, 3.0, 4.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        # Test with numpy array
        multiplier = np.array([0.1, 2.0, 1.5])
        op = BaseOperation(constant_mul=multiplier, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        expected = xp.array([0.2, 6.0, 6.0])  # [2*0.1, 3*2.0, 4*1.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_sum_vector(self, target_device_idx, xp):
        """Test constant addition with vector"""

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        op = BaseOperation(constant_sum=[10, -5, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        expected = xp.array([11.0, -3.0, 3.5])  # [1+10, 2-5, 3+0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_div_vector(self, target_device_idx, xp):
        """Test constant division with vector (implemented as 1/constant_div * value)"""

        value1 = BaseValue(value=xp.array([6.0, 8.0, 10.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        op = BaseOperation(constant_div=[2, 4, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        expected = xp.array([3.0, 2.0, 20.0])  # [6/2, 8/4, 10/0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_sub_vector(self, target_device_idx, xp):
        """Test constant subtraction with vector (implemented as value + (-constant_sub))"""

        value1 = BaseValue(value=xp.array([10.0, 5.0, 3.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        op = BaseOperation(constant_sub=[2, 1, -1], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        expected = xp.array([8.0, 4.0, 4.0])  # [10-2, 5-1, 3-(-1)]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu  
    def test_generation_time_set_correctly(self, target_device_idx, xp):
        """Test that generation_time is set for both scalar and vector constants"""

        value1 = BaseValue(value=xp.array([1.0, 2.0]), target_device_idx=target_device_idx)
        value1.generation_time = 5

        # Test one scalar and one vector operation
        op_scalar = BaseOperation(constant_mul=2.0, target_device_idx=target_device_idx)
        op_vector = BaseOperation(constant_sum=[1.0, 1.0], target_device_idx=target_device_idx)

        for op in [op_scalar, op_vector]:
            op.inputs['in_value1'].set(value1)
            op.setup()
            op.check_ready(5)
            op.prepare_trigger(5)
            op.trigger()
            op.post_trigger()
            self.assertEqual(op.outputs['out_value'].generation_time, 5)

    @cpu_and_gpu
    def test_scalar_vs_vector_consistency(self, target_device_idx, xp):
        """Test that scalar and vector constants give same results when appropriate"""

        value1 = BaseValue(value=xp.array([2.0, 2.0, 2.0]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        # Test scalar multiplication
        op_scalar = BaseOperation(constant_mul=3.0, target_device_idx=target_device_idx)
        op_scalar.inputs['in_value1'].set(value1)
        op_scalar.setup()
        op_scalar.check_ready(1)
        op_scalar.prepare_trigger(1)
        op_scalar.trigger()
        op_scalar.post_trigger()

        # Test vector multiplication with same value
        op_vector = BaseOperation(constant_mul=[3.0, 3.0, 3.0], target_device_idx=target_device_idx)
        op_vector.inputs['in_value1'].set(value1)
        op_vector.setup()
        op_vector.check_ready(1)
        op_vector.prepare_trigger(1)
        op_vector.trigger()
        op_vector.post_trigger()

        # Results should be identical
        np.testing.assert_array_equal(
            cpuArray(op_scalar.outputs['out_value'].value),
            cpuArray(op_vector.outputs['out_value'].value)
        )

    @cpu_and_gpu
    def test_vector_dimension_mismatch(self, target_device_idx, xp):
        """Test error handling when vector constant has wrong dimensions"""

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]),
                           target_device_idx=target_device_idx)
        value1.generation_time = 1

        # Vector with wrong length should cause an error during trigger
        op = BaseOperation(constant_mul=[1.0, 2.0],
                           target_device_idx=target_device_idx)  # 2 elements vs 3
        op.inputs['in_value1'].set(value1)
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)

        # This should raise an error due to shape mismatch
        with self.assertRaises((ValueError, RuntimeError)):
            op.trigger()

    @cpu_and_gpu
    def test_mixed_scalar_vector_constants(self, target_device_idx, xp):
        """Test that mixing scalars and vectors in different operations works"""

        value1 = BaseValue(value=xp.array([1.0, 2.0]),
                           target_device_idx=target_device_idx)
        value1.generation_time = 1

        # Test that constant_div (scalar) overrides constant_mul (vector)
        # According to your code, the last one wins
        op = BaseOperation(
            constant_mul=[10.0, 20.0],  # This should be overridden
            constant_div=2.0,           # This should win
            target_device_idx=target_device_idx
        )
        op.inputs['in_value1'].set(value1)
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        # Should be value1 / 2.0 = [0.5, 1.0]
        expected = xp.array([0.5, 1.0])
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_empty_arrays(self, target_device_idx, xp):
        """Test behavior with empty arrays"""

        value1 = BaseValue(value=xp.array([]), target_device_idx=target_device_idx)
        value1.generation_time = 1

        op = BaseOperation(constant_mul=2.0, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.setup()
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        # Should produce empty array
        self.assertEqual(len(op.outputs['out_value'].value), 0)
        self.assertEqual(op.outputs['out_value'].generation_time, 1)