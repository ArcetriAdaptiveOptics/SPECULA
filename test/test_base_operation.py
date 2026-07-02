import specula
specula.init(0)  # Default target device
from specula.loop_control import LoopControl

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
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)
        
        assert cpuArray(op.outputs['out_value'].value) == 3

    @cpu_and_gpu
    def test_sub(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([1]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([2]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(sub=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == -1

    @cpu_and_gpu
    def test_mul(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([2]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(mul=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 6

    @cpu_and_gpu
    def test_div(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(div=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 2

    @cpu_and_gpu
    def test_concat(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([1, 2]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(concat=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        output_value = cpuArray(op.outputs['out_value'].value)
                     
        np.testing.assert_array_almost_equal(output_value, [1,2,3])

    @cpu_and_gpu
    def test_const_sum(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_sum=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 8

    @cpu_and_gpu
    def test_const_sub(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_sub=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 4

    @cpu_and_gpu
    def test_const_mul(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 12

    @cpu_and_gpu
    def test_const_div(self, target_device_idx, xp):

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_div=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert cpuArray(op.outputs['out_value'].value) == 3

    @cpu_and_gpu
    def test_missing_value2(self, target_device_idx, xp):
        '''Test that setup() raises ValueError when input2 has not been set'''

        value1 = BaseValue(value=xp.array([6.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

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
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

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
        value2 = BaseValue(value=xp.array([2.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(concat=True, target_device_idx=target_device_idx)

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert op.inputs['in_value1'].get(target_device_idx=target_device_idx).value == 1.0

    @cpu_and_gpu
    def test_const_mul_vector(self, target_device_idx, xp):
        """Test constant multiplication with vector"""

        value1 = BaseValue(value=xp.array([2.0, 3.0, 4.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        # Test with list
        op = BaseOperation(constant_mul=[2, 3, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([4.0, 9.0, 2.0])  # [2*2, 3*3, 4*0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_mul_numpy_array(self, target_device_idx, xp):
        """Test constant multiplication with numpy array"""

        value1 = BaseValue(value=xp.array([2.0, 3.0, 4.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        # Test with numpy array
        multiplier = np.array([0.1, 2.0, 1.5])
        op = BaseOperation(constant_mul=multiplier, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([0.2, 6.0, 6.0])  # [2*0.1, 3*2.0, 4*1.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_sum_vector(self, target_device_idx, xp):
        """Test constant addition with vector"""

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_sum=[10, -5, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([11.0, -3.0, 3.5])  # [1+10, 2-5, 3+0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_div_vector(self, target_device_idx, xp):
        """Test constant division with vector (implemented as 1/constant_div * value)"""

        value1 = BaseValue(value=xp.array([6.0, 8.0, 10.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_div=[2, 4, 0.5], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([3.0, 2.0, 20.0])  # [6/2, 8/4, 10/0.5]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_const_sub_vector(self, target_device_idx, xp):
        """Test constant subtraction with vector (implemented as value + (-constant_sub))"""

        value1 = BaseValue(value=xp.array([10.0, 5.0, 3.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_sub=[2, 1, -1], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([8.0, 4.0, 4.0])  # [10-2, 5-1, 3-(-1)]
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu  
    def test_generation_time_set_correctly(self, target_device_idx, xp):
        """Test that generation_time is set for both scalar and vector constants"""

        value1 = BaseValue(value=xp.array([1.0, 2.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(5)

        # Test one scalar and one vector operation
        op_scalar = BaseOperation(constant_mul=2.0, target_device_idx=target_device_idx)
        op_vector = BaseOperation(constant_sum=[1.0, 1.0], target_device_idx=target_device_idx)

        for op in [op_scalar, op_vector]:
            op.inputs['in_value1'].set(value1)
            loop = LoopControl()
            loop.add(op, idx=0)
            loop.run(run_time=10, dt=5, t0=5)
            self.assertEqual(op.outputs['out_value'].generation_time, value1.seconds_to_t(5))

    @cpu_and_gpu
    def test_scalar_vs_vector_consistency(self, target_device_idx, xp):
        """Test that scalar and vector constants give same results when appropriate"""

        value1 = BaseValue(value=xp.array([2.0, 2.0, 2.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        # Test scalar multiplication
        op_scalar = BaseOperation(constant_mul=3.0, target_device_idx=target_device_idx)
        op_scalar.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op_scalar, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Test vector multiplication with same value
        op_vector = BaseOperation(constant_mul=[3.0, 3.0, 3.0], target_device_idx=target_device_idx)
        op_vector.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op_vector, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

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
        value1.generation_time = value1.seconds_to_t(1)

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
        value1.generation_time = value1.seconds_to_t(1)

        # Test that constant_div (scalar) is combined with constant_mul (vector)
        op = BaseOperation(
            constant_mul=[10.0, 20.0],
            constant_div=2.0,
            target_device_idx=target_device_idx
        )
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Should be value1 / 2 * [10.0, 20.0] = [5.0, 20.0]
        expected = xp.array([5.0, 20.0])
        np.testing.assert_array_almost_equal(cpuArray(op.outputs['out_value'].value),
                                             cpuArray(expected))

    @cpu_and_gpu
    def test_empty_arrays(self, target_device_idx, xp):
        """Test behavior with empty arrays"""

        value1 = BaseValue(value=xp.array([]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=2.0, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Should produce empty array
        self.assertEqual(len(op.outputs['out_value'].value), 0)
        self.assertEqual(op.outputs['out_value'].generation_time, value1.seconds_to_t(1))

    @cpu_and_gpu
    def test_never_generated_input_is_not_used(self, target_device_idx, xp):
        '''
        An input that is connected but has never been generated
        (generation_time still at its default -1) must be treated as
        absent (zero) instead of using its raw (potentially garbage) value.
        '''

        value1 = BaseValue(value=xp.array([5.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([999.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        # value2 is connected but never generated: generation_time stays at -1

        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # value2 must be ignored (treated as 0), not 999
        assert cpuArray(op.outputs['out_value'].value) == 5.0

    @cpu_and_gpu
    def test_never_generated_input_with_nan_is_not_used(self, target_device_idx, xp):
        '''
        A never-generated input may contain NaN (e.g. leftover from an
        uninitialized allocation). Multiplying by zero would keep it NaN,
        so it must be replaced with an actual zero array instead.
        '''

        value1 = BaseValue(value=xp.array([5.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([xp.nan]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        # value2 is connected but never generated: generation_time stays at -1

        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # value2 must be treated as 0, not NaN
        assert cpuArray(op.outputs['out_value'].value) == 5.0

    @cpu_and_gpu
    def test_stale_input_keeps_last_value(self, target_device_idx, xp):
        '''
        Once an input has been generated at least once, a later trigger
        where that input is not refreshed should keep using its last
        known value instead of treating it as zero.
        '''

        value1 = BaseValue(value=xp.array([1.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([10.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        op.setup()

        # Step 1: both inputs generated at t=1
        t1 = value1.seconds_to_t(1)
        op.check_ready(t1)
        op.prepare_trigger(t1)
        op.trigger()
        op.post_trigger()
        assert cpuArray(op.outputs['out_value'].value) == 11.0

        # Step 2: only value1 is refreshed, value2 keeps its old generation_time
        value1.value[:] = 2.0
        t2 = value1.seconds_to_t(2)
        value1.generation_time = t2

        op.check_ready(t2)
        op.prepare_trigger(t2)
        op.trigger()
        op.post_trigger()

        # value2 was not refreshed this step, but its last known value (10.0)
        # must still be used, not zero
        assert cpuArray(op.outputs['out_value'].value) == 12.0

    @cpu_and_gpu
    def test_never_generated_input_with_concat(self, target_device_idx, xp):
        '''
        A never-generated input used with concat=True must be zeroed using
        its own shape, not out_value's (concatenated) shape, otherwise the
        slice assignment in trigger_code raises a shape-mismatch error.
        '''

        value1 = BaseValue(value=xp.array([1.0, 2.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0, 4.0, 5.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        # value2 is connected but never generated

        op = BaseOperation(concat=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([1.0, 2.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_never_generated_input_with_value2_remap(self, target_device_idx, xp):
        '''
        A never-generated input used together with value2_remap must be
        zeroed using its own (remapped) shape, not out_value's shape,
        otherwise the remap indexing in trigger_code raises a
        shape-mismatch error.
        '''

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]), target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([10.0]), target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)
        # value2 is connected but never generated

        op = BaseOperation(sum=True, value2_remap=[1], target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # value2 ignored (treated as 0): out = value1 unchanged
        expected = xp.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_multiple_operation_flags_raise(self, target_device_idx, xp):
        """Only one of sum/sub/mul/div/concat can be True"""

        with self.assertRaises(ValueError):
            BaseOperation(sum=True, mul=True, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            BaseOperation(div=True, concat=True, target_device_idx=target_device_idx)


    @cpu_and_gpu
    def test_concat_with_value2_remap_raises(self, target_device_idx, xp):
        """concat and value2_remap cannot be used together"""

        with self.assertRaises(ValueError):
            BaseOperation(concat=True,
                        value2_remap=[0, 1],
                        target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_constants_applied_after_concat(self, target_device_idx, xp):
        """Verify constants are applied after concatenation"""

        value1 = BaseValue(value=xp.array([1.0, 2.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0, 4.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            concat=True,
            constant_mul=2.0,
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Expect concat first → [1,2,3,4], then *2 → [2,4,6,8]
        expected = xp.array([2.0, 4.0, 6.0, 8.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
    )

    @cpu_and_gpu
    def test_mul_div_before_sum_sub(self, target_device_idx, xp):
        """Verify order: mul/div happens before sum/sub"""

        value1 = BaseValue(value=xp.array([2.0]),
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(
            constant_mul=3.0,
            constant_div=2.0,
            constant_sum=4.0,
            constant_sub=1.0,
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Expected order:
        # ((2 * 3) / 2) + 4 - 1 = (6 / 2) + 4 - 1 = 3 + 4 - 1 = 6
        expected = xp.array([6.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_min_max_applied_last(self, target_device_idx, xp):
        """Verify min/max are applied after other constants"""

        value1 = BaseValue(value=xp.array([5.0]),
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(
            constant_mul=2.0,   # → 10
            constant_sum=5.0,   # → 15
            constant_max=12.0,  # → max(15,12)=15
            constant_min=14.0,  # → min(15,14)=14
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([14.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_scalar_constant_broadcast(self, target_device_idx, xp):
        """Scalar constants should broadcast over vectors"""

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0]),
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=2.0,
                        constant_sum=1.0,
                        target_device_idx=target_device_idx)

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # (value * 2) + 1
        expected = xp.array([3.0, 5.0, 7.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_inplace_broadcast_shape_expansion_fails(self, target_device_idx, xp):
        """In-place ops should fail if broadcasting would change shape"""

        value1 = BaseValue(value=xp.array([2.0]),  # shape (1,)
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=xp.array([1.0, 2.0, 3.0]),  # shape (3,)
                        target_device_idx=target_device_idx)

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)

        with self.assertRaises(Exception):
            loop.run(run_time=2, dt=1, t0=1)

    @cpu_and_gpu
    def test_inplace_broadcast_same_shape_via_scalar_axis(self, target_device_idx, xp):
        """Broadcast that keeps same shape should work"""

        value1 = BaseValue(value=xp.array([[1.0, 2.0, 3.0]]),  # shape (1,3)
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=xp.array([10.0, 20.0, 30.0]),  # shape (3,)
                        target_device_idx=target_device_idx)

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([[10.0, 40.0, 90.0]])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_non_broadcastable_shapes_raise(self, target_device_idx, xp):
        """Incompatible shapes should raise an error"""

        value1 = BaseValue(value=xp.array([1.0, 2.0]),
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(constant_mul=xp.array([1.0, 2.0, 3.0]),
                        target_device_idx=target_device_idx)

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)

        with self.assertRaises(Exception):
            loop.run(run_time=2, dt=1, t0=1)

    @cpu_and_gpu
    def test_min_max_broadcasting(self, target_device_idx, xp):
        """min/max should broadcast correctly"""

        value1 = BaseValue(value=xp.array([1.0, 5.0, 10.0]),
                        target_device_idx=target_device_idx)
        value1.generation_time = value1.seconds_to_t(1)

        op = BaseOperation(
            constant_max=4.0,  # → [4,5,10]
            constant_min=xp.array([3.0, 6.0, 8.0]),  # → [3,5,8]
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([3.0, 5.0, 8.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_value2_sum_applied_last(self, target_device_idx, xp):
        """value2 (sum) should be applied after all constant operations"""

        value1 = BaseValue(value=xp.array([2.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            constant_mul=5.0,   # → 2 * 5 = 10
            constant_sum=1.0,   # → 10 + 1 = 11
            sum=True,           # → 11 + 3 = 14
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([14.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_value2_sub_applied_last(self, target_device_idx, xp):
        """value2 (sub) should be applied after constants"""

        value1 = BaseValue(value=xp.array([10.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([4.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            constant_div=2.0,   # → 10 / 2 = 5
            constant_sum=3.0,   # → 5 + 3 = 8
            sub=True,           # → 8 - 4 = 4
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([4.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_value2_mul_applied_last(self, target_device_idx, xp):
        """value2 (mul) should be applied after constants"""

        value1 = BaseValue(value=xp.array([2.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([3.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            constant_mul=2.0,   # → 2 * 2 = 4
            constant_sum=1.0,   # → 4 + 1 = 5
            mul=True,           # → 5 * 3 = 15
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([15.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_value2_div_applied_last(self, target_device_idx, xp):
        """value2 (div) should be applied after constants"""

        value1 = BaseValue(value=xp.array([20.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([5.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            constant_div=2.0,   # → 20 / 2 = 10
            constant_sub=5.0,  # → 10 - 5 = 5
            div=True,           # → 5 / 5 = 1
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([1.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_full_order_constants_then_minmax_then_value2(self, target_device_idx, xp):
        """Verify execution order:
        constants → min/max → value2 operation
        """

        value1 = BaseValue(value=xp.array([2.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([10.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            constant_mul=3.0,   # step 1 → 2 * 3 = 6
            constant_sum=4.0,   # step 1 → 6 + 4 = 10
            constant_min=8.0,   # step 2 → min(10, 8) = 8
            sum=True,           # step 3 → 8 + 10 = 18
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        expected = xp.array([18.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )

    @cpu_and_gpu
    def test_value2_shorter_overlap_only(self, target_device_idx, xp):
        """value2 shorter than value1: operate only on overlapping elements"""

        value1 = BaseValue(value=xp.array([1.0, 2.0, 3.0, 4.0]),
                        target_device_idx=target_device_idx)
        value2 = BaseValue(value=xp.array([10.0, 20.0]),
                        target_device_idx=target_device_idx)

        value1.generation_time = value1.seconds_to_t(1)
        value2.generation_time = value2.seconds_to_t(1)

        op = BaseOperation(
            sum=True,
            target_device_idx=target_device_idx
        )

        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)

        loop = LoopControl()
        loop.add(op, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        # Only first 2 elements affected:
        # [1,2,3,4] + [10,20] → [11,22,3,4]
        expected = xp.array([11.0, 22.0, 3.0, 4.0])

        np.testing.assert_array_almost_equal(
            cpuArray(op.outputs['out_value'].value),
            cpuArray(expected)
        )
