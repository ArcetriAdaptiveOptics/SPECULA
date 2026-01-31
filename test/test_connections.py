import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cp
from specula import cpuArray

from specula.base_value import BaseValue
from specula.connections import InputList, InputValue

from test.specula_testlib import cpu_and_gpu

class TestConnections(unittest.TestCase):
   
    @cpu_and_gpu
    def test_input_value_same_device(self, target_device_idx, xp):

        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        result = input_v.get(target_device_idx=target_device_idx)
        assert(result.target_device_idx == target_device_idx)
        np.testing.assert_array_equal(cpuArray(data), cpuArray(result.value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_value_other_device(self, target_device_idx, xp):
            
        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result = input_v.get(target_device_idx=my_target)
        assert(result.target_device_idx == my_target)
        np.testing.assert_array_equal(cpuArray(data), cpuArray(result.value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_value_transfer_does_not_allocate_a_new_object(self, target_device_idx, xp):

        data = xp.arange(2)
        input_v = InputValue(type=BaseValue)

        output_v = BaseValue(value=data, target_device_idx=target_device_idx)
        input_v.set(output_v)

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result1 = input_v.get(target_device_idx=my_target)
        result2 = input_v.get(target_device_idx=my_target)
        assert(id(result1) == id(result2))

    @cpu_and_gpu
    def test_input_list_same_device(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        result = input_v.get(target_device_idx=target_device_idx)
        assert(result[0].target_device_idx == target_device_idx)
        assert(result[1].target_device_idx == target_device_idx)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_other_device(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result = input_v.get(target_device_idx=my_target)

        assert(result[0].target_device_idx == my_target)
        assert(result[1].target_device_idx == my_target)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_transfer_does_not_allocate_a_new_object(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1, output2])

        if target_device_idx == 0:
            my_target = -1
        else:
            my_target = 0

        result1 = input_v.get(target_device_idx=my_target)
        result2 = input_v.get(target_device_idx=my_target)
        assert(id(result1[0]) == id(result2[0]))
        assert(id(result1[1]) == id(result2[1]))

    @cpu_and_gpu
    @unittest.skipIf(cp is None, "cupy not installed")
    def test_input_list_can_append(self, target_device_idx, xp):

        data1 = xp.arange(2)
        data2 = xp.arange(2)+2
        input_v = InputList(type=BaseValue)

        output1 = BaseValue(value=data1, target_device_idx=target_device_idx)
        output2 = BaseValue(value=data2, target_device_idx=target_device_idx)
        input_v.set([output1])
        input_v.append(output2)

        result = input_v.get(target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(data1), cpuArray(result[0].value))
        np.testing.assert_array_equal(cpuArray(data2), cpuArray(result[1].value))

