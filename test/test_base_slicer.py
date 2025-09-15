import specula
specula.init(0)

import unittest
from specula import cpuArray, np
from specula.processing_objects.base_slicer import BaseSlicer
from specula.base_value import BaseValue

class TestBaseSlicer(unittest.TestCase):

    def test_indices(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = 1
        slicer = BaseSlicer(indices=[1, 3, 5])
        slicer.inputs['in_value'].set(value)

        slicer.setup()
        slicer.check_ready(1)
        slicer.prepare_trigger(1)
        slicer.trigger()
        slicer.post_trigger()

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [1, 3, 5])

    def test_slice_args(self):
        arr = np.arange(10)
        value = BaseValue(value=arr)
        value.generation_time = 1
        slicer = BaseSlicer(slice_args=[2, 7, 2])
        slicer.inputs['in_value'].set(value)

        slicer.setup()
        slicer.check_ready(1)
        slicer.prepare_trigger(1)
        slicer.trigger()
        slicer.post_trigger()

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, [2, 4, 6])

    def test_no_args(self):
        arr = np.arange(5)
        value = BaseValue(value=arr)
        value.generation_time = 1
        slicer = BaseSlicer()
        slicer.inputs['in_value'].set(value)

        slicer.setup()
        slicer.check_ready(1)
        slicer.prepare_trigger(1)
        slicer.trigger()
        slicer.post_trigger()

        output = cpuArray(slicer.outputs['out_value'].value)
        np.testing.assert_array_equal(output, arr)