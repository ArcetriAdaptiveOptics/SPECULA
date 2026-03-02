import specula
specula.init(0)

import unittest
from specula import cpuArray, np
from specula.processing_objects.base_inserter import BaseInserter
from specula.base_value import BaseValue


def run_inserter(small_array, output_size, slice_pairs):
    value = BaseValue(value=small_array)
    value.generation_time = 1
    inserter = BaseInserter(output_size=output_size, slice_pairs=slice_pairs)
    inserter.inputs['in_value'].set(value)
    inserter.setup()
    inserter.check_ready(1)
    inserter.prepare_trigger(1)
    inserter.trigger()
    inserter.post_trigger()
    return cpuArray(inserter.outputs['out_value'].value)


class TestBaseInserter(unittest.TestCase):

    def test_single_slice_into_middle(self):
        """Insert the entire small vector into the middle of a larger one."""
        small = np.array([10, 20, 30], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              slice_pairs=[((0, 3), (2, 5))])   # tuple -> slice
        expected = np.array([0, 0, 10, 20, 30, 0, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_two_slices_distributed(self):
        """Distribute two halves of the small vector into two separate regions."""
        small = np.array([1, 2, 3, 4], dtype=np.float64)
        # src[0:2] -> dest[0:2], src[2:4] -> dest[5:7]
        output = run_inserter(small, output_size=7,
                              slice_pairs=[((0, 2), (0, 2)),    # tuple -> slice
                                           ((2, 4), (5, 7))])
        expected = np.array([1, 2, 0, 0, 0, 3, 4], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_three_slices_distributed(self):
        """Distribute three equal parts of the small vector with gaps in between."""
        small = np.arange(6, dtype=np.float64)
        # src[0:2] -> dest[0:2], src[2:4] -> dest[4:6], src[4:6] -> dest[8:10]
        output = run_inserter(small, output_size=10,
                              slice_pairs=[((0, 2), (0, 2)),    # tuple -> slice
                                           ((2, 4), (4, 6)),
                                           ((4, 6), (8, 10))])
        expected = np.array([0, 1, 0, 0, 2, 3, 0, 0, 4, 5], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_with_step(self):
        """Insert using a step: src[0:3] -> dest[1:6:2] (positions 1, 3, 5)."""
        small = np.array([7, 8, 9], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              slice_pairs=[((0, 3), (1, 6, 2))])  # tuple -> slice with step
        expected = np.array([0, 7, 0, 8, 0, 9, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_index_list(self):
        """Use explicit index lists instead of slices."""
        small = np.array([5, 6, 7], dtype=np.float64)
        # src indices [0,1,2] -> dest indices [1,3,5]
        output = run_inserter(small, output_size=7,
                              slice_pairs=[([0, 1, 2], [1, 3, 5])])  # list -> index array
        expected = np.array([0, 5, 0, 6, 0, 7, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_zeros_outside_slices(self):
        """Values outside inserted regions must remain zero."""
        small = np.ones(2, dtype=np.float64)
        output = run_inserter(small, output_size=6,
                              slice_pairs=[((0, 2), (2, 4))])   # tuple -> slice
        np.testing.assert_array_equal(output[:2], [0, 0])
        np.testing.assert_array_equal(output[4:], [0, 0])

    def test_empty_slice_pairs_raises(self):
        """Constructing with an empty slice_pairs list must raise ValueError."""
        with self.assertRaises(ValueError):
            BaseInserter(output_size=5, slice_pairs=[])
