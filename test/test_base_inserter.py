import specula
specula.init(0)

import unittest
from specula import cpuArray, np
from specula.processing_objects.base_inserter import BaseInserter
from specula.base_value import BaseValue


def run_inserter(small_array, output_size, **kwargs):
    value = BaseValue(value=small_array)
    value.generation_time = 1
    inserter = BaseInserter(output_size=output_size, **kwargs)
    inserter.inputs['in_value'].set(value)
    inserter.setup()
    inserter.check_ready(1)
    inserter.prepare_trigger(1)
    inserter.trigger()
    inserter.post_trigger()
    return cpuArray(inserter.outputs['out_value'].value)


class TestBaseInserter(unittest.TestCase):

    # --- indices interface ---

    def test_indices_single_pair(self):
        """Insert small vector at explicit indices (single pair)."""
        small = np.array([5, 6, 7], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              indices=[[0, 1, 2], [1, 3, 5]])
        expected = np.array([0, 5, 0, 6, 0, 7, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_indices_two_pairs(self):
        """Insert two groups of indices."""
        small = np.array([5, 6, 7, 8], dtype=np.float64)
        output = run_inserter(small, output_size=8,
                              indices=[[[0, 1], [1, 3]],
                                       [[2, 3], [5, 7]]])
        expected = np.array([0, 5, 0, 6, 0, 7, 0, 8], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    # --- slice_args interface ---

    def test_slice_args_single_pair(self):
        """Insert small vector into a contiguous region (single pair)."""
        small = np.array([10, 20, 30], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              slice_args=[[0, 3], [2, 5]])
        expected = np.array([0, 0, 10, 20, 30, 0, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_slice_args_with_step(self):
        """Insert small vector into a strided region."""
        small = np.array([7, 8, 9], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              slice_args=[[0, 3], [1, 6, 2]])
        expected = np.array([0, 7, 0, 8, 0, 9, 0], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_slice_args_two_pairs(self):
        """Distribute two parts of the small vector into two separate regions."""
        small = np.array([1, 2, 3, 4], dtype=np.float64)
        output = run_inserter(small, output_size=7,
                              slice_args=[[[0, 2], [0, 2]],
                                          [[2, 4], [5, 7]]])
        expected = np.array([1, 2, 0, 0, 0, 3, 4], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    def test_slice_args_three_pairs(self):
        """Distribute three parts with gaps, like a real MORFEO use case."""
        small = np.arange(6, dtype=np.float64)+1
        output = run_inserter(small, output_size=10,
                              slice_args=[[[0, 2], [0, 2]],
                                          [[2, 4], [4, 6]],
                                          [[4, 6], [8, 10]]])
        expected = np.array([1, 2, 0, 0, 3, 4, 0, 0, 5, 6], dtype=np.float64)
        np.testing.assert_array_equal(output, expected)

    # --- error cases ---

    def test_no_args_raises(self):
        with self.assertRaises(ValueError):
            BaseInserter(output_size=5)

    def test_both_args_raises(self):
        with self.assertRaises(ValueError):
            BaseInserter(output_size=5, indices=[[0, 1], [0, 1]], slice_args=[[0, 2], [0, 2]])
