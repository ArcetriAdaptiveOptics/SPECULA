import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.mvm import MVM
from specula.data_objects.recmat import Recmat
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu


class TestMVM(unittest.TestCase):

    @cpu_and_gpu
    def test_mvm_basic_multiplication(self, target_device_idx, xp):
        """Test basic matrix-vector multiplication"""
        # Create a simple 3x4 matrix
        matrix = xp.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ], dtype=xp.float64)

        recmat = Recmat(matrix, target_device_idx=target_device_idx)

        # Create MVM
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Create input vector
        input_vector = xp.array([1, 2, 3, 4], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        # Set up and run
        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()
        mvm.prepare_trigger(0)
        mvm.trigger_code()

        # Expected result: matrix @ vector
        expected = matrix @ input_vector

        xp.testing.assert_allclose(mvm.output.value, expected, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_mvm_wrong_input_size(self, target_device_idx, xp):
        """Test that MVM raises error for wrong input vector size"""
        # Create a 3x4 matrix (expects 4-element input vector)
        matrix = xp.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12]
        ], dtype=xp.float64)

        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Create input vector with wrong size (5 instead of 4)
        input_vector = xp.array([1, 2, 3, 4, 5], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)

        # Should raise ValueError during setup
        with self.assertRaises(ValueError) as cm:
            mvm.setup()

        self.assertIn("Input vector size mismatch", str(cm.exception))
        self.assertIn("got 5, expected 4", str(cm.exception))

    @cpu_and_gpu
    def test_mvm_null_recmat(self, target_device_idx, xp):
        """Test MVM behavior with null reconstruction matrix"""
        # Create recmat with None matrix
        # Create a valid matrix first, then set it to None after
        matrix = xp.eye(3)
        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)
        
        # Now set the internal matrix to None to test the warning
        mvm.recmat.recmat = None

        # This should work during initialization but skip during trigger_code
        input_vector = xp.array([1, 2, 3], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)

        # Should not raise during trigger_code, just print warning
        mvm.trigger_code()

        # Output should remain zeros (initial allocation)
        self.assertTrue(xp.allclose(mvm.output.value, 0))

    @cpu_and_gpu
    def test_mvm_no_input_provided(self, target_device_idx, xp):
        """Test that MVM raises error when no input is provided"""
        matrix = xp.eye(3)
        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Don't set any input

        # Should raise ValueError during setup
        with self.assertRaises(ValueError) as cm:
            mvm.setup()

        self.assertIn("InputValue is empty and not optional", str(cm.exception))

    @cpu_and_gpu
    def test_mvm_none_recmat_initialization(self, target_device_idx, xp):
        """Test that MVM raises error when recmat is None during initialization"""

        # Should raise ValueError during initialization
        with self.assertRaises(ValueError) as cm:
            MVM(recmat=None, target_device_idx=target_device_idx)

        self.assertIn("recmat must be provided!", str(cm.exception))

    @cpu_and_gpu
    def test_mvm_square_matrix(self, target_device_idx, xp):
        """Test MVM with square matrix (identity case)"""
        # Identity matrix
        matrix = xp.eye(4)
        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Input vector
        input_vector = xp.array([1, 2, 3, 4], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()
        mvm.prepare_trigger(0)
        mvm.trigger_code()

        # With identity matrix, output should equal input
        xp.testing.assert_allclose(mvm.output.value, input_vector, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_mvm_rectangular_matrix_tall(self, target_device_idx, xp):
        """Test MVM with tall rectangular matrix (more rows than columns)"""
        # 5x3 matrix
        matrix = xp.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1]
        ], dtype=xp.float64)

        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Input vector (3 elements)
        input_vector = xp.array([2, 3, 4], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()
        mvm.prepare_trigger(0)
        mvm.trigger_code()

        # Expected output (5 elements)
        expected = matrix @ input_vector  # [2, 3, 4, 5, 7]

        self.assertEqual(len(mvm.output.value), 5)
        xp.testing.assert_allclose(mvm.output.value, expected, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_mvm_rectangular_matrix_wide(self, target_device_idx, xp):
        """Test MVM with wide rectangular matrix (more columns than rows)"""
        # 2x5 matrix
        matrix = xp.array([
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10]
        ], dtype=xp.float64)

        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Input vector (5 elements)
        input_vector = xp.array([1, 1, 1, 1, 1], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()
        mvm.prepare_trigger(0)
        mvm.trigger_code()

        # Expected output (2 elements)
        expected = matrix @ input_vector  # [15, 40]

        self.assertEqual(len(mvm.output.value), 2)
        xp.testing.assert_allclose(mvm.output.value, expected, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_mvm_vector_update(self, target_device_idx, xp):
        """Test that MVM correctly updates when input vector changes"""
        matrix = xp.array([
            [2, 0],
            [0, 3]
        ], dtype=xp.float64)

        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # First input vector
        input_vector1 = xp.array([1, 2], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector1,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()
        mvm.prepare_trigger(0)
        mvm.trigger_code()

        expected1 = matrix @ input_vector1  # [2, 6]
        xp.testing.assert_allclose(mvm.output.value, expected1, rtol=1e-10, atol=1e-12)

        # Update input vector
        input_vector2 = xp.array([3, 4], dtype=xp.float64)
        input_value.value[:] = input_vector2

        mvm.prepare_trigger(1)
        mvm.trigger_code()

        expected2 = matrix @ input_vector2  # [6, 12]
        xp.testing.assert_allclose(mvm.output.value, expected2, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_mvm_generation_time(self, target_device_idx, xp):
        """Test that MVM correctly sets generation_time"""
        matrix = xp.eye(2)
        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        input_vector = xp.array([1, 2], dtype=xp.float64)
        input_value = BaseValue('test input', value=input_vector,
                                target_device_idx=target_device_idx)

        mvm.inputs['in_vector'].set(input_value)
        mvm.setup()

        # Test different time values
        for t in [5, 10, 15]:
            mvm.prepare_trigger(t)
            mvm.trigger_code()
            self.assertEqual(mvm.output.generation_time, mvm.current_time)

    @cpu_and_gpu
    def test_mvm_output_dimensions(self, target_device_idx, xp):
        """Test that MVM output has correct dimensions for various matrix sizes"""
        test_cases = [
            (3, 4),  # 3x4 matrix
            (5, 2),  # 5x2 matrix
            (1, 10), # 1x10 matrix (row vector)
            (10, 1), # 10x1 matrix (column vector)
        ]

        for n_rows, n_cols in test_cases:
            with self.subTest(matrix_shape=(n_rows, n_cols)):
                matrix = xp.ones((n_rows, n_cols))
                recmat = Recmat(matrix, target_device_idx=target_device_idx)
                mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

                input_vector = xp.ones(n_cols)
                input_value = BaseValue('test input', value=input_vector,
                                        target_device_idx=target_device_idx)

                mvm.inputs['in_vector'].set(input_value)
                mvm.setup()
                mvm.prepare_trigger(0)
                mvm.trigger_code()

                # Output should have n_rows elements
                self.assertEqual(len(mvm.output.value), n_rows)
                self.assertEqual(mvm.output.value.shape, (n_rows,))