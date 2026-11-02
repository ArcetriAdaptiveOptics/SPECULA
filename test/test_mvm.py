import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.mvm import MVM
from specula.data_objects.recmat import Recmat
from iteras.data_objects.matrix import Matrix
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu
from specula.loop_control import LoopControl


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
        input_value.generation_time = 0

        # Set up and run
        mvm.inputs['in_vector'].set(input_value)

        loop = LoopControl()
        loop.add(mvm, idx=0)
        loop.run(run_time=1, dt=1)

        # Expected result: matrix @ vector
        expected = matrix @ input_vector

        xp.testing.assert_allclose(mvm.output.value, expected, rtol=1e-10, atol=1e-12)
        self.assertTrue(mvm.can_accelerate())

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
    def test_mvm_no_input_provided(self, target_device_idx, xp):
        """Test that MVM raises error when no input is provided"""
        matrix = xp.eye(3)
        recmat = Recmat(matrix, target_device_idx=target_device_idx)
        mvm = MVM(recmat=recmat, target_device_idx=target_device_idx)

        # Don't set any input

        # Should raise ValueError during setup
        with self.assertRaises(ValueError) as cm:
            mvm.setup()

        self.assertIn("Input in_vector of object MVM is empty and not optional", str(cm.exception))

    @cpu_and_gpu
    def test_mvm_none_recmat_initialization(self, target_device_idx, xp):
        """Test that MVM raises error when recmat is None during initialization"""

        # Should raise ValueError during initialization
        with self.assertRaises(ValueError) as cm:
            MVM(recmat=None, target_device_idx=target_device_idx)

        self.assertIn("matrix must be provided!", str(cm.exception))

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

    @cpu_and_gpu
    def test_matrix_parameter(self, target_device_idx, xp):
        """matrix is the parameter name, recmat its previous name: only one can be given"""
        matrix = Matrix(xp.eye(3, dtype=xp.float32), target_device_idx=target_device_idx)
        self.assertIs(MVM(matrix=matrix, target_device_idx=target_device_idx).matrix, matrix)
        self.assertIs(MVM(recmat=matrix, target_device_idx=target_device_idx).matrix, matrix)
        with self.assertRaisesRegex(ValueError, 'Only one'):
            MVM(matrix=matrix, recmat=matrix, target_device_idx=target_device_idx)


class TestMVMParameterFiles(unittest.TestCase):

    def test_matrix_object_and_recmat_object(self):
        """
        The matrix is restored from the calibration files with matrix_object,
        or recmat_object as before, from the rec/ subdirectory in SPECULA
        """
        import os
        import tempfile
        import numpy as np
        import yaml
        from iteras.simul import Simul
        from iteras.lib import storage
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, 'rec'))
            Recmat(np.array([[1, 0], [0, 2], [1, 1]], dtype=np.float32),
                   target_device_idx=-1).save(os.path.join(tmp, 'rec', 'm1.fits'))
            for parname in ('matrix_object', 'recmat_object'):
                with self.subTest(parname=parname):
                    store_dir = os.path.join(tmp, f'store_{parname}')
                    params = {
                        'main': {'class': 'SimParams', 'root_dir': tmp,
                                 'total_time': 0.002, 'time_step': 0.001},
                        'gen': {'class': 'WaveGenerator', 'constant': [1.0, 2.0],
                                'target_device_idx': -1},
                        'mvm': {'class': 'MVM', parname: 'm1', 'target_device_idx': -1,
                                'inputs': {'in_vector': 'gen.output'}},
                        'store': {'class': 'DataStore', 'store_dir': store_dir, 'create_tn': False,
                                  'data_format': 'npz',
                                  'inputs': {'input_list': ['out-mvm.out_vector']}},
                    }
                    yml = os.path.join(tmp, f'{parname}.yml')
                    with open(yml, 'w') as f:
                        yaml.dump(params, f)
                    Simul(yml, speed_report=False).run()
                    with storage.read_object(os.path.join(store_dir, 'out.npz')) as f:
                        np.testing.assert_allclose(f.read('data', 0)[-1], [1, 4, 3])


@unittest.skipIf(specula.cp is None, 'Cupy not found')
class TestMVMAccelerated(unittest.TestCase):
    '''The product uses graph_matmul.matmul(): MVM is captured in a CUDA graph'''

    def _run(self, accelerate, matrix_dtype, vector_dtype, precision):
        import numpy as np
        from iteras import cpuArray
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
        rng = np.random.default_rng(1)
        matrix = Matrix(rng.standard_normal((30, 70)).astype(matrix_dtype), target_device_idx=0,
                        precision=int(matrix_dtype == np.float32))
        vector = BaseValue(value=np.zeros(70, dtype=vector_dtype), target_device_idx=0,
                           precision=int(vector_dtype == np.float32))
        mvm = MVM(matrix=matrix, target_device_idx=0, precision=precision)
        mvm.inputs['in_vector'].set(vector)
        accelerator = None
        if accelerate:
            accelerator = Accelerator(AcceleratorOptions())
            accelerator.apply([mvm])
        mvm.setup()
        outputs = []
        for step in range(5):
            t = mvm.seconds_to_t(0.001 * step)
            vector.value[:] = specula.cp.asarray(rng.standard_normal(70))
            vector.generation_time = t
            mvm.check_ready(t)
            mvm.trigger()
            mvm.post_trigger()
            outputs.append(cpuArray(mvm.output.value).copy())
        return np.array(outputs), mvm

    def test_accelerated(self):
        '''Also with a matrix and an input in a different precision from the object:
        both are used in the precision of the object, with and without the accelerator'''
        import numpy as np
        for matrix_dtype, vector_dtype, precision in ((np.float32, np.float32, 1),
                                                      (np.float64, np.float32, 1),
                                                      (np.float32, np.float64, 1),
                                                      (np.float32, np.float32, 0),
                                                      (np.float64, np.float32, 0)):
            with self.subTest(matrix_dtype=matrix_dtype, vector_dtype=vector_dtype, precision=precision):
                ref, _ = self._run(False, matrix_dtype, vector_dtype, precision)
                with self.assertNoLogs('iteras.accelerator', level='WARNING'):
                    out, mvm = self._run(True, matrix_dtype, vector_dtype, precision)
                np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
                self.assertEqual(mvm._accel.status, 'accelerated')
