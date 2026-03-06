import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.modalrec import Modalrec
from specula.processing_objects.modalrec_implicit_polc import ModalrecImplicitPolc
from specula.data_objects.recmat import Recmat
from specula.data_objects.intmat import Intmat
from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu

import gc
import tracemalloc

class TestModalrec(unittest.TestCase):

    @cpu_and_gpu
    def test_modalrec_wrong_size(self, target_device_idx, xp):

        recmat = Recmat(xp.arange(12).reshape((3,4)), target_device_idx=target_device_idx)
        rec = Modalrec(recmat=recmat, target_device_idx=target_device_idx)

        slopes = Slopes(slopes=xp.arange(5), target_device_idx=target_device_idx)
        rec.inputs['in_slopes'].set(slopes)

        t = 1
        slopes.generation_time = t
        rec.setup()
        rec.prepare_trigger(t)
        with self.assertRaises(ValueError):
            rec.trigger_code()

    @cpu_and_gpu
    def test_modalrec_vs_implicit_polc(self, target_device_idx, xp):

        # intmat (shape 6x4)
        intmat_arr = xp.array([
                            [1, 0,  1,  1],
                            [0, 1, -1,  1],
                            [1, 0, -1,  1],
                            [0, 1,  1, -1],
                            [1, 0,  1, -1],
                            [0, 1, -1, -1]
                        ])
        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)

        # recmat: pseudo-inverse or intmat (shape 4x6)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # projmat: 2x4 with a diagonal of 2
        projmat_arr = xp.eye(4) * 2
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # slopes:
        slopes_list = [3,  1.5,  3,  -0.5,  1,  1.5]
        slopes    = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)
        slopes_ip = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)
        slopes.generation_time = 0
        slopes_ip.generation_time = 0

        # commands:
        commands_list = [0.1, 0.2, 0.3, 0.4]
        commands    = BaseValue('commands', value=xp.array(commands_list),
                                target_device_idx=target_device_idx)
        commands_ip = BaseValue('commands', value=xp.array(commands_list),
                                target_device_idx=target_device_idx)
        commands.generation_time = 0
        commands_ip.generation_time = 0

        # Modalrec standard (POLC)
        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            target_device_idx=target_device_idx
        )
        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()
        out1 = rec.modes.value.copy()

        # ModalrecImplicitPolc
        rec2 = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            target_device_idx=target_device_idx
        )
        rec2.inputs['in_slopes'].set(slopes_ip)
        rec2.inputs['in_commands'].set(commands_ip)
        rec2.setup()
        rec2.prepare_trigger(0)
        rec2.trigger_code()
        out2 = rec2.modes.value.copy()

        xp.testing.assert_allclose(out1, out2, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_modalrec_polc_wrong_size(self, target_device_idx, xp):

        # intmat which expects 4 commands and produces 6 slopes
        intmat_arr = xp.array([
                            [1, 0,  1,  1],
                            [0, 1, -1,  1],
                            [1, 0, -1,  1],
                            [0, 1,  1, -1],
                            [1, 0,  1, -1],
                            [0, 1, -1, -1]
                        ])
        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)

        # recmat: pseudo-inverse of intmat (shape 4x6)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # projmat: 2x4 with a diagonal of 2
        projmat_arr = xp.eye(4) * 2
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Create a Modalrec which expects 6 slopes and 4 commands
        rec = Modalrec(
            nmodes=4,
            recmat=recmat,
            intmat=intmat,
            projmat=projmat,
            polc=True,
            target_device_idx=target_device_idx
        )

        # Slopes with wrong size (5 instead of 6)
        slopes = Slopes(slopes=xp.arange(5), target_device_idx=target_device_idx)
        commands = BaseValue('commands', value=xp.array([0.1, 0.2, 0.3, 0.4]),
                             target_device_idx=target_device_idx)

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)

        t = 1
        slopes.generation_time = t
        commands.generation_time = t

        rec.setup()

        # We expect a ValueError during prepare_trigger due to size mismatch
        with self.assertRaises(ValueError) as cm:
            rec.prepare_trigger(t)

        # Verify that the error message is as expected
        self.assertIn("Dimension mismatch in POLC mode", str(cm.exception))
        self.assertIn("intmat @ commands will produce 6 slopes", str(cm.exception))
        self.assertIn("but input slopes has size 5", str(cm.exception))

    @cpu_and_gpu
    def test_modalrec_implicit_polc_memory_cleanup(self, target_device_idx, xp):
        """Test that matrices are properly deleted to free memory"""

        # Start memory tracking BEFORE creating any matrices
        if target_device_idx == -1:
            tracemalloc.start()
            gc.collect()

        # Create test matrices with correct dimensions
        n_slopes = 1000
        n_modes = 100

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.random.randn(n_modes, n_slopes)
        projmat_arr = xp.random.randn(n_modes, n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Track memory for GPU only
        if target_device_idx != -1: # pragma: no cover
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
            mem_before = mempool.used_bytes()
        else:
            mem_before = tracemalloc.get_traced_memory()[0]

        # Create ModalrecImplicitPolc - should delete recmat, projmat, intmat internals
        rec = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            target_device_idx=target_device_idx
        )

        del intmat_arr
        del projmat_arr
        del recmat_arr
        del intmat
        del projmat
        del recmat

        # Check that original matrices were deleted
        self.assertIsNone(rec.recmat)
        self.assertIsNone(rec.projmat)
        self.assertIsNone(rec.intmat)

        # Check that comm_mat and h_mat exist
        self.assertIsNotNone(rec.comm_mat)
        self.assertIsNotNone(rec.h_mat)

        # Verify shapes
        # comm_mat = projmat @ recmat = (n_modes, n_modes) @ (n_modes, n_slopes)
        #          = (n_modes, n_slopes)
        self.assertEqual(rec.comm_mat.recmat.shape, (n_modes, n_slopes))
        # h_mat = I - comm_mat @ intmat = (n_modes, n_modes)
        self.assertEqual(rec.h_mat.recmat.shape, (n_modes, n_modes))

        # Memory tracking
        if target_device_idx != -1: # pragma: no cover
            gc.collect()
            mempool.free_all_blocks()
            mem_after = mempool.used_bytes()
        else:
            gc.collect()
            mem_after = tracemalloc.get_traced_memory()[0]
            tracemalloc.stop()

        verbose = False
        if verbose: # pragma: no cover
            print(f"{'GPU' if target_device_idx != -1 else 'CPU'} Memory before:"
                f" {mem_before} bytes, after: {mem_after} bytes")

        bytes_per_element = 4  # float32

        # Original matrices total: recmat + projmat + intmat
        original_mem = (n_modes * n_slopes + n_modes * n_modes + n_slopes * n_modes) \
                        * bytes_per_element
        # New matrices total: comm_mat + h_mat
        new_mem = (n_modes * n_slopes + n_modes * n_modes) * bytes_per_element
        # Expected savings: intmat memory
        expected_savings = (n_slopes * n_modes) * bytes_per_element

        if verbose: # pragma: no cover
            print(f"Original matrices size: {original_mem} bytes")
            print(f"New matrices size: {new_mem} bytes")
            print(f"Expected net savings: {expected_savings} bytes")
            print(f"Actual memory change: {mem_after - mem_before:+} bytes")

        # Memory should not increase significantly
        # Allow more tolerance for CPU due to Python memory management
        tolerance_factor = 2.0 if target_device_idx == -1 else 1.5
        max_increase = new_mem * tolerance_factor

        self.assertLess(mem_after - mem_before, max_increase,
                       f"Memory increased by {mem_after - mem_before} bytes, "
                       f"expected less than {max_increase} bytes")


    @cpu_and_gpu
    def test_modalrec_implicit_polc_no_commands_first_step(self, target_device_idx, xp):
        """Test that implicit POLC works with no commands on first step"""

        intmat_arr = xp.array([[1, 0], [0, 1], [1, 1]])
        recmat_arr = xp.linalg.pinv(intmat_arr)
        projmat_arr = xp.eye(2)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        rec = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            target_device_idx=target_device_idx
        )

        slopes = Slopes(slopes=xp.array([1.0, 2.0, 3.0]),
                        target_device_idx=target_device_idx)
        slopes.generation_time = 0

        # Add commands input with None value to simulate first step
        commands = BaseValue('commands', value=None, target_device_idx=target_device_idx)
        commands.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)

        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        # Should not crash and produce valid output
        self.assertEqual(rec.modes.value.shape[0], 2)
        self.assertIsNotNone(rec.modes.value)

    @cpu_and_gpu
    def test_modalrec_implicit_polc_matrix_shapes(self, target_device_idx, xp):
        """Verify that comm_mat and h_mat have correct shapes"""

        n_slopes = 100
        n_modes = 50

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.random.randn(n_modes, n_slopes)
        projmat_arr = xp.eye(n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        rec = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            target_device_idx=target_device_idx
        )

        # comm_mat should be (n_modes, n_slopes)
        self.assertEqual(rec.comm_mat.recmat.shape, (n_modes, n_slopes))

        # h_mat should be (n_modes, n_modes)
        self.assertEqual(rec.h_mat.recmat.shape, (n_modes, n_modes))
