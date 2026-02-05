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
    def test_modalrec_implicit_polc_with_slicing(self, target_device_idx, xp):
        """Test implicit POLC with input_modes_slice when commands are larger than needed"""

        n_slopes = 10
        n_modes = 4  # Matrices work with 4 modes
        n_input_commands = 10  # But commands come from a larger vector (e.g., multiple DMs)

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        projmat_arr = xp.eye(n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Use only modes 2:6 from the input commands vector
        rec = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            in_commands_size=n_input_commands,
            input_modes_slice=slice(2, 6),  # Take 4 modes starting from index 2
            target_device_idx=target_device_idx
        )

        slopes = Slopes(slopes=xp.random.randn(n_slopes), target_device_idx=target_device_idx)
        # Pass the full commands vector (10 elements)
        full_commands = xp.random.randn(n_input_commands)
        commands = BaseValue('commands', value=full_commands,
                             target_device_idx=target_device_idx)

        slopes.generation_time = 0
        commands.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)

        rec.setup()

        # Check that internal commands array has the correct size (n_input_commands)
        self.assertEqual(rec.commands.shape[0], n_input_commands)

        rec.prepare_trigger(0)

        # After prepare_trigger, rec.commands should contain commands.value
        xp.testing.assert_array_equal(rec.commands, commands.value)

        rec.trigger_code()

        # Should produce valid output with n_modes size
        self.assertEqual(rec.modes.value.shape[0], n_modes)

        # Verify that the computation used the correct slice of commands
        # Manual computation: modes = C @ slopes - H @ commands[2:6]
        expected_modes = (rec.comm_mat.recmat @ rec.slopes -
                         rec.h_mat.recmat @ full_commands[2:6])
        xp.testing.assert_allclose(rec.modes.value, expected_modes, rtol=1e-10)

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

    @cpu_and_gpu
    def test_modalrec_output_slice(self, target_device_idx, xp):
        """Test output_slice extracts correct subset of modes"""

        n_slopes = 10
        n_modes = 8

        # Create simple matrices for testing
        recmat_arr = xp.eye(n_modes, n_slopes)  # Identity-like for easy verification
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # Extract modes 2:6 (4 modes)
        rec = Modalrec(
            recmat=recmat,
            output_slice=[2, 6, 1],  # [start, stop, step]
            target_device_idx=target_device_idx
        )

        # Create slopes with known values
        slopes_values = xp.arange(n_slopes, dtype=xp.float32)
        slopes = Slopes(slopes=slopes_values, target_device_idx=target_device_idx)
        slopes.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        # Should output 4 modes (indices 2,3,4,5)
        self.assertEqual(rec.modes.value.shape[0], 4)

        # Compute expected output: full reconstruction then slice
        full_output = recmat_arr @ slopes_values
        expected_output = full_output[2:6]

        xp.testing.assert_allclose(rec.modes.value, expected_output, rtol=1e-10)

    @cpu_and_gpu
    def test_modalrec_output_slice_with_step(self, target_device_idx, xp):
        """Test output_slice with non-unit step"""

        n_slopes = 12
        n_modes = 10

        recmat_arr = xp.eye(n_modes, n_slopes)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # Extract every other mode: indices 1,3,5,7,9 (5 modes)
        rec = Modalrec(
            recmat=recmat,
            output_slice=[1, 10, 2],  # start=1, stop=10, step=2
            target_device_idx=target_device_idx
        )

        slopes_values = xp.arange(n_slopes, dtype=xp.float32)
        slopes = Slopes(slopes=slopes_values, target_device_idx=target_device_idx)
        slopes.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        self.assertEqual(rec.modes.value.shape[0], 5)

        full_output = recmat_arr @ slopes_values
        expected_output = full_output[1:10:2]

        xp.testing.assert_allclose(rec.modes.value, expected_output, rtol=1e-10)

    @cpu_and_gpu
    def test_modalrec_input_modes_index(self, target_device_idx, xp):
        """Test input_modes_index selects specific command indices"""

        n_slopes = 6
        n_modes = 4
        n_input_commands = 10

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        projmat_arr = xp.eye(n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Select specific indices: [1, 3, 5, 8] from 10-element command vector
        indices_to_use = [1, 3, 5, 8]

        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            in_commands_size=n_input_commands,
            input_modes_index=indices_to_use,
            target_device_idx=target_device_idx
        )

        slopes_values = xp.random.randn(n_slopes)
        slopes = Slopes(slopes=slopes_values, target_device_idx=target_device_idx)
        slopes.generation_time = 0

        full_commands = xp.random.randn(n_input_commands)
        commands = BaseValue('commands', value=full_commands,
                            target_device_idx=target_device_idx)
        commands.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        # Verify computation used only selected indices
        selected_commands = full_commands[indices_to_use]
        expected_comm_slopes = intmat_arr @ selected_commands
        expected_pol_modes = recmat_arr @ (slopes_values + expected_comm_slopes)
        expected_output = projmat_arr @ expected_pol_modes - selected_commands

        xp.testing.assert_allclose(rec.modes.value, expected_output, rtol=1e-9)

    @cpu_and_gpu
    def test_modalrec_polc_combined_slicing(self, target_device_idx, xp):
        """Test combining input_modes_slice and output_slice"""

        n_slopes = 8
        n_modes = 6
        n_input_commands = 12

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        projmat_arr = xp.eye(n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Input: use commands[2:8] (6 commands)
        # Output: return modes[1:5] (4 modes)
        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            in_commands_size=n_input_commands,
            input_modes_slice=slice(2, 8),
            output_slice=[1, 5, 1],
            target_device_idx=target_device_idx
        )

        slopes_values = xp.random.randn(n_slopes)
        slopes = Slopes(slopes=slopes_values, target_device_idx=target_device_idx)
        slopes.generation_time = 0

        full_commands = xp.random.randn(n_input_commands)
        commands = BaseValue('commands', value=full_commands,
                            target_device_idx=target_device_idx)
        commands.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        # Output should have 4 modes
        self.assertEqual(rec.modes.value.shape[0], 4)

        # Manual computation
        selected_commands = full_commands[2:8]
        expected_comm_slopes = intmat_arr @ selected_commands
        expected_pol_modes = recmat_arr @ (slopes_values + expected_comm_slopes)
        expected_full_output = projmat_arr @ expected_pol_modes - selected_commands
        expected_output = expected_full_output[1:5]

        xp.testing.assert_allclose(rec.modes.value, expected_output, rtol=1e-9)

    @cpu_and_gpu
    def test_modalrec_input_modes_slice_multiple_ranges(self, target_device_idx, xp):
        """Test input_modes_slice with multiple slice ranges"""

        n_slopes = 8
        n_modes = 6  # Will use indices [0,1,2, 7,8,9] = 6 modes
        n_input_commands = 12

        intmat_arr = xp.random.randn(n_slopes, n_modes)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        projmat_arr = xp.eye(n_modes)

        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Select commands from multiple ranges: [0:3] and [7:10]
        # This creates indices [0,1,2,7,8,9]
        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            in_commands_size=n_input_commands,
            input_modes_slice=[[0, 3], [7, 10]],  # Multiple slice specs
            target_device_idx=target_device_idx
        )

        slopes_values = xp.random.randn(n_slopes)
        slopes = Slopes(slopes=slopes_values, target_device_idx=target_device_idx)
        slopes.generation_time = 0

        full_commands = xp.random.randn(n_input_commands)
        commands = BaseValue('commands', value=full_commands,
                            target_device_idx=target_device_idx)
        commands.generation_time = 0

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()

        # Verify correct indices were used
        expected_indices = list(range(0, 3)) + list(range(7, 10))
        selected_commands = full_commands[expected_indices]

        expected_comm_slopes = intmat_arr @ selected_commands
        expected_pol_modes = recmat_arr @ (slopes_values + expected_comm_slopes)
        expected_output = projmat_arr @ expected_pol_modes - selected_commands

        xp.testing.assert_allclose(rec.modes.value, expected_output, rtol=1e-9)
