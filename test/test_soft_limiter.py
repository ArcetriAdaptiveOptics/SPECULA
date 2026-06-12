import unittest
import numpy as np

import specula
specula.init(0)  # Default target device

from specula import cpuArray
from specula.base_value import BaseValue
from specula.data_objects.recmat import Recmat
from specula.processing_objects.soft_limiter import SoftLimiter
from test.specula_testlib import cpu_and_gpu

class TestSoftLimiter(unittest.TestCase):

    def _get_dummy_recmats(self, n_modes, n_petals_rel, target_device_idx, xp):
        """
        Generates two dummy matrices (Recmat and Intmat) for testing purposes.
        We use partial identity matrices so the linear algebra is trivial to verify.
        """
        # recmat: (5, 10). Maps the first 5 continuous modes directly to the 5 relative petals
        recmat_data = xp.zeros((n_petals_rel, n_modes), dtype=xp.float32)
        for i in range(n_petals_rel):
            recmat_data[i, i] = 1.0
  
        # intmat: (10, 5). Maps the 5 relative petals back onto the first 5 continuous modes
        intmat_data = xp.zeros((n_modes, n_petals_rel), dtype=xp.float32)
        for i in range(n_petals_rel):
            intmat_data[i, i] = 1.0

        recmat_obj = Recmat(recmat_data, target_device_idx=target_device_idx)
        intmat_obj = Recmat(intmat_data, target_device_idx=target_device_idx)

        return [recmat_obj, intmat_obj]

    @cpu_and_gpu
    def test_initialization(self, target_device_idx, xp):
        """Tests that the object initializes correctly only with a valid list of Recmat objects."""
        n_modes, n_petals_rel = 10, 5
        recmat_list = self._get_dummy_recmats(n_modes, n_petals_rel, target_device_idx, xp)

        # Should initialize without errors
        unwrapper = SoftLimiter(
            recmat_list=recmat_list,
            target_device_idx=target_device_idx
        )
        self.assertEqual(unwrapper.recmat.shape, (5, 10))
        self.assertEqual(unwrapper.intmat.shape, (10, 5))

        # Should raise an exception if the required matrices are missing
        with self.assertRaises(ValueError):
            SoftLimiter(recmat_list=[recmat_list[0]], target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_trigger_math_correctness(self, target_device_idx, xp):
        """Tests the pure algebra: extraction and subtraction with gain=1.0."""
        n_modes, n_petals_rel = 10, 5
        recmat_list = self._get_dummy_recmats(n_modes, n_petals_rel, target_device_idx, xp)

        unwrapper = SoftLimiter(
            recmat_list=recmat_list,
            gain=1.0,
            target_device_idx=target_device_idx
        )

        # Create an in_comm vector of 10 elements: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # The first 5 represent the differential piston error, the last 5 are true atmospheric modes.
        in_comm_data = xp.arange(10, 110, 10, dtype=xp.float32)
        in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)

        unwrapper.inputs['in_comm'].set(in_comm)
        unwrapper.setup()

        in_comm.generation_time = 1
        unwrapper.check_ready(1)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_comm = cpuArray(unwrapper.outputs['out_comm'].value)
        out_ost = cpuArray(unwrapper.outputs['out_ost'].value)

        # With gain 1.0, the Unwrapper must perfectly subtract the first 5 values and leave the rest untouched
        expected_out_comm = np.array([0, 0, 0, 0, 0, 60, 70, 80, 90, 100], dtype=np.float32)
        expected_out_ost = np.array([10, 20, 30, 40, 50, 0, 0, 0, 0, 0], dtype=np.float32)

        np.testing.assert_allclose(out_comm, expected_out_comm, rtol=1e-5)
        np.testing.assert_allclose(out_ost, expected_out_ost, rtol=1e-5)

    @cpu_and_gpu
    def test_trigger_gain_scaling(self, target_device_idx, xp):
        """Tests that the 'gain' parameter (soft nudging) scales the correction correctly."""
        n_modes, n_petals_rel = 10, 5
        recmat_list = self._get_dummy_recmats(n_modes, n_petals_rel, target_device_idx, xp)

        # Set gain to 0.1 (removes only 10% of the estimated piston error per frame)
        unwrapper = SoftLimiter(
            recmat_list=recmat_list,
            gain=0.1,
            target_device_idx=target_device_idx
        )

        in_comm_data = xp.array([100.0] * 10, dtype=xp.float32)
        in_comm = BaseValue(value=in_comm_data, target_device_idx=target_device_idx)
        unwrapper.inputs['in_comm'].set(in_comm)
        unwrapper.setup()

        in_comm.generation_time = 1
        unwrapper.check_ready(1)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_comm = cpuArray(unwrapper.outputs['out_comm'].value)
        out_ost = cpuArray(unwrapper.outputs['out_ost'].value)

        # The error read on the first 5 modes is 100. The gain is 0.1. The correction must be 10.0.
        # out_comm must be 100 - 10 = 90 for the first 5 modes, and 100 for the remaining 5.
        expected_out_comm = np.array([90]*5 + [100]*5, dtype=np.float32)
        expected_out_ost = np.array([10]*5 + [0]*5, dtype=np.float32)

        np.testing.assert_allclose(out_comm, expected_out_comm, rtol=1e-5)
        np.testing.assert_allclose(out_ost, expected_out_ost, rtol=1e-5)

    @cpu_and_gpu
    def test_time_intervals(self, target_device_idx, xp):
        """Tests that the filter correctly respects start_time and interval_time."""
        n_modes, n_petals_rel = 10, 5
        recmat_list = self._get_dummy_recmats(n_modes, n_petals_rel, target_device_idx, xp)

        unwrapper = SoftLimiter(
            recmat_list=recmat_list,
            start_time=2.0,
            interval_time=2.0,
            target_device_idx=target_device_idx
        )

        unwrapper.start_time_t = 2
        unwrapper.interval_time_t = 2

        in_comm_data = xp.array([100.0] * 10, dtype=xp.float32)

        # Initial setup with a dummy value to validate the input ports
        dummy_val = BaseValue(value=in_comm_data.copy(), target_device_idx=target_device_idx)
        unwrapper.inputs['in_comm'].set(dummy_val)
        unwrapper.setup()

        # --- Frame 1 (t=1) ---
        # Before start_time: bypass (out_comm == in_comm, out_ost == 0)
        in_comm_1 = BaseValue(value=in_comm_data.copy(), target_device_idx=target_device_idx)
        in_comm_1.generation_time = 1
        unwrapper.inputs['in_comm'].set(in_comm_1)
        unwrapper.check_ready(1)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_ost_t1 = cpuArray(unwrapper.outputs['out_ost'].value)
        np.testing.assert_allclose(out_ost_t1, np.zeros(10))

        # --- Frame 2 (t=2) ---
        # start_time reached: correction applied
        in_comm_2 = BaseValue(value=in_comm_data.copy(), target_device_idx=target_device_idx)
        in_comm_2.generation_time = 2
        unwrapper.inputs['in_comm'].set(in_comm_2)
        unwrapper.check_ready(2)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_ost_t2 = cpuArray(unwrapper.outputs['out_ost'].value)
        self.assertEqual(out_ost_t2[0], 100.0) # Correction applied on the first mode

        # --- Frame 3 (t=3) ---
        # The unwrapper should bypass again since the interval_time of 2s has not yet elapsed
        in_comm_3 = BaseValue(value=in_comm_data.copy(), target_device_idx=target_device_idx)
        in_comm_3.generation_time = 3
        unwrapper.inputs['in_comm'].set(in_comm_3)
        unwrapper.check_ready(3)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_ost_t3 = cpuArray(unwrapper.outputs['out_ost'].value)
        np.testing.assert_allclose(out_ost_t3, np.zeros(10))

        # --- Frame 4 (t=4) ---
        # interval_time elapsed: correction applied again
        in_comm_4 = BaseValue(value=in_comm_data.copy(), target_device_idx=target_device_idx)
        in_comm_4.generation_time = 4
        unwrapper.inputs['in_comm'].set(in_comm_4)
        unwrapper.check_ready(4)
        unwrapper.trigger()
        unwrapper.post_trigger()

        out_ost_t4 = cpuArray(unwrapper.outputs['out_ost'].value)
        self.assertEqual(out_ost_t4[0], 100.0)
