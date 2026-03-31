import unittest

import specula
specula.init(0)  # Default target device

import numpy as np
from specula import cpuArray
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes
from specula.processing_objects.modalrec_multirate import ModalrecMultirate

from test.specula_testlib import cpu_and_gpu

class TestModalrecMultirate(unittest.TestCase):

    def _setup_reconstructor(self, target_device_idx, xp):
        """
        Helper method to build the reconstructor and mock data.
        Since we need target_device_idx and xp, we cannot do this in setUp().
        """
        self.n_modes = 5
        self.n_slopes_per_wfs = 2

        # 1. Create Mock Reconstruction Matrices for a 2-sensor system
        # (True, True) -> Both sensors. Input vector size: 4
        mat_both = xp.full((self.n_modes, 4), 1.0, dtype=xp.float32)

        # (True, False) -> Only Sensor 1. Input vector size: 2
        mat_s1 = xp.full((self.n_modes, 2), 2.0, dtype=xp.float32)

        # (False, True) -> Only Sensor 2. Input vector size: 2
        mat_s2 = xp.full((self.n_modes, 2), 3.0, dtype=xp.float32)

        recmat_dict = {
            (True, True): Recmat(mat_both, target_device_idx=target_device_idx),
            (True, False): Recmat(mat_s1, target_device_idx=target_device_idx),
            (False, True): Recmat(mat_s2, target_device_idx=target_device_idx)
        }

        # 2. Initialize the Reconstructor
        rec = ModalrecMultirate(
            recmat_dict=recmat_dict,
            n_modes_total=self.n_modes,
            target_device_idx=target_device_idx
        )

        # 3. Create Input Slopes objects
        slopes_s1 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)
        slopes_s2 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)

        # Set dummy slope values: S1 = [10, 10], S2 = [20, 20]
        slopes_s1.slopes[:] = 10.0
        slopes_s2.slopes[:] = 20.0

        # Connect inputs
        rec.inputs['in_slopes_list'].set([slopes_s1, slopes_s2])
        # Specula framework requires to move inputs to local_inputs for unit tests
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)

        rec.setup()

        return rec, slopes_s1, slopes_s2

    @cpu_and_gpu
    def test_both_sensors_valid(self, target_device_idx, xp):
        """Test Case 1: Both sensors have fresh data"""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 1.0
        # Simulate both sensors providing new data exactly now
        s1.generation_time = current_time
        s2.generation_time = current_time

        rec.check_ready(current_time) # Internally sets rec.current_time
        rec.trigger_code()

        # Expected: Uses mat_both (all 1.0s).
        # Input vector is [10, 10, 20, 20].
        # Output should be 1*10 + 1*10 + 1*20 + 1*20 = 60.0 for all modes
        out = cpuArray(rec.out_modes.value)
        np.testing.assert_allclose(out, 60.0)
        self.assertEqual(rec.out_modes.generation_time, current_time)

    @cpu_and_gpu
    def test_single_sensor_valid(self, target_device_idx, xp):
        """Test Case 2: Only Sensor 1 has fresh data (Multirate Asynchronous)"""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 2.0
        # Sensor 1 is fresh, Sensor 2 is OLD (from previous frame)
        s1.generation_time = current_time
        s2.generation_time = 1.0

        rec.check_ready(current_time)
        rec.trigger_code()

        # Expected: The dynamic scheduler drops Sensor 2 and uses mat_s1 (all 2.0s).
        # Input vector is just [10, 10].
        # Output should be 2*10 + 2*10 = 40.0 for all modes
        out = cpuArray(rec.out_modes.value)
        np.testing.assert_allclose(out, 40.0)
        self.assertEqual(rec.out_modes.generation_time, current_time)

    @cpu_and_gpu
    def test_zero_stuffing_no_sensors_valid(self, target_device_idx, xp):
        """Test Case 3: No sensors are valid. Verifies ZERO-STUFFING."""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 3.0
        # Both sensors have old data
        s1.generation_time = 2.0
        s2.generation_time = 1.0

        rec.check_ready(current_time)
        rec.trigger_code()

        # Expected: Zero-stuffing kicks in. Output must be perfectly 0.0.
        out = cpuArray(rec.out_modes.value)
        np.testing.assert_allclose(out, 0.0)
        self.assertEqual(rec.out_modes.generation_time, current_time)

    @cpu_and_gpu
    def test_missing_dictionary_key_raises_error(self, target_device_idx, xp):
        """Test Case 4: Exception raised if LUT is missing a configuration"""
        # Create dict without (False, True)
        mat_both = xp.full((5, 4), 1.0, dtype=xp.float32)
        recmat_dict = {
            (True, True): Recmat(mat_both, target_device_idx=target_device_idx)
        }

        rec = ModalrecMultirate(recmat_dict=recmat_dict, n_modes_total=5, 
                                target_device_idx=target_device_idx)
        s1 = Slopes(length=2, target_device_idx=target_device_idx)
        s2 = Slopes(length=2, target_device_idx=target_device_idx)

        rec.inputs['in_slopes_list'].set([s1, s2])
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)
        rec.setup()

        # Trigger only S2 -> should map to (False, True) which we omitted
        current_time = 1.0
        s1.generation_time = 0.0
        s2.generation_time = current_time

        rec.check_ready(current_time)
        with self.assertRaises(KeyError):
            rec.trigger_code()
