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
        """
        self.n_modes = 5
        self.n_slopes_per_wfs = 2

        # 1. Create Mock Reconstruction Matrices for a 2-sensor system
        mat_both = xp.full((self.n_modes, 4), 1.0, dtype=xp.float32)
        mat_s1 = xp.full((self.n_modes, 2), 2.0, dtype=xp.float32)
        mat_s2 = xp.full((self.n_modes, 2), 3.0, dtype=xp.float32)

        # Add a zero row to s1 to simulate a lost mode (perfectly valid physically)
        mat_s1[4, :] = 0.0

        # Dictionary with arbitrary string keys (simulating YAML _dict_ref behavior)
        recmat_dict = {
            'rec_both': Recmat(mat_both, target_device_idx=target_device_idx),
            'rec_s1': Recmat(mat_s1, target_device_idx=target_device_idx),
            'rec_s2': Recmat(mat_s2, target_device_idx=target_device_idx)
        }

        # Explicit masks in the same order
        validity_masks = [
            [True, True],
            [True, False],
            [False, True]
        ]

        # 2. Initialize the Reconstructor
        rec = ModalrecMultirate(
            recmat_dict=recmat_dict,
            validity_masks=validity_masks,
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
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)
        rec.setup()

        return rec, slopes_s1, slopes_s2

    @cpu_and_gpu
    def test_both_sensors_valid(self, target_device_idx, xp):
        """Test Case 1: Both sensors have fresh data"""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 1.0
        s1.generation_time = current_time
        s2.generation_time = current_time

        rec.check_ready(current_time)
        rec.trigger_code()

        out = cpuArray(rec.out_modes.value)
        np.testing.assert_allclose(out, 60.0)
        self.assertEqual(rec.out_modes.generation_time, current_time)

    @cpu_and_gpu
    def test_single_sensor_valid(self, target_device_idx, xp):
        """Test Case 2: Only Sensor 1 has fresh data (Multirate Asynchronous)"""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 2.0
        s1.generation_time = current_time
        s2.generation_time = 1.0

        rec.check_ready(current_time)
        rec.trigger_code()

        out = cpuArray(rec.out_modes.value)
        # Mode 4 is intentionally set to 0.0 in mat_s1 inside _setup_reconstructor
        expected = np.array([40.0, 40.0, 40.0, 40.0, 0.0])
        np.testing.assert_allclose(out, expected)

    @cpu_and_gpu
    def test_zero_stuffing_no_sensors_valid(self, target_device_idx, xp):
        """Test Case 3: No sensors are valid. Verifies ZERO-STUFFING."""
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 3.0
        s1.generation_time = 2.0
        s2.generation_time = 1.0

        rec.check_ready(current_time)
        rec.trigger_code()

        out = cpuArray(rec.out_modes.value)
        np.testing.assert_allclose(out, 0.0)

    @cpu_and_gpu
    def test_sanity_check_dimensions(self, target_device_idx, xp):
        """Test that matrix row dimensions must match n_modes_total"""
        # Matrix with 4 rows, but n_modes_total is 5
        mat_wrong = xp.full((4, 4), 1.0, dtype=xp.float32)
        recmat_dict = {'rec_both': Recmat(mat_wrong, target_device_idx=target_device_idx)}

        with self.assertRaisesRegex(ValueError, "n_modes_total"):
            ModalrecMultirate(recmat_dict=recmat_dict, validity_masks=[[True, True]], 
                              n_modes_total=5, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_sanity_check_observability(self, target_device_idx, xp):
        """Test that dropping sensors cannot magically increase observability (fewer zero rows)"""

        # All-True matrix has 2 unobservable modes (rows of zeros)
        mat_both = xp.full((5, 4), 1.0, dtype=xp.float32)
        mat_both[3:, :] = 0.0

        # Single sensor matrix has 0 unobservable modes (Physically impossible!)
        mat_s1 = xp.full((5, 2), 2.0, dtype=xp.float32)

        recmat_dict = {
            'rec_both': Recmat(mat_both, target_device_idx=target_device_idx),
            'rec_s1': Recmat(mat_s1, target_device_idx=target_device_idx),
        }
        validity_masks = [[True, True], [True, False]]

        # The class should throw a ValueError because mat_s1 has fewer zero rows than mat_both
        with self.assertRaisesRegex(ValueError, "Logical inconsistency"):
            ModalrecMultirate(recmat_dict=recmat_dict, validity_masks=validity_masks,
                              n_modes_total=5, target_device_idx=target_device_idx)
