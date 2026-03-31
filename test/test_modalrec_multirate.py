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
        self.n_modes = 5
        self.n_slopes_per_wfs = 2

        # 1. Create Mock Reconstruction Matrices
        # Both sensors active -> 4 columns
        mat_both = xp.full((self.n_modes, 4), 1.0, dtype=xp.float32)

        # Single sensor active -> 2 columns
        mat_s1 = xp.full((self.n_modes, 2), 2.0, dtype=xp.float32)
        mat_s2 = xp.full((self.n_modes, 2), 3.0, dtype=xp.float32)

        # Simulate MMSE natural attenuation for unobservable modes (rows 3 and 4)
        mat_s1[3:, :] = 0.1
        mat_s2[3:, :] = 0.1

        recmat_dict = {
            'rec_both': Recmat(mat_both, target_device_idx=target_device_idx),
            'rec_s1': Recmat(mat_s1, target_device_idx=target_device_idx),
            'rec_s2': Recmat(mat_s2, target_device_idx=target_device_idx)
        }

        validity_masks = [
            [True, True],
            [True, False],
            [False, True]
        ]

        rec = ModalrecMultirate(
            recmat_dict=recmat_dict,
            validity_masks=validity_masks,
            n_modes_total=self.n_modes,
            target_device_idx=target_device_idx
        )

        slopes_s1 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)
        slopes_s2 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)

        slopes_s1.slopes[:] = 10.0
        slopes_s2.slopes[:] = 20.0

        rec.inputs['in_slopes_list'].set([slopes_s1, slopes_s2])
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)
        rec.setup()

        return rec, slopes_s1, slopes_s2

    @cpu_and_gpu
    def test_both_sensors_valid(self, target_device_idx, xp):
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 1.0
        s1.generation_time = current_time
        s2.generation_time = current_time

        rec.check_ready(current_time)
        rec.trigger_code()

        out = cpuArray(rec.out_modes.value)
        # 1.0 * 10 + 1.0 * 10 + 1.0 * 20 + 1.0 * 20 = 60.0
        np.testing.assert_allclose(out, 60.0)
        self.assertEqual(rec.out_modes.generation_time, current_time)

    @cpu_and_gpu
    def test_single_sensor_valid(self, target_device_idx, xp):
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 2.0
        s1.generation_time = current_time
        s2.generation_time = 1.0  # Old frame

        rec.check_ready(current_time)
        rec.trigger_code()

        out = cpuArray(rec.out_modes.value)

        # Expected outputs due to MMSE attenuation mock
        # Observable modes (0, 1, 2): 2.0 * 10 + 2.0 * 10 = 40.0
        # Attenuated modes (3, 4): 0.1 * 10 + 0.1 * 10 = 2.0
        expected = np.array([40.0, 40.0, 40.0, 2.0, 2.0])
        np.testing.assert_allclose(out, expected)

    @cpu_and_gpu
    def test_zero_stuffing_no_sensors_valid(self, target_device_idx, xp):
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
        """Test that matrix row dimensions must exactly match n_modes_total"""
        mat_wrong = xp.full((4, 4), 1.0, dtype=xp.float32)
        recmat_dict = {'rec_both': Recmat(mat_wrong, target_device_idx=target_device_idx)}

        with self.assertRaisesRegex(ValueError, "n_modes_total"):
            ModalrecMultirate(recmat_dict=recmat_dict, validity_masks=[[True, True]], 
                              n_modes_total=5, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_sanity_check_columns_consistency(self, target_device_idx, xp):
        """Test that dropping sensors cannot increase the required number of slopes (columns)"""

        # Baseline All-True has 2 columns
        mat_both = xp.full((5, 2), 1.0, dtype=xp.float32)

        # Single sensor matrix mysteriously requires 4 columns -> Error!
        mat_s1 = xp.full((5, 4), 2.0, dtype=xp.float32)

        recmat_dict = {
            'rec_both': Recmat(mat_both, target_device_idx=target_device_idx),
            'rec_s1': Recmat(mat_s1, target_device_idx=target_device_idx),
        }
        validity_masks = [[True, True], [True, False]]

        with self.assertRaisesRegex(ValueError, "Logical inconsistency"):
            ModalrecMultirate(recmat_dict=recmat_dict, validity_masks=validity_masks, 
                              n_modes_total=5, target_device_idx=target_device_idx)
