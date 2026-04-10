from __future__ import annotations

import unittest
from unittest.mock import patch

import specula
specula.init(0)  # Default target device

import numpy as np
from specula import cpuArray
from specula.data_objects.recmat import Recmat
from specula.data_objects.slopes import Slopes
from specula.processing_objects.modalrec_multirate import ModalrecMultirate
from specula.simul import Simul
from specula.lib.utils import import_class as real_import_class

from test.specula_testlib import cpu_and_gpu

class TestModalrecMultirate(unittest.TestCase):

    @cpu_and_gpu
    def test_outputs_created_in_init(self, target_device_idx, xp):
        n_modes = 5
        recmat_list = [
            Recmat(xp.ones((n_modes, 4), dtype=xp.float32), target_device_idx=target_device_idx),
            Recmat(xp.ones((n_modes, 4), dtype=xp.float32), target_device_idx=target_device_idx),
            Recmat(xp.ones((n_modes, 4), dtype=xp.float32), target_device_idx=target_device_idx)
        ]
        validity_masks = [[True, True], [True, False], [False, True]]

        rec = ModalrecMultirate(
            recmat_list=recmat_list,
            validity_masks=validity_masks,
            n_modes_total=n_modes,
            target_device_idx=target_device_idx
        )

        self.assertEqual(rec.n_sensors, 2)
        self.assertEqual(len(rec.out_modes_list), 2)
        self.assertIn('out_modes_0', rec.outputs)
        self.assertIn('out_modes_1', rec.outputs)
        np.testing.assert_allclose(cpuArray(rec.out_modes_list[0].value), 0.0)
        np.testing.assert_allclose(cpuArray(rec.out_modes_list[1].value), 0.0)

    def _setup_reconstructor(self, target_device_idx, xp):
        self.n_modes = 5
        self.n_slopes_per_wfs = 2

        # 1. Create Mock Reconstruction Matrices
        # M=5 rows. N=2 sensors -> 4 columns if both active.
        mat_both = xp.full((self.n_modes, 4), 1.0, dtype=xp.float32)

        # Single-sensor validity states still use the full 2-sensor geometry
        mat_s1 = xp.full((self.n_modes, 4), 2.0, dtype=xp.float32)
        mat_s2 = xp.full((self.n_modes, 4), 3.0, dtype=xp.float32)

        recmat_list = [
            Recmat(mat_both, target_device_idx=target_device_idx),
            Recmat(mat_s1, target_device_idx=target_device_idx),
            Recmat(mat_s2, target_device_idx=target_device_idx)
        ]

        validity_masks = [
            [True, True],
            [True, False],
            [False, True]
        ]

        rec = ModalrecMultirate(
            recmat_list=recmat_list,
            validity_masks=validity_masks,
            n_modes_total=self.n_modes,
            target_device_idx=target_device_idx
        )

        slopes_s1 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)
        slopes_s2 = Slopes(length=self.n_slopes_per_wfs, target_device_idx=target_device_idx)

        # Sensor 1 has slopes [10, 10]. Sensor 2 has slopes [20, 20].
        slopes_s1.slopes[:] = 10.0
        slopes_s2.slopes[:] = 20.0

        rec.inputs['in_slopes_list'].set([slopes_s1, slopes_s2])
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)

        # setup() validates topology and moves matrices to active device
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

        out_0 = cpuArray(rec.out_modes_list[0].value)
        out_1 = cpuArray(rec.out_modes_list[1].value)

        # Sensor 0 output: block of 1.0s * [10, 10] = [20, 20, 20, 20, 20]
        np.testing.assert_allclose(out_0, 20.0)
        # Sensor 1 output: block of 1.0s * [20, 20] = [40, 40, 40, 40, 40]
        np.testing.assert_allclose(out_1, 40.0)

        self.assertEqual(rec.out_modes_list[0].generation_time, current_time)

    @cpu_and_gpu
    def test_single_sensor_valid(self, target_device_idx, xp):
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 2.0
        s1.generation_time = current_time
        s2.generation_time = 1.0  # Old frame

        rec.check_ready(current_time)
        rec.trigger_code()

        out_0 = cpuArray(rec.out_modes_list[0].value)
        out_1 = cpuArray(rec.out_modes_list[1].value)

        # Sensor 0 output: block of 2.0s * [10, 10] = [40, 40, 40, 40, 40]
        np.testing.assert_allclose(out_0, 40.0)
        # Sensor 1 output: inactive -> zeros
        np.testing.assert_allclose(out_1, 0.0)

    @cpu_and_gpu
    def test_zero_stuffing_no_sensors_valid(self, target_device_idx, xp):
        rec, s1, s2 = self._setup_reconstructor(target_device_idx, xp)

        current_time = 3.0
        s1.generation_time = 2.0
        s2.generation_time = 1.0

        rec.check_ready(current_time)
        rec.trigger_code()

        out_0 = cpuArray(rec.out_modes_list[0].value)
        out_1 = cpuArray(rec.out_modes_list[1].value)

        np.testing.assert_allclose(out_0, 0.0)
        np.testing.assert_allclose(out_1, 0.0)

    @cpu_and_gpu
    def test_sanity_check_dimensions(self, target_device_idx, xp):
        """Test that matrix row dimensions must exactly match n_modes_total"""
        mat_wrong = xp.full((4, 4), 1.0, dtype=xp.float32)
        recmat_list = [Recmat(mat_wrong, target_device_idx=target_device_idx)]

        with self.assertRaisesRegex(ValueError, "n_modes_total"):
            ModalrecMultirate(recmat_list=recmat_list, validity_masks=[[True, True]],
                              n_modes_total=5, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_setup_loads_masks_from_list_order(self, target_device_idx, xp):
        n_modes = 5
        mat_both = xp.full((n_modes, 4), 1.0, dtype=xp.float32)
        mat_s1 = xp.full((n_modes, 4), 2.0, dtype=xp.float32)
        mat_s2 = xp.full((n_modes, 4), 3.0, dtype=xp.float32)

        recmat_list = [
            Recmat(mat_both, target_device_idx=target_device_idx),
            Recmat(mat_s1, target_device_idx=target_device_idx),
            Recmat(mat_s2, target_device_idx=target_device_idx)
        ]

        rec = ModalrecMultirate(
            recmat_list=recmat_list,
            validity_masks=[[True, True], [True, False], [False, True]],
            n_modes_total=n_modes,
            target_device_idx=target_device_idx
        )

        slopes_s1 = Slopes(length=2, target_device_idx=target_device_idx)
        slopes_s2 = Slopes(length=2, target_device_idx=target_device_idx)
        rec.inputs['in_slopes_list'].set([slopes_s1, slopes_s2])
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)
        rec.setup()

        self.assertIn((True, True), rec.xp_recmat_by_mask)
        self.assertIn((True, False), rec.xp_recmat_by_mask)
        self.assertIn((False, True), rec.xp_recmat_by_mask)

    @cpu_and_gpu
    def test_accepts_full_width_matrices_for_multirate_masks(self, target_device_idx, xp):
        n_modes = 5
        mat_all = xp.full((n_modes, 6), 1.0, dtype=xp.float32)
        mat_110 = xp.full((n_modes, 6), 2.0, dtype=xp.float32)
        mat_101 = xp.full((n_modes, 6), 3.0, dtype=xp.float32)

        rec = ModalrecMultirate(
            recmat_list=[
                Recmat(mat_all, target_device_idx=target_device_idx),
                Recmat(mat_110, target_device_idx=target_device_idx),
                Recmat(mat_101, target_device_idx=target_device_idx),
            ],
            validity_masks=[[True, True, True], [True, True, False], [True, False, True]],
            n_modes_total=n_modes,
            target_device_idx=target_device_idx
        )

        slopes = [Slopes(length=2, target_device_idx=target_device_idx) for _ in range(3)]
        rec.inputs['in_slopes_list'].set(slopes)
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)
        rec.setup()

        self.assertEqual(rec.n_sensors, 3)
        self.assertIn((True, True, True), rec.xp_recmat_by_mask)
        self.assertIn((True, True, False), rec.xp_recmat_by_mask)
        self.assertIn((True, False, True), rec.xp_recmat_by_mask)

    @cpu_and_gpu
    def test_sanity_check_columns(self, target_device_idx, xp):
        n_modes = 5
        mat_bad = xp.full((n_modes, 3), 1.0, dtype=xp.float32)
        recmat_list = [Recmat(mat_bad, target_device_idx=target_device_idx)]

        rec = ModalrecMultirate(
            recmat_list=recmat_list,
            validity_masks=[[True, False]],
            n_modes_total=n_modes,
            target_device_idx=target_device_idx
        )

        slopes_s1 = Slopes(length=2, target_device_idx=target_device_idx)
        slopes_s2 = Slopes(length=2, target_device_idx=target_device_idx)
        rec.inputs['in_slopes_list'].set([slopes_s1, slopes_s2])
        rec.local_inputs['in_slopes_list'] = rec.inputs['in_slopes_list'].get(target_device_idx)

        with self.assertRaisesRegex(ValueError, "full sensor vector"):
            rec.setup()

    def test_integration_simul_with_list_object(self):
        """Integration test: Simul builds ModalrecMultirate from `recmat_list_object`."""
        class DummySimulParams:
            def __init__(self, root_dir='dummy', **_kwargs):
                self.root_dir = root_dir

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            return real_import_class(classname, additional_modules)

        rec_both = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s1 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s2 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'rec': {
                'class': 'ModalrecMultirate',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_list_object': ['tag_both', 'tag_s1', 'tag_s2'],
                'validity_masks': [[True, True], [True, False], [False, True]],
                'n_modes_total': 5
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore', side_effect=[rec_both, rec_s1, rec_s2]):
                simul = Simul([])
                simul.build_objects(params)

                rec_obj = simul.objs['rec']
                self.assertIsInstance(rec_obj, ModalrecMultirate)
                self.assertEqual(set(rec_obj.recmat_by_mask.keys()),
                                 {(True, True), (True, False), (False, True)})

    def test_integration_simul_with_list_object_and_validity_masks(self):
        """Integration test: Simul builds ModalrecMultirate from a list of recmat tags."""
        class DummySimulParams:
            def __init__(self, root_dir='dummy', **_kwargs):
                self.root_dir = root_dir

        def mock_import(classname, additional_modules=None):
            if classname == 'SimulParams':
                return DummySimulParams
            return real_import_class(classname, additional_modules)

        rec_both = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s1 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)
        rec_s2 = Recmat(np.ones((5, 4), dtype=np.float32), target_device_idx=-1, precision=0)

        params = {
            'main': {
                'class': 'SimulParams',
                'root_dir': 'dummy'
            },
            'rec': {
                'class': 'ModalrecMultirate',
                'target_device_idx': -1,
                'precision': 0,
                'recmat_list_object': ['tag_both', 'tag_s1', 'tag_s2'],
                'validity_masks': [[True, True], [True, False], [False, True]],
                'n_modes_total': 5
            }
        }

        with patch('specula.simul.import_class', side_effect=mock_import):
            with patch('specula.data_objects.recmat.Recmat.restore',
                       side_effect=[rec_both, rec_s1, rec_s2]):
                simul = Simul([])
                simul.build_objects(params)

                rec_obj = simul.objs['rec']
                self.assertIsInstance(rec_obj, ModalrecMultirate)
                self.assertEqual(set(rec_obj.recmat_by_mask.keys()),
                                 {(True, True), (True, False), (False, True)})
                self.assertEqual(rec_obj.recmat_by_mask[(True, True)].tag, 'tag_both')
                self.assertEqual(rec_obj.recmat_by_mask[(True, False)].tag, 'tag_s1')
                self.assertEqual(rec_obj.recmat_by_mask[(False, True)].tag, 'tag_s2')
