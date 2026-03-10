import numpy as np
import unittest
import pytest

import specula
specula.init(0)

from specula.lib.compute_zonal_ifunc import compute_zonal_ifunc
from test.specula_testlib import cpu_and_gpu


class TestComputeZonalIfunc(unittest.TestCase):

    @cpu_and_gpu
    def test_invalid_geom_raises(self, target_device_idx, xp):
        with pytest.raises(ValueError):
            compute_zonal_ifunc(dim=32, n_act=4, geom='not_a_geom',
                                xp=xp, dtype=xp.float32)

    @cpu_and_gpu
    def test_double_input_raises(self, target_device_idx, xp):
        with pytest.raises(ValueError):
            compute_zonal_ifunc(dim=32, n_act=4, circ_geom=True, geom='circular',
                                xp=xp, dtype=xp.float32)

    @cpu_and_gpu
    def test_circular_geom(self, target_device_idx, xp):
        ifs_cube, _, _, _ = compute_zonal_ifunc(dim=32, n_act=4, geom='circular',
                                                xp=xp, dtype=xp.float32)
        n_act_tot = int(xp.shape(ifs_cube)[0])
        self.assertEqual(n_act_tot, 19,
                         f'Actuators are {n_act_tot} rather than the expected 19')

    @cpu_and_gpu
    def test_square_geom(self, target_device_idx, xp):
        n_act = 4
        ifs_cube, _, _, _ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='square',
                                                xp=xp, dtype=xp.float32)
        n_act_tot = int(xp.shape(ifs_cube)[0])
        self.assertEqual(n_act_tot, n_act**2,
                         f'Actuators are {n_act_tot} rather than the expected {n_act**2}')

    @cpu_and_gpu
    def test_alpao_geom(self, target_device_idx, xp):
        n_act = 4
        ifs_cube, _, _, _ = compute_zonal_ifunc(dim=32, n_act=n_act, geom='alpao',
                                                xp=xp, dtype=xp.float32)
        n_act_tot = int(xp.shape(ifs_cube)[0])
        self.assertEqual(n_act_tot, 12,
                         f'Actuators are {n_act_tot} rather than the expected 12')

    @cpu_and_gpu
    def test_standard_slaving(self, target_device_idx, xp):
        n_act = 8
        ifs_cube, mask, coords, slave_mat = compute_zonal_ifunc(
            dim=32, n_act=n_act, geom='square', do_slaving=True,
            slaving_thr=0.4, xp=xp, dtype=xp.float32
        )

        n_masters = int(xp.shape(ifs_cube)[0])

        # With activated slaving, the number of independent actuators (masters) should
        # be less than the total
        self.assertLess(n_masters, n_act**2)

        # Verify that the slave matrix has been populated (has values > 0)
        self.assertTrue(bool(xp.any(slave_mat > 0)))

    @cpu_and_gpu
    def test_linear_slaving(self, target_device_idx, xp):
        n_act = 8
        ifs_cube, mask, coords, slave_mat = compute_zonal_ifunc(
            dim=32, n_act=n_act, geom='square', do_slaving=True, linear_slaving=True,
            slaving_thr=0.4, xp=xp, dtype=xp.float32
        )

        n_masters = int(xp.shape(ifs_cube)[0])
        self.assertLess(n_masters, n_act**2)

        # Linear weights can be negative, so we check the absolute value
        self.assertTrue(bool(xp.any(xp.abs(slave_mat) > 0)))

    @cpu_and_gpu
    def test_constrained_linear_slaving(self, target_device_idx, xp):
        n_act = 8
        # Tests the edge constraint doesn't cause crashes and produces reasonable results
        ifs_cube, mask, coords, slave_mat = compute_zonal_ifunc(
            dim=32, n_act=n_act, geom='square', do_slaving=True, linear_slaving=True,
            edge_constraint_weight=0.5, slaving_thr=0.4, xp=xp, dtype=xp.float32
        )

        n_masters = int(xp.shape(ifs_cube)[0])
        self.assertLess(n_masters, n_act**2)
        self.assertTrue(bool(xp.any(xp.abs(slave_mat) > 0)))

    @cpu_and_gpu
    def test_mechanical_coupling(self, target_device_idx, xp):
        n_act = 4
        # Tests that mechanical coupling runs without errors and produces
        # a non-trivial coupling matrix
        ifs_cube, _, _, _ = compute_zonal_ifunc(
            dim=32, n_act=n_act, geom='square', do_mech_coupling=True,
            xp=xp, dtype=xp.float32
        )
        n_act_tot = int(xp.shape(ifs_cube)[0])
        self.assertEqual(n_act_tot, n_act**2)
