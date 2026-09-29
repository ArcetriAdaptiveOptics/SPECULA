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

    @cpu_and_gpu
    def test_ifs_match_scipy_rbf(self, target_device_idx, xp):
        '''Compare IFs with a direct scipy Rbf thin plate interpolation,
        both when all actuators are used as nodes (n_act=6) and when only
        the neighbours within 9*dim/n_act are used (n_act=20)'''
        from scipy.interpolate import Rbf
        from specula import cpuArray

        dim = 40
        grid_x, grid_y = np.meshgrid(np.arange(dim), np.arange(dim))
        for n_act in [6, 20]:
            ifs, mask, coords, _ = compute_zonal_ifunc(dim=dim, n_act=n_act, geom='square',
                                                       xp=xp, dtype=xp.float32)
            ifs = cpuArray(ifs)
            mask = cpuArray(mask).astype(bool)
            x, y = cpuArray(coords)
            min_distance = 9 * dim / n_act
            n_act_tot = len(x)
            for i in [0, n_act_tot // 2 + n_act // 2, n_act_tot - 1]:
                if min_distance >= dim / 2:
                    close = np.ones(n_act_tot, dtype=bool)
                else:
                    close = np.hypot(x - x[i], y - y[i]) <= min_distance
                z = np.zeros(n_act_tot)
                z[i] = 1.0
                rbf = Rbf(x[close], y[close], z[close], function='thin_plate')
                ref = rbf(grid_x, grid_y)
                if min_distance < dim / 2:
                    ref[np.hypot(grid_x - x[i], grid_y - y[i]) > 0.8 * min_distance] = 0
                np.testing.assert_allclose(ifs[i], ref[mask], rtol=0, atol=1e-6,
                                           err_msg=f'n_act={n_act}, actuator {i}')
