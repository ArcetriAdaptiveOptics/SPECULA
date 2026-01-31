import specula
specula.init(0)  # Default target device

import os
import numpy as np
import unittest

from specula import cpuArray
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv
from specula.lib.make_mask import make_mask

from test.specula_testlib import cpu_and_gpu

class TestIFunv(unittest.TestCase):

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        self.mask = np.array([[0, 1, 0], [0, 1, 0]], dtype=int)

        self.inv_data = np.array([[-0.94444444,  0.44444444],
                             [-0.11111111,  0.11111111],
                             [ 0.72222222, -0.22222222]])

        self.inv_filename = os.path.join(self.datadir, 'ifunc_inv.fits')
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    def tearDown(self):
        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_ifunc_inv_data(self, target_device_idx, xp):
        '''Test that the inversion in IFunc is correct'''
        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv = ifunc.inverse()
        assert isinstance(inv, IFuncInv)

        np.testing.assert_array_almost_equal(cpuArray(self.inv_data), cpuArray(inv.ifunc_inv))

    @cpu_and_gpu
    def test_ifunc_inv_idx(self, target_device_idx, xp):
        '''Test that the mask indexes in the inverted ifunc are the same as in IFunc'''

        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv = ifunc.inverse()

        idx1 = cpuArray(ifunc.idx_inf_func[0]), cpuArray(ifunc.idx_inf_func[1])
        idx2 = cpuArray(inv.idx_inf_func[0]), cpuArray(inv.idx_inf_func[1])
        np.testing.assert_array_equal(idx1[0], idx2[0])
        np.testing.assert_array_equal(idx1[1], idx2[1])

    @cpu_and_gpu
    def test_ifunc_inv_restore(self, target_device_idx, xp):
        '''Test that data saved and restored is the same as data obtained from IFunc.inverse()'''

        try:
            os.unlink(self.inv_filename)
        except FileNotFoundError:
            pass

        inv = IFuncInv(self.inv_data, mask=self.mask, target_device_idx=target_device_idx)
        inv.save(self.inv_filename)

        ifunc = IFunc(self.data, mask=self.mask, target_device_idx=target_device_idx)
        inv1 = ifunc.inverse()
        inv2 = IFuncInv.restore(self.inv_filename)

        np.testing.assert_array_almost_equal(cpuArray(inv1.ifunc_inv), cpuArray(inv2.ifunc_inv))
        idx1 = cpuArray(inv1.idx_inf_func[0]), cpuArray(inv1.idx_inf_func[1])
        idx2 = cpuArray(inv2.idx_inf_func[0]), cpuArray(inv2.idx_inf_func[1])
        np.testing.assert_array_equal(idx1[0], idx2[0])
        np.testing.assert_array_equal(idx1[1], idx2[1])

    @cpu_and_gpu
    def test_ifunc_2d_to_3d(self, target_device_idx, xp):
        '''Test for ifunc_2d_to_3d method'''
        mask = make_mask(64)
        ifunc = IFunc(type_str='zernike', mask=mask, nmodes=3, npixels=64, target_device_idx=target_device_idx)
        data = ifunc.influence_function
        result_3d = ifunc.ifunc_2d_to_3d(normalize=False)
        result_3d_cpu = cpuArray(result_3d)

        # Check that values are placed only where mask > 0
        mask_idx = np.where(mask > 0)

        # Result should have the same dtype as the ifunc
        self.assertEqual(result_3d.dtype, ifunc.dtype)

        # Expected shape: (npixels, npixels, nmodes)
        expected_shape = (mask.shape[0], mask.shape[1], data.shape[0])
        self.assertEqual(result_3d.shape, expected_shape)

        # For each mode, check values at masked positions
        for mode in range(data.shape[0]):
            # Get values at masked positions for this mode
            values_at_mask = result_3d_cpu[mask_idx[0], mask_idx[1], mode]
            expected_values = data[mode, :]  # All pixels for this mode
            np.testing.assert_array_equal(cpuArray(values_at_mask), cpuArray(expected_values))

        # Check that values are zero where mask == 0
        mask_zero_idx = np.where(mask == 0)
        for mode in range(data.shape[0]):
            values_at_zero = result_3d_cpu[mask_zero_idx[0], mask_zero_idx[1], mode]
            expected_zeros = np.zeros(len(mask_zero_idx[0]))
            np.testing.assert_array_equal(cpuArray(values_at_zero), expected_zeros)

        # Test with normalization
        result_norm = ifunc.ifunc_2d_to_3d(normalize=True)

        # Calculate expected RMS values for each mode
        expected_rms = np.sqrt(np.mean(cpuArray(data)**2, axis=1))

        # Check that normalized result has correct scaling
        mask_idx = np.where(mask > 0)

        for mode in range(data.shape[0]):
            values_no_norm = cpuArray(result_3d[mask_idx[0], mask_idx[1], mode])
            values_norm = cpuArray(result_norm[mask_idx[0], mask_idx[1], mode])

            # Normalized values should equal non-normalized values divided by RMS
            expected_normalized = values_no_norm / expected_rms[mode]
            np.testing.assert_array_almost_equal(values_norm, expected_normalized)

    @cpu_and_gpu
    def test_zonal_central_ifunc_position(self, target_device_idx, xp):
        '''Test that central influence function is centered in the array'''
        dim = 64
        n_act = 9
        mask = make_mask(dim)

        # Test for different geometries
        for circ_geom in [True, False]:
            ifunc = IFunc(type_str='zonal', mask=mask, n_act=n_act, npixels=dim,
                         circ_geom=circ_geom, target_device_idx=target_device_idx)

            # Get the influence function cube
            ifunc_3d = ifunc.ifunc_2d_to_3d(normalize=False)
            ifunc_3d_cpu = cpuArray(ifunc_3d)

            # For zonal, we need to find the central actuator
            # The central IF should have its peak at the center
            center = (dim - 1) / 2.0

            # Find which IF has maximum value closest to center
            peaks = np.array([np.unravel_index(np.argmax(ifunc_3d_cpu[:, :, i]),
                                               ifunc_3d_cpu[:, :, i].shape)
                             for i in range(ifunc_3d_cpu.shape[2])])

            # Find actuator closest to center
            distances_to_center = np.sqrt((peaks[:, 0] - center)**2 + (peaks[:, 1] - center)**2)
            central_idx = np.argmin(distances_to_center)

            # Get peak position of central IF
            central_if = ifunc_3d_cpu[:, :, central_idx]
            peak_pos = np.unravel_index(np.argmax(np.abs(central_if)), central_if.shape)

            # Check that peak is close to center (within 0.5 pixel)
            self.assertAlmostEqual(peak_pos[0], center, delta=0.5,
                        msg=f"Central IF peak Y position not centered for {circ_geom} geometry")
            self.assertAlmostEqual(peak_pos[1], center, delta=0.5,
                        msg=f"Central IF peak X position not centered for {circ_geom} geometry")

    @cpu_and_gpu
    def test_zonal_edge_ifunc_rotation_symmetry(self, target_device_idx, xp):
        '''Test that edge influence functions have rotational symmetry'''
        dim = 64
        n_act = 9
        mask = make_mask(dim)

        # Test for symmetric geometries
        for circ_geom in [True, False]:
            ifunc = IFunc(type_str='zonal', mask=mask, n_act=n_act, npixels=dim, 
                         circ_geom=circ_geom, target_device_idx=target_device_idx)

            # Get the influence function cube
            ifunc_3d = ifunc.ifunc_2d_to_3d(normalize=False)
            ifunc_3d_cpu = cpuArray(ifunc_3d)

            # Find center and peak positions
            center = np.array([(dim - 1) / 2.0, (dim - 1) / 2.0])
            n_actuators = ifunc_3d_cpu.shape[2]

            # Get peak position for each actuator
            peaks = np.array([np.unravel_index(np.argmax(ifunc_3d_cpu[:, :, i]),
                                               ifunc_3d_cpu[:, :, i].shape)
                             for i in range(n_actuators)])

            # Find pairs of opposite actuators (180deg rotation)
            tested_pairs = 0

            for i in range(n_actuators):
                pos_i = peaks[i]
                # Calculate 180deg rotated position
                pos_i_rotated = 2 * center - pos_i

                # Find actuator with peak closest to rotated position
                distances = np.sqrt(np.sum((peaks - pos_i_rotated)**2, axis=1))
                j = np.argmin(distances)

                # Only test if we found a true opposite (distance to rotated pos < 1.5 pixels)
                # i < j to avoid testing same pair twice
                if distances[j] < 1.5 and i != j and i < j:
                    if_i = ifunc_3d_cpu[:, :, i]
                    if_j = ifunc_3d_cpu[:, :, j]

                    # Rotate if_j by 180deg
                    if_j_rotated = np.rot90(if_j, k=2)

                    # Compare using correlation in the mask region
                    mask_cpu = cpuArray(mask)
                    mask_idx = np.where(mask_cpu > 0)

                    correlation = np.corrcoef(
                        if_i[mask_idx].ravel(),
                        if_j_rotated[mask_idx].ravel()
                    )[0, 1]

                    self.assertGreater(correlation, 0.999,
                        msg=f"Low correlation ({correlation:.3f}) between opposite IFs {i} and {j}"
                            f" for {circ_geom} geometry")
                    tested_pairs += 1

            # Make sure we actually tested some pairs
            self.assertGreater(tested_pairs, 0,
                             msg=f"No opposite actuator pairs found for {circ_geom} geometry")
