import unittest
import numpy as np
import os

import specula
specula.init(0)  # Default target device

from specula.data_objects.subap_data import SubapData
from specula import cpuArray
import tempfile

from test.specula_testlib import cpu_and_gpu

class TestSubapData(unittest.TestCase):

    @cpu_and_gpu
    def test_initialization_and_properties(self, target_device_idx, xp):
        """Test initialization and basic properties."""
        idxs = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        display_map = np.array([0, 1])
        nx, ny = 2, 2
        energy_th = 0.1

        obj = SubapData(idxs=idxs, display_map=display_map, nx=nx, ny=ny,
                        energy_th=energy_th, target_device_idx=target_device_idx)
        
        np.testing.assert_array_equal(cpuArray(obj.idxs), idxs)
        np.testing.assert_array_equal(cpuArray(obj.display_map), display_map)
        self.assertEqual(obj.nx, nx)
        self.assertEqual(obj.ny, ny)
        self.assertEqual(obj.energy_th, energy_th)
        self.assertEqual(obj.n_subaps, 2)
        self.assertEqual(obj.np_sub, 2)  # sqrt(4) = 2

    @cpu_and_gpu
    def test_single_mask(self, target_device_idx, xp):
        """Check single_mask returns correct shape and binary content."""
        idxs = np.array([[0, 1], [2, 3]])
        display_map = np.array([0, 3])
        obj = SubapData(idxs=idxs, display_map=display_map, nx=2, ny=2, target_device_idx=target_device_idx)
        mask = obj.single_mask()
        self.assertEqual(mask.shape, (2, 2))
        self.assertTrue(set(tuple(cpuArray(mask.flatten()))) <= {0, 1})
        self.assertEqual(mask[0, 0], 1)
        self.assertEqual(mask[1, 1], 1)

    @cpu_and_gpu
    def test_subap_idx_and_display_map_idx(self, target_device_idx, xp):
        """Test accessors for subap indices and display_map positions."""
        idxs = np.array([[0, 1], [2, 3]])
        display_map = np.array([0, 3])
        obj = SubapData(idxs=idxs, display_map=display_map, nx=2, ny=2, target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(obj.subap_idx(0)), idxs[0])
        np.testing.assert_array_equal(cpuArray(obj.subap_idx(1)), idxs[1])
        self.assertEqual(obj.display_map_idx(0), display_map[0])
        self.assertEqual(obj.display_map_idx(1), display_map[1])

    def test_save_and_restore(self):
        """Check that saving and restoring preserves all data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_subapdata.fits")
            idxs = np.array([[0, 1], [2, 3]])
            display_map = np.array([0, 3])
            nx, ny = 2, 2
            energy_th = 0.5

            obj = SubapData(idxs=idxs, display_map=display_map, nx=nx, ny=ny, energy_th=energy_th)
            obj.save(filename, overwrite=True)
            self.assertTrue(os.path.exists(filename))

            restored = SubapData.restore(filename)
            np.testing.assert_array_equal(cpuArray(restored.idxs), idxs)
            np.testing.assert_array_equal(cpuArray(restored.display_map), display_map)
            self.assertEqual(restored.nx, nx)
            self.assertEqual(restored.ny, ny)
            self.assertEqual(restored.energy_th, energy_th)
            self.assertEqual(restored.n_subaps, obj.n_subaps)
            self.assertEqual(restored.np_sub, obj.np_sub)

