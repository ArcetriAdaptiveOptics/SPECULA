import unittest
import numpy as np
import tempfile
import os

from specula.data_objects.pupdata import PupData
from specula import cpuArray
from test.specula_testlib import cpu_and_gpu

class TestPupData(unittest.TestCase):

    @cpu_and_gpu
    def test_default_initialization(self, target_device_idx, xp):
        """Test PupData default init values."""
        obj = PupData(target_device_idx=target_device_idx)
        self.assertEqual(obj.ind_pup.shape, (0, 4))
        self.assertTrue(np.all(cpuArray(obj.radius) == 0))
        self.assertTrue(np.all(cpuArray(obj.cx) == 0))
        self.assertTrue(np.all(cpuArray(obj.cy) == 0))
        np.testing.assert_array_equal(obj.framesize, np.zeros(2, dtype=int))
        self.assertFalse(obj.slopes_from_intensity)

    @cpu_and_gpu
    def test_initialization_with_data(self, target_device_idx, xp):
        """Initialize PupData with given arrays."""
        ind_pup = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        radius = np.array([1.0, 2.0, 3.0, 4.0])
        cx = np.array([0.1, 0.2, 0.3, 0.4])
        cy = np.array([0.5, 0.6, 0.7, 0.8])
        framesize = [4, 4]

        obj = PupData(ind_pup=ind_pup, radius=radius, cx=cx, cy=cy,
                      framesize=framesize, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(obj.ind_pup), ind_pup)
        np.testing.assert_array_equal(cpuArray(obj.radius), radius)
        np.testing.assert_array_equal(cpuArray(obj.cx), cx)
        np.testing.assert_array_equal(cpuArray(obj.cy), cy)
        np.testing.assert_array_equal(obj.framesize, framesize)

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        """Test get_value and set_value behavior."""
        ind_pup = np.array([[1, 2, 3, 4]])
        obj = PupData(ind_pup=ind_pup.copy(), target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(obj.get_value()), ind_pup)

        new_ind = np.array([[5, 6, 7, 8]])
        obj.set_value(new_ind)
        np.testing.assert_array_equal(cpuArray(obj.ind_pup), new_ind)

        with self.assertRaises(AssertionError):
            obj.set_value(np.array([[1, 2]]))  # Shape mismatch

    def test_n_subap_property(self):
        """Check n_subap property returns number of rows."""
        ind_pup = np.zeros((3, 4))
        obj = PupData(ind_pup=ind_pup)
        self.assertEqual(obj.n_subap, 3)

    def test_zcorrection(self):
        """Check that zcorrection swaps last two columns."""
        ind_pup = np.array([[1, 2, 3, 4]])
        obj = PupData(ind_pup=ind_pup)
        corrected = obj.zcorrection(ind_pup)
        np.testing.assert_array_equal(corrected, np.array([[1, 2, 4, 3]]))

    @cpu_and_gpu
    def test_display_map(self, target_device_idx, xp):
        """Check display_map behavior with and without slopes_from_intensity."""
        ind_pup = np.array([[0, 1, 2, 3]])
        obj = PupData(ind_pup=ind_pup.copy(), framesize=[2, 2], target_device_idx=target_device_idx)
        # slopes_from_intensity False => returns flattened mask indices
        mask_indices = obj.display_map
        self.assertTrue(all(mask_indices >= 0))

        obj.set_slopes_from_intensity(True)
        display_map = obj.display_map
        np.testing.assert_array_equal(cpuArray(display_map), np.array([0, 1, 2, 3]))

    @cpu_and_gpu
    def test_single_and_complete_mask(self, target_device_idx, xp):
        ind_pup = np.array([[0, 1, 2, 3]])
        framesize = [2, 2]
        obj = PupData(ind_pup=ind_pup.copy(), framesize=framesize, target_device_idx=target_device_idx)
        single_mask = obj.single_mask()
        complete_mask = obj.complete_mask()
        self.assertEqual(single_mask.shape, (1, 1))  # half-framesize slicing
        self.assertEqual(complete_mask.shape, (2, 2))
        self.assertTrue(np.all(np.isin(cpuArray(complete_mask), [0, 1])))

    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):
        obj = PupData(framesize=[5, 6])
        hdr = obj.get_fits_header()
        self.assertEqual(hdr['VERSION'], 2)
        self.assertEqual(hdr['FSIZEX'], 5)
        self.assertEqual(hdr['FSIZEY'], 6)

    @cpu_and_gpu
    def test_save_and_restore(self, target_device_idx, xp):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, "test_pupdata.fits")

            ind_pup = np.array([[0, 1, 2, 3]])
            radius = np.array([1, 2, 3, 4])
            cx = np.array([5, 6, 7, 8])
            cy = np.array([9, 10, 11, 12])
            framesize = [2, 2]

            obj = PupData(ind_pup=ind_pup, radius=radius, cx=cx, cy=cy, framesize=framesize)
            obj.save(filename, overwrite=True)
            self.assertTrue(os.path.exists(filename))

            restored = PupData.restore(filename)
            np.testing.assert_array_equal(cpuArray(restored.ind_pup), ind_pup)
            np.testing.assert_array_equal(cpuArray(restored.radius), radius)
            np.testing.assert_array_equal(cpuArray(restored.cx), cx)
            np.testing.assert_array_equal(cpuArray(restored.cy), cy)
            np.testing.assert_array_equal(restored.framesize, framesize)


