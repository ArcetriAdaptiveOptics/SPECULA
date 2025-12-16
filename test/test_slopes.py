
import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu

class TestSlopes(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_slopes.fits')

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        id_slopes_before = id(slopes.slopes)
        
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        
        id_slopes_after = id(slopes.slopes)

        assert id_slopes_before == id_slopes_after

    @cpu_and_gpu
    def test_set_value_shape_mismatch(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            new_slopes_data = xp.ones(11)
            slopes.set_value(new_slopes_data)

    @cpu_and_gpu
    def test_get_value(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        expected_value = xp.ones(10)
        slopes.set_value(expected_value)

        np.testing.assert_array_equal(cpuArray(slopes.get_value()), cpuArray(expected_value))

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(new_slopes_data))

    @cpu_and_gpu
    def test_slopes_save_restore_roundtrip(self, target_device_idx, xp):
        
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        slopes.save(self.filename)

        slopes2 = Slopes.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(slopes2.slopes))
        np.testing.assert_array_equal(cpuArray(slopes.indices_x), cpuArray(slopes2.indices_x))
        np.testing.assert_array_equal(cpuArray(slopes.indices_y), cpuArray(slopes2.indices_y))

    @cpu_and_gpu
    def test_slopes_save_restore_roundtrip_version2(self, target_device_idx, xp):
        
        slopes = Slopes(10, target_device_idx=target_device_idx)
        new_slopes_data = xp.ones(10)
        slopes.set_value(new_slopes_data)
        slopes.save(self.filename)

        with fits.open(self.filename, mode="update") as f:
            f[0].header["VERSION"] = 2   # pylint: disable=no-member
            del f[0].header["LENGTH"]    # pylint: disable=no-member

        slopes2 = Slopes.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(slopes.slopes), cpuArray(slopes2.slopes))
        np.testing.assert_array_equal(cpuArray(slopes.indices_x), cpuArray(slopes2.indices_x))
        np.testing.assert_array_equal(cpuArray(slopes.indices_y), cpuArray(slopes2.indices_y))

    @cpu_and_gpu
    def test_resize(self, target_device_idx, xp):
        slopes = Slopes(10, target_device_idx=target_device_idx)
        slopes.resize(20)
        assert slopes.size == 20
        slopes_ref = Slopes(20, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(slopes.indices_x), cpuArray(slopes_ref.indices_x))
        np.testing.assert_array_equal(cpuArray(slopes.indices_y), cpuArray(slopes_ref.indices_y))

    @cpu_and_gpu
    def test_resize_with_interleave(self, target_device_idx, xp):
        slopes = Slopes(10, interleave=True, target_device_idx=target_device_idx)
        slopes.resize(20)
        assert slopes.size == 20
        slopes_ref = Slopes(20, interleave=True, target_device_idx=target_device_idx)
        np.testing.assert_array_equal(cpuArray(slopes.indices_x), cpuArray(slopes_ref.indices_x))
        np.testing.assert_array_equal(cpuArray(slopes.indices_y), cpuArray(slopes_ref.indices_y))

    @cpu_and_gpu
    def test_indices_correct_non_interleaved(self, target_device_idx, xp):
        """
        Test that indices_x and indices_y are computed correctly when interleave=False.
        """
        slopes = Slopes(length=6, interleave=False, target_device_idx=target_device_idx)
        expected_x = xp.arange(0, slopes.size // 2)
        expected_y = expected_x + slopes.size // 2
        assert xp.all(slopes.indices_x == expected_x)
        assert xp.all(slopes.indices_y == expected_y)

    @cpu_and_gpu
    def test_indices_correct_interleaved(self, target_device_idx, xp):
        """
        Test that indices_x and indices_y are computed correctly when interleave=True.
        """
        slopes = Slopes(length=6, interleave=True, target_device_idx=target_device_idx)
        expected_x = xp.arange(0, slopes.size // 2) * 2
        expected_y = expected_x + 1
        assert xp.all(slopes.indices_x == expected_x)
        assert xp.all(slopes.indices_y == expected_y)


    @cpu_and_gpu
    def test_sum_adds_scaled_slopes(self, target_device_idx, xp):
        """
        Test that sum() adds slopes correctly using a scaling factor.
        """
        s1 = Slopes(length=4, target_device_idx=target_device_idx)
        s2 = Slopes(length=4, target_device_idx=target_device_idx)
        s1.slopes[:] = xp.array([1, 2, 3, 4])
        s2.slopes[:] = xp.array([1, 1, 1, 1])
        s1.sum(s2, factor=2)
        xp.testing.assert_array_equal(s1.slopes, xp.array([3, 4, 5, 6]))


    @cpu_and_gpu
    def test_subtract_slopes_from_slopes(self, target_device_idx, xp):
        """
        Test that subtract() correctly subtracts slopes when given another Slopes object.
        """
        s1 = Slopes(length=4, target_device_idx=target_device_idx)
        s2 = Slopes(length=4, target_device_idx=target_device_idx)
        s1.slopes[:] = xp.array([5, 6, 7, 8])
        s2.slopes[:] = xp.array([1, 2, 3, 4])
        s1.subtract(s2)
        xp.testing.assert_array_equal(s1.slopes, xp.array([4, 4, 4, 4]))


    @cpu_and_gpu
    def test_subtract_slopes_from_basevalue(self, target_device_idx, xp):
        """
        Test that subtract() works when given a BaseValue object.
        """
        s1 = Slopes(length=4, target_device_idx=target_device_idx)
        s1.slopes[:] = xp.array([5, 6, 7, 8])
        base_value = BaseValue(value=xp.array([1, 1, 1, 1]), target_device_idx=target_device_idx)
        s1.subtract(base_value)
        xp.testing.assert_array_equal(s1.slopes, xp.array([4, 5, 6, 7]))


    @cpu_and_gpu
    def test_x_remap2d_and_y_remap2d_flat_indices(self, target_device_idx, xp):
        """
        Test x_remap2d and y_remap2d with flat (1D) indices.
        """
        slopes = Slopes(length=6, interleave=False, target_device_idx=target_device_idx)
        slopes.slopes[:] = xp.array([1, 2, 3, 4, 5, 6])
        frame_x = xp.zeros(3)
        frame_y = xp.zeros(3)
        idx = xp.array([0, 1, 2])
        slopes.x_remap2d(frame_x, idx)
        slopes.y_remap2d(frame_y, idx)
        xp.testing.assert_array_equal(frame_x, slopes.slopes[slopes.indices_x])
        xp.testing.assert_array_equal(frame_y, slopes.slopes[slopes.indices_y])


    @cpu_and_gpu
    def test_x_remap2d_and_y_remap2d_2d_indices(self, target_device_idx, xp):
        """
        Test x_remap2d and y_remap2d with 2D indices.
        """
        slopes = Slopes(length=6, interleave=False, target_device_idx=target_device_idx)
        slopes.slopes[:] = xp.array([1, 2, 3, 4, 5, 6])
        frame_x = xp.zeros((4, 4))
        frame_y = xp.zeros((4, 4))
        idx = xp.array([[0, 1, 2], [0, 1, 2]])
        slopes.x_remap2d(frame_x, idx)
        slopes.y_remap2d(frame_y, idx)
        xp.testing.assert_array_equal(frame_x[(idx[0], idx[1])], slopes.slopes[slopes.indices_x])
        xp.testing.assert_array_equal(frame_y[(idx[0], idx[1])], slopes.slopes[slopes.indices_y])

    @cpu_and_gpu
    def test_x_remap2d_invalid_index_shape(self, target_device_idx, xp):
        """Test that x_remap2d raises ValueError for invalid idx shape."""
        slopes = Slopes(length=6, interleave=False, target_device_idx=target_device_idx)
        frame = np.zeros(10, dtype=np.float32)
        idx = np.zeros((2, 2, 2))  # 3D shape → invalid
        with self.assertRaises(ValueError):
            slopes.x_remap2d(frame, idx)

    @cpu_and_gpu
    def test_y_remap2d_invalid_index_shape(self, target_device_idx, xp):
        """Test that y_remap2d raises ValueError for invalid idx shape."""
        slopes = Slopes(length=6, interleave=False, target_device_idx=target_device_idx)
        frame = np.zeros(10, dtype=np.float32)
        idx = np.zeros((2, 2, 2))  # 3D shape → invalid
        with self.assertRaises(ValueError):
            slopes.y_remap2d(frame, idx)

    @cpu_and_gpu
    def test_get2d_intensity_case(self, target_device_idx, xp):
        """
        Test get2d() for the intensity case where slopes.size == len(display_map).
        """
        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        slopes.slopes[:] = xp.array([10, 20, 30, 40])
        slopes.single_mask = xp.zeros((2, 2))
        slopes.display_map = xp.array([0, 1, 2, 3])
        result = slopes.get2d()
        expected = xp.zeros_like(slopes.single_mask)
        xp.put(expected.ravel(), slopes.display_map, slopes.slopes)
        xp.testing.assert_array_equal(result, expected)

    @cpu_and_gpu
    def test_get2d_with_slopes_xy(self, target_device_idx, xp):
        """Test get2d returns stacked XY slopes when slopes size != len(display_map)."""
        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        mask = xp.zeros((2, 2), dtype=bool)
        slopes.single_mask = mask
        slopes.display_map = xp.array([0, 1])
        slopes.slopes = xp.arange(4)
        result = slopes.get2d()
        self.assertEqual(result.shape[0], 2)  # two arrays stacked
        self.assertEqual(result.shape[1:], mask.shape)

    @cpu_and_gpu
    def test_get2d_without_single_mask_raises(self, target_device_idx, xp):
        """Test get2d raises ValueError if single_mask is None."""
        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        slopes.single_mask = None
        slopes.display_map = xp.zeros(4)
        with self.assertRaises(ValueError):
            slopes.get2d()

    @cpu_and_gpu
    def test_get2d_without_display_map_raises(self, target_device_idx, xp):
        """Test get2d raises ValueError if display_map is None."""
        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        slopes.single_mask = xp.zeros((2, 2))
        slopes.display_map = None
        with self.assertRaises(ValueError):
            slopes.get2d()

    @cpu_and_gpu
    def test_rotate_slopes(self, target_device_idx, xp):
        """
        Test rotate() correctly rotates and flips x and y slopes.
        """
        slopes = Slopes(length=4, interleave=False, target_device_idx=target_device_idx)
        slopes.slopes[:] = xp.array([1, 0, 0, 1])
        slopes.rotate(angle=90, flipx=False, flipy=False)

        # Use absolute tolerance in the following tests, since
        # we are comparing with zero and one.
        # Zero would fail the rtol test, and for the value
        # of one, atol and rtol are the same.
        xp.testing.assert_allclose(
            slopes.xslopes, xp.array([0, -1]), atol=1e-5
        )
        xp.testing.assert_allclose(
            slopes.yslopes, xp.array([1, 0]), atol=1e-5
        )

    @cpu_and_gpu
    def test_fits_header(self, target_device_idx, xp):

        slopes = Slopes(length=4, interleave=True, target_device_idx=target_device_idx)
        slopes.pupdata_tag = 'test_tag1'
        slopes.subapdata_tag = 'test_tag2'

        hdr = slopes.get_fits_header()

        assert hdr['OBJ_TYPE'] == 'Slopes'
        assert hdr['VERSION'] == 3
        assert hdr['INTRLVD'] == 1
        assert hdr['LENGTH'] == 4
        assert hdr['PUPDTAG'] == 'test_tag1'
        assert hdr['SUBAPTAG'] == 'test_tag2'

