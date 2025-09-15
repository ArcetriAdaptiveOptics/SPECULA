

import specula
specula.init(0)  # Default target device

import os
import sys
import unittest

from specula import np
from specula import cpuArray
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from test.specula_testlib import cpu_and_gpu

class TestIntmat(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_im.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):
        '''
        Test that an Intmat can be saved into a file and then
        restore with no change in its data.
        '''

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        im_data = xp.arange(10).reshape((5,2))
        im = Intmat(im_data, target_device_idx=target_device_idx)
        
        im.save(self.filename)
        im2 = Intmat.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(im.intmat), cpuArray(im2.intmat))

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_init_with_defaults(self, target_device_idx, xp):
        mat = xp.zeros((3, 3))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        assert xp.allclose(intmat.intmat, mat)
        assert intmat.slope_mm is None
        assert intmat.slope_rms is None
        assert intmat.pupdata_tag == ''
        assert intmat.subapdata_tag == ''
        assert intmat.norm_factor == 0.0

    @cpu_and_gpu
    def test_init_with_intmat(self, target_device_idx, xp):
        """Test initializing Intmat with an existing intmat array"""
        intmat = xp.array([[1, 2], [3, 4]])
        im = Intmat(intmat=intmat, target_device_idx=target_device_idx)

        # The intmat should match the input values
        xp.testing.assert_array_equal(im.intmat, intmat)

        # Other default attributes should be set
        self.assertIsNone(im.slope_mm)
        self.assertIsNone(im.slope_rms)
        self.assertEqual(im.pupdata_tag, "")
        self.assertEqual(im.subapdata_tag, "")
        self.assertEqual(im.norm_factor, 0.0)

        # Modes and slopes views should exist
        self.assertTrue(hasattr(im, "modes"))
        self.assertTrue(hasattr(im, "slopes"))

    @cpu_and_gpu
    def test_init_with_nmodes_and_nslopes(self, target_device_idx, xp):
        """Test initializing Intmat with nmodes and nslopes when intmat is not provided"""
        nmodes = 3
        nslopes = 5
        im = Intmat(nmodes=nmodes, nslopes=nslopes, target_device_idx=target_device_idx)

        # Shape should match nslopes x nmodes
        self.assertEqual(im.intmat.shape, (nslopes, nmodes))

        # The array should be zeros initially
        xp.testing.assert_array_equal(im.intmat, xp.zeros((nslopes, nmodes), dtype=im.dtype))

    @cpu_and_gpu
    def test_init_without_intmat_and_missing_nmodes_raises(self, target_device_idx, xp):
        """Test that missing nmodes raises ValueError when intmat is not provided"""
        with self.assertRaises(ValueError):
            Intmat(nmodes=None, nslopes=5, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_init_without_intmat_and_missing_nslopes_raises(self, target_device_idx, xp):
        """Test that missing nslopes raises ValueError when intmat is not provided"""
        with self.assertRaises(ValueError):
            Intmat(nmodes=3, nslopes=None, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        mat = xp.ones((4, 4))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        new_mat = xp.full((4, 4), 7.0)
        intmat.set_value(new_mat)
        assert xp.allclose(intmat.get_value(), new_mat)

    @cpu_and_gpu
    def test_set_value_shape_mismatch_raises(self, target_device_idx, xp):
        mat = xp.ones((3, 3))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            intmat.set_value(xp.ones((2, 2)))

    @cpu_and_gpu
    def test_reduce_size_and_slopes_and_set_start_mode(self, target_device_idx, xp):
        mat = xp.arange(30).reshape(6, 5)
        intmat = Intmat(mat, target_device_idx=target_device_idx)

        # Reduce modes
        intmat.reduce_size(2)
        assert intmat.intmat.shape == (6, 3)

        # Reduce slopes
        intmat.reduce_slopes(1)
        assert intmat.intmat.shape == (5, 3)

        # Set start mode
        intmat.set_start_mode(1)
        assert intmat.intmat.shape == (5, 2)

    @cpu_and_gpu
    def test_reduce_size_and_slopes_raises(self, target_device_idx, xp):
        mat = xp.ones((5, 5))
        intmat = Intmat(mat, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            intmat.reduce_size(5)
        with self.assertRaises(ValueError):
            intmat.reduce_slopes(5)
        with self.assertRaises(ValueError):
            intmat.set_start_mode(5)

    @cpu_and_gpu
    def test_nmodes_and_nslopes_properties(self, target_device_idx, xp):
        mat = xp.zeros((7, 9))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        assert intmat.nmodes == 9
        assert intmat.nslopes == 7

    @cpu_and_gpu
    def test_generate_rec_and_pseudo_invert(self, target_device_idx, xp):
        mat = xp.eye(5)
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        rec = intmat.generate_rec()
        assert isinstance(rec, Recmat)
        assert xp.allclose(rec.recmat, xp.linalg.pinv(mat))

    @cpu_and_gpu
    def test_build_from_slopes(self, target_device_idx, xp):
        times = [0, 1, 2]
        slopes = {
            t: xp.array([1.0, 2.0, 3.0]) for t in times
        }
        disturbance = {
            t: xp.array([1.0, -1.0, 1.0]) for t in times
        }
        im = Intmat.build_from_slopes(slopes, disturbance)
        assert isinstance(im, Intmat)
        assert im.intmat.shape == (3, 3)
        assert xp.all(im.intmat[:, 0] != 0)


class TestIntmatViews(unittest.TestCase):
    """Unit tests for Intmat.modes and Intmat.slopes"""

    # ========================================================
    # BASIC READ TESTS
    # ========================================================
    @cpu_and_gpu
    def test_read_views(self, target_device_idx, xp):
        test_cases = [
            ("modes", 1),
            ("modes", slice(1, 3)),
            ("modes", [0, 2]),
            ("slopes", 2),
            ("slopes", slice(0, 2)),
            ("slopes", [0, 2]),
        ]

        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr, key in test_cases:
            with self.subTest(view=view_attr, key=key):
                view = getattr(intmat_obj, view_attr)
                result = view[key]

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(result), cpuArray(expected))

    # ========================================================
    # BASIC WRITE TESTS
    # ========================================================
    @cpu_and_gpu
    def test_write_views(self, target_device_idx, xp):
        test_cases = [
            ("modes", 1, [10, 20, 30]),
            ("modes", slice(1, 3), [[1, 2], [3, 4], [5, 6]]),
            ("modes", [0, 3], [[11, 12], [21, 22], [31, 32]]),
            ("slopes", 2, [100, 200, 300, 400]),
            ("slopes", slice(0, 2), [[9, 9, 9, 9], [8, 8, 8, 8]]),
            ("slopes", [0, 2], [[5, 5, 5, 5], [7, 7, 7, 7]]),
        ]

        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr, key, value in test_cases:
            with self.subTest(view=view_attr, key=key):
                view = getattr(intmat_obj, view_attr)
                view[key] = value

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(expected), cpuArray(value))

    # ========================================================
    # EDGE CASE TESTS
    # ========================================================
    @cpu_and_gpu
    def test_negative_indices(self, target_device_idx, xp):
        test_cases = [
            ("modes", -1),
            ("modes", [-1, -2]),
            ("slopes", -1),
            ("slopes", [-1, -3]),
        ]

        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr, key in test_cases:
            with self.subTest(view=view_attr, key=key):
                view = getattr(intmat_obj, view_attr)
                result = view[key]

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(result), cpuArray(expected))

    @cpu_and_gpu
    def test_scalar_assignment(self, target_device_idx, xp):
        test_cases = [
            ("modes", 1, 42),
            ("modes", slice(1, 3), 99),
            ("slopes", 0, 7),
            ("slopes", slice(0, 2), 5),
        ]

        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr, key, value in test_cases:
            with self.subTest(view=view_attr, key=key):
                view = getattr(intmat_obj, view_attr)
                view[key] = value

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(expected), cpuArray(value))


    @cpu_and_gpu
    def test_numpy_array_indexing(self, target_device_idx, xp):
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr in ["modes", "slopes"]:
            with self.subTest(view=view_attr):
                view = getattr(intmat_obj, view_attr)
                key = np.array([0, 2])
                result = view[key]

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(result), cpuArray(expected))

    @cpu_and_gpu
    def test_numpy_array_assignment(self, target_device_idx, xp):
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr in ["modes", "slopes"]:
            with self.subTest(view=view_attr):
                view = getattr(intmat_obj, view_attr)
                key = np.array([0, 2])

                # Adjust shape based on view
                if view_attr == "slopes":
                    value = np.ones((2, intmat_obj.intmat.shape[1]))
                else:
                    value = np.ones((intmat_obj.intmat.shape[0], 2))

                view[key] = value

                if view_attr == "modes":
                    expected = intmat_obj.intmat[:, key]
                else:
                    expected = intmat_obj.intmat[key, :]

                np.testing.assert_array_equal(cpuArray(expected), cpuArray(value))

    @cpu_and_gpu
    def test_empty_indexing(self, target_device_idx, xp):
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        for view_attr in ["modes", "slopes"]:
            with self.subTest(view=view_attr):
                view = getattr(intmat_obj, view_attr)

                # Empty list
                result = view[[]]
                self.assertEqual(result.size, 0)

                # Empty slice
                result = view[slice(0, 0)]
                self.assertEqual(result.size, 0)

    # ========================================================
    # TEST set_nmodes
    # ========================================================
    @cpu_and_gpu
    def test_set_nmodes_increase(self, target_device_idx, xp):
        """When increasing nmodes, new columns should be zero-initialized."""
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        old_nmodes = intmat_obj.intmat.shape[1]
        intmat_obj.set_nmodes(6)

        # Check new shape
        self.assertEqual(intmat_obj.intmat.shape, (3, 6))

        # Check that the added columns are zero
        np.testing.assert_array_equal(
            cpuArray(intmat_obj.intmat[:, old_nmodes:]),
            np.zeros((3, 6 - old_nmodes))
        )

    @cpu_and_gpu
    def test_set_nmodes_decrease(self, target_device_idx, xp):
        """When decreasing nmodes, columns should be truncated."""
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        intmat_obj.set_nmodes(2)

        # Check new shape
        self.assertEqual(intmat_obj.intmat.shape, (3, 2))

        # Validate that the first columns remain unchanged
        expected = np.arange(1, 13).reshape(3, 4)[:, :2]
        np.testing.assert_array_equal(cpuArray(intmat_obj.intmat), expected)

    # ========================================================
    # TEST set_nslopes
    # ========================================================
    @cpu_and_gpu
    def test_set_nslopes_increase(self, target_device_idx, xp):
        """When increasing nslopes, new rows should be zero-initialized."""
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        old_nslopes = intmat_obj.intmat.shape[0]
        intmat_obj.set_nslopes(5)

        # Check new shape
        self.assertEqual(intmat_obj.intmat.shape, (5, 4))

        # Check that the added rows are zero
        np.testing.assert_array_equal(
            cpuArray(intmat_obj.intmat[old_nslopes:, :]),
            np.zeros((5 - old_nslopes, 4))
        )

    @cpu_and_gpu
    def test_set_nslopes_decrease(self, target_device_idx, xp):
        """When decreasing nslopes, rows should be truncated."""
        intmat_obj = Intmat(intmat=xp.arange(1, 13).reshape(3, 4), target_device_idx=target_device_idx)
        intmat_obj.set_nslopes(2)

        # Check new shape
        self.assertEqual(intmat_obj.intmat.shape, (2, 4))

        # Validate that the first rows remain unchanged
        expected = np.arange(1, 13).reshape(3, 4)[:2, :]
        np.testing.assert_array_equal(cpuArray(intmat_obj.intmat), expected)
