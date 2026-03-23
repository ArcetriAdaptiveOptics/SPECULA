import os
import tempfile
from astropy.io import fits
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray, np
from specula.data_objects.spatio_temp_array import SpatioTempArray

from test.specula_testlib import cpu_and_gpu


class TestSpatioTempArray(unittest.TestCase):
    """Unit tests for SpatioTempArray data object."""

    @cpu_and_gpu
    def test_creation_3d(self, target_device_idx, xp):
        """Test creation of 3D spatio-temporal array."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        self.assertEqual(sta.array.shape, (5, 10, 10))
        self.assertEqual(sta.time_vector.shape, (5,))
        np.testing.assert_array_almost_equal(cpuArray(sta.get_value()), np.moveaxis(array_3d, -1, 0))
        np.testing.assert_array_almost_equal(cpuArray(sta.get_time_vector()), time_vec)

    @cpu_and_gpu
    def test_creation_2d(self, target_device_idx, xp):
        """Test creation of 2D spatio-temporal array."""
        array_2d = np.random.rand(20, 8)
        time_vec = np.linspace(0, 1.0, 8)

        sta = SpatioTempArray(array_2d, time_vec,
                              target_device_idx=target_device_idx)

        self.assertEqual(sta.array.shape, (8, 20))
        self.assertEqual(sta.time_vector.shape, (8,))

    @cpu_and_gpu
    def test_creation_1d_time_only(self, target_device_idx, xp):
        """Test creation of 1D array (time values only)."""
        array_1d = np.array([1.0, 2.0, 3.0, 4.0])
        time_vec = np.array([0.0, 0.5, 1.0, 1.5])

        sta = SpatioTempArray(array_1d, time_vec,
                              target_device_idx=target_device_idx)

        self.assertEqual(sta.array.shape, (4,))
        self.assertEqual(sta.time_vector.shape, (4,))

    @cpu_and_gpu
    def test_shape_validation_error(self, target_device_idx, xp):
        """Test that shape mismatch raises ValueError."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.array([0.0, 0.1, 0.2])  # Wrong length!

        with self.assertRaises(ValueError) as context:
            SpatioTempArray(array_3d, time_vec,
                            target_device_idx=target_device_idx)

        self.assertIn("Selected temporal dimension", str(context.exception))

    @cpu_and_gpu
    def test_set_value(self, target_device_idx, xp):
        """Test setting array values in-place."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        new_array = np.ones((10, 10, 5)) * 2.5
        sta.set_value(new_array)

        np.testing.assert_array_almost_equal(cpuArray(sta.get_value()), np.moveaxis(new_array, -1, 0))

    @cpu_and_gpu
    def test_set_time_vector(self, target_device_idx, xp):
        """Test setting time vector values in-place."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        new_time_vec = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        sta.set_time_vector(new_time_vec)

        np.testing.assert_array_almost_equal(cpuArray(sta.get_time_vector()),
                                             new_time_vec)

    @cpu_and_gpu
    def test_array_for_display(self, target_device_idx, xp):
        """Test array_for_display returns correct data."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.linspace(0, 1.0, 5)

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        display_array = sta.array_for_display()
        np.testing.assert_array_almost_equal(cpuArray(display_array),
                             np.moveaxis(array_3d, -1, 0))

    @cpu_and_gpu
    def test_save_restore(self, target_device_idx, xp):
        """Test save and restore from FITS file."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.linspace(0.0, 0.4, 5)

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_sta.fits')
            sta.save(fname)

            # Restore on same device
            sta_restored = SpatioTempArray.restore(fname,
                                                   target_device_idx=target_device_idx)

            np.testing.assert_array_almost_equal(cpuArray(sta_restored.get_value()),
                                                 np.moveaxis(array_3d, -1, 0))
            np.testing.assert_array_almost_equal(cpuArray(sta_restored.get_time_vector()),
                                                 time_vec)

    def test_from_header(self):
        """Test from_header creates object with correct shape."""

        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.linspace(0.0, 0.4, 5)

        sta = SpatioTempArray(array_3d, time_vec)
        hdr = sta.get_fits_header()

        # Verify header uses FITS-compliant keyword names
        self.assertIn('VERSION', hdr)
        self.assertIn('OBJ_TYPE', hdr)
        self.assertIn('ARSHAPE', hdr)  # Should be 8 chars or less (formerly ARRAY_SHAPE)
        self.assertIn('NTIME', hdr)

        # Create new object from header
        sta_from_header = SpatioTempArray.from_header(hdr, target_device_idx=-1)

        # Check shapes are correct
        self.assertEqual(sta_from_header.array.shape, (5, 10, 10))
        self.assertEqual(sta_from_header.time_vector.shape, (5,))

    def test_from_header_invalid_version(self):
        """Test from_header raises error for invalid version."""

        hdr = fits.Header()
        hdr['VERSION'] = 999
        hdr['ARSHAPE'] = '10 10 5'
        hdr['NTIME'] = 5

        with self.assertRaises(ValueError) as context:
            SpatioTempArray.from_header(hdr)

        self.assertIn("Unknown version", str(context.exception))

    def test_from_header_missing_metadata(self):
        """Test from_header raises error when metadata is missing."""

        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['ARSHAPE'] = '10 10 5'
        # Missing NTIME

        with self.assertRaises(ValueError) as context:
            SpatioTempArray.from_header(hdr)

        self.assertIn("Missing", str(context.exception))

    @cpu_and_gpu
    def test_generation_time_attribute(self, target_device_idx, xp):
        """Test that generation_time attribute is inherited from BaseDataObj."""
        array_3d = np.random.rand(10, 10, 5)
        time_vec = np.linspace(0, 1.0, 5)

        sta = SpatioTempArray(array_3d, time_vec,
                              target_device_idx=target_device_idx)

        # Should have generation_time from BaseDataObj
        self.assertEqual(sta.generation_time, -1)

        # Should be settable
        sta.generation_time = 1e9
        self.assertEqual(sta.generation_time, 1e9)

    @cpu_and_gpu
    def test_4d_array(self, target_device_idx, xp):
        """Test creation of 4D spatio-temporal array."""
        array_4d = np.random.rand(8, 8, 8, 10)
        time_vec = np.linspace(0, 1.0, 10)

        sta = SpatioTempArray(array_4d, time_vec,
                              target_device_idx=target_device_idx)

        self.assertEqual(sta.array.shape, (10, 8, 8, 8))
        self.assertEqual(sta.time_vector.shape, (10,))

    @cpu_and_gpu
    def test_creation_time_first_input(self, target_device_idx, xp):
        """Test that time-first input is accepted without axis move."""
        array_3d_tf = np.random.rand(5, 10, 10)
        time_vec = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

        sta = SpatioTempArray(array_3d_tf, time_vec, time_axis=0,
                              target_device_idx=target_device_idx)

        self.assertEqual(sta.array.shape, (5, 10, 10))
        np.testing.assert_array_almost_equal(cpuArray(sta.get_value()), array_3d_tf)
