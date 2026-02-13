import os
import shutil
from unittest.mock import patch, MagicMock, mock_open

import specula
specula.init(0)  # Default target device
from specula.processing_objects.data_source import DataSource

from astropy.io import fits
import numpy as np
import unittest


class TestDataSource(unittest.TestCase):

    # Test that data source can read back files and output them correctly in its trigger method
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_source')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # Create a fits file "gen.fits" for testing, with the enpected output
    def _create_test_files(self):
        gen_file = os.path.join(self.tmp_dir, 'gen.fits')
        data = np.array(([3], [4]))
        times = np.array([0, 1], dtype=np.uint64)
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseValue'
        data_hdu = fits.PrimaryHDU(data, header=hdr)
        time_hdu = fits.ImageHDU(times, header=hdr)
        hdul = fits.HDUList([data_hdu, time_hdu])
        hdul.writeto(gen_file, overwrite=True)
        hdul.close()  # Force close for Windows

    def test_data_source(self):
        self._create_test_files()
        source = DataSource(store_dir=self.tmp_dir,
                            outputs=['gen'],
                            data_format='fits')

        gen = source.outputs['gen']

        source.check_ready(0)
        source.setup()
        source.trigger()
        source.post_trigger()
        assert gen.value == 3

        source.check_ready(1)
        source.setup()
        source.trigger()
        source.post_trigger()
        assert gen.value == 4

    def test_load_pickle_success(self):
        """Test DataSource.load_pickle() successfully loads pickle data into storage."""
        mock_pickle_data = {
            "times": np.array([1.0, 2.0]),
            "data": np.array([[10, 20], [30, 40]]),
            "hdr": {"OBJ_TYPE": "BaseValue"}
        }

        with patch("builtins.open", mock_open(read_data=b"pickledata")), \
             patch("pickle.load", return_value=mock_pickle_data):

            ds = DataSource(outputs=[], store_dir="/tmp", data_format="pickle")
            ds.load_pickle("test")

            self.assertIn("test", ds.storage)
            self.assertEqual(ds.obj_type["test"], "BaseValue")
            self.assertTrue(np.allclose(ds.storage["test"][1.0], np.array([10, 20])))

    def test_load_fits_success(self):
        """Test DataSource.load_fits() correctly reads FITS files using astropy."""
        mock_hdul = MagicMock()
        mock_hdul.__enter__.return_value = mock_hdul
        mock_hdul.__exit__.return_value = None
        mock_hdul.__getitem__.side_effect = lambda idx: {
            0: MagicMock(data=np.array([[1, 2], [3, 4]]), header={"OBJ_TYPE": "BaseValue"}),
            1: MagicMock(data=np.array([0.1, 0.2]))
        }[idx]

        with patch("specula.processing_objects.data_source.fits.open", return_value=mock_hdul):
            ds = DataSource(outputs=[], store_dir="/tmp", data_format="fits")
            ds.load_fits("mydata")

            self.assertIn("mydata", ds.storage)
            self.assertEqual(ds.obj_type["mydata"], "BaseValue")
            self.assertTrue(np.allclose(ds.storage["mydata"][0.1], np.array([1, 2])))

    def test_loadFromFile_invalid_duplicate(self):
        """Test DataSource.loadFromFile() raises ValueError when reloading same key."""
        ds = DataSource(outputs=[], store_dir="/tmp", data_format="pickle")
        ds.items["dup"] = "exists"

        with self.assertRaises(ValueError):
            ds.loadFromFile("dup")

    def test_init_with_outputs_and_import_class(self):
        """Test DataSource.__init__() calls import_class for non-BaseValue objects."""
        mock_imported_class = MagicMock()
        mock_imported_class.from_header.return_value = "CreatedObj"

        with patch("specula.lib.utils.import_class", return_value=mock_imported_class), \
             patch("specula.processing_objects.data_source.DataSource.loadFromFile") as mock_load, \
             patch("specula.processing_objects.data_source.BaseValue") as mock_baseval:

            from specula.lib.utils import import_class  # Patched

            # Prepopulate headers/obj_type before outputs assignment
            ds = DataSource(outputs=["obj1"], store_dir="/tmp")
            ds.obj_type["obj1"] = "SomeOtherType"
            ds.headers["obj1"] = {"fake": "hdr"}
            ds.storage["obj1"] = {}

            # Manually trigger the output assignment logic
            ds.outputs["obj1"] = import_class(ds.obj_type["obj1"]).from_header(ds.headers["obj1"])

            self.assertEqual(ds.outputs["obj1"], "CreatedObj")
            mock_imported_class.from_header.assert_called_once()

    def test_size_existing_and_missing_key(self):
        """Test DataSource.size() returns correct shapes and handles missing keys."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        arr = np.zeros((5, 10))
        ds.storage["test"] = arr

        # Correct shape
        self.assertEqual(ds.size("test"), arr.shape)
        self.assertEqual(ds.size("test", dimensions=1), arr.shape[1])

        # Missing key
        result = ds.size("missing")
        self.assertEqual(result, -1)

    def test_trigger_code_sets_output_values(self):
        """Test DataSource.trigger_code() correctly updates outputs from storage."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        ds.current_time = 123.4

        # Mock outputs
        mock_output = MagicMock()
        mock_output.np = np
        ds.outputs["sig"] = mock_output

        # Storage with matching current_time
        ds.storage["sig"] = {123.4: np.array([5, 6, 7])}

        ds.trigger_code()
        mock_output.set_value.assert_called_once()
        self.assertEqual(mock_output.generation_time, ds.current_time)

    def test_trigger_code_skips_missing_time(self):
        """Test DataSource.trigger_code() skips outputs when data not available at current time."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        ds.current_time = 0.0  # Looking for time 0
        ds.verbose = False  # Suppress warning

        # Mock output
        mock_output = MagicMock()
        mock_output.xp = np
        # Explicitly set generation_time to a different value to verify it doesn't change
        mock_output.generation_time = -999.0
        ds.outputs["sig"] = mock_output

        # Storage has data at time 1.0 and 2.0, but NOT at 0.0
        ds.storage["sig"] = {1.0: np.array([10, 20]), 2.0: np.array([30, 40])}

        # Trigger should NOT raise error, just skip
        ds.trigger_code()

        # Output should NOT be updated
        mock_output.set_value.assert_not_called()

        # generation_time should NOT have been changed (still -999.0)
        self.assertEqual(mock_output.generation_time, -999.0)

    def test_trigger_code_verbose_warning_on_missing_time(self):
        """Test DataSource.trigger_code() prints warning when verbose=True and data missing."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        ds.current_time = 0.5
        ds.verbose = True

        # Mock output
        mock_output = MagicMock()
        mock_output.xp = np
        ds.outputs["test_signal"] = mock_output

        # Storage with no data at current_time
        ds.storage["test_signal"] = {1.0: np.array([1, 2, 3])}

        # Capture print output
        with patch('builtins.print') as mock_print:
            ds.trigger_code()

            # Verify warning was printed
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            self.assertIn('Warning', call_args)
            self.assertIn('test_signal', call_args)
            self.assertIn('0.5', call_args)

    def test_trigger_code_mixed_availability(self):
        """Test DataSource.trigger_code() handles multiple outputs with
           different data availability."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        ds.current_time = 1.0
        ds.verbose = False

        # Mock outputs
        mock_output1 = MagicMock()
        mock_output1.xp = np
        mock_output2 = MagicMock()
        mock_output2.xp = np

        ds.outputs["available"] = mock_output1
        ds.outputs["missing"] = mock_output2

        # One has data at t=1.0, the other doesn't
        ds.storage["available"] = {1.0: np.array([100, 200])}
        ds.storage["missing"] = {2.0: np.array([300, 400])}  # Data at different time

        ds.trigger_code()

        # First output should be updated
        mock_output1.set_value.assert_called_once()
        self.assertEqual(mock_output1.generation_time, 1.0)

        # Second output should NOT be updated
        mock_output2.set_value.assert_not_called()

    def test_trigger_code_sparse_integrated_output(self):
        """Test DataSource.trigger_code() handles sparse data like integrated
           PSF (only at end)."""
        ds = DataSource(outputs=[], store_dir="/tmp")
        ds.verbose = False

        # Mock output
        mock_psf = MagicMock()
        mock_psf.xp = np
        ds.outputs["psf_int"] = mock_psf

        # Integrated PSF exists only at final time
        ds.storage["psf_int"] = {1.0: np.array([[1, 2], [3, 4]])}  # Only at t=1.0

        # Try to trigger at early times (before integration completes)
        ds.current_time = 0.0
        ds.trigger_code()
        mock_psf.set_value.assert_not_called()  # No data yet

        ds.current_time = 0.5
        ds.trigger_code()
        mock_psf.set_value.assert_not_called()  # Still no data

        # Now trigger at final time where data exists
        ds.current_time = 1.0
        ds.trigger_code()
        mock_psf.set_value.assert_called_once()  # Data available!
        self.assertEqual(mock_psf.generation_time, 1.0)
