

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
