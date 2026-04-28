
import unittest
import tempfile
import os
import numpy as np
from astropy.io import fits

from specula.scalar_values import (
    _BaseScalarValue,
    IntValue,
    FloatValue,
    StringValue,
)


class TestScalarValues(unittest.TestCase):

    # -------------------------
    # Initialization
    # -------------------------

    def test_int_value_initialization(self):
        v = IntValue(value=10, description="integer")
        self.assertEqual(v.get_value(), 10)
        self.assertEqual(v.description, "integer")
        self.assertIs(v.type, int)

    def test_float_value_initialization(self):
        v = FloatValue(value=3.14)
        self.assertEqual(v.get_value(), 3.14)
        self.assertIs(v.type, float)

    def test_string_value_initialization(self):
        v = StringValue(value="hello")
        self.assertEqual(v.get_value(), "hello")
        self.assertIs(v.type, str)

    # -------------------------
    # Type validation
    # -------------------------

    def test_int_value_rejects_wrong_type(self):
        with self.assertRaises(AssertionError):
            IntValue(value="not an int")

    def test_float_value_rejects_wrong_type(self):
        with self.assertRaises(AssertionError):
            FloatValue(value="not a float")

    def test_string_value_rejects_wrong_type(self):
        with self.assertRaises(AssertionError):
            StringValue(value=123)

    # -------------------------
    # set_value()
    # -------------------------

    def test_set_value_updates_correctly(self):
        v = IntValue(value=1)
        v.set_value(42)
        self.assertEqual(v.get_value(), 42)

    def test_set_value_rejects_invalid_type(self):
        v = IntValue(value=1)
        with self.assertRaises(AssertionError):
            v.set_value("wrong")

    # -------------------------
    # array_for_display()
    # -------------------------

    def test_array_for_display_returns_value(self):
        v = FloatValue(value=2.5)
        self.assertEqual(v.array_for_display(), 2.5)

    # -------------------------
    # FITS header
    # -------------------------

    def test_get_fits_header_contains_metadata(self):
        v = StringValue(value="abc")
        hdr = v.get_fits_header()

        self.assertEqual(hdr["VERSION"], 1)
        self.assertEqual(hdr["OBJ_TYPE"], "StringValue")

    # -------------------------
    # save() / restore()
    # -------------------------

    def test_save_and_restore_int_value(self):
        v = IntValue(value=123)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scalar.fits")

            v.save(file_path)

            self.assertTrue(os.path.exists(file_path))

            restored = IntValue.restore(file_path)

            self.assertEqual(restored.get_value(), 123)
            self.assertIsInstance(restored, IntValue)

    def test_save_writes_correct_fits_header(self):
        v = IntValue(value=99)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "scalar.fits")

            v.save(file_path)
            hdr = fits.getheader(file_path)

            self.assertEqual(hdr["VALUE"], "99")
            self.assertEqual(hdr["OBJ_TYPE"], "IntValue")

    def test_restore_missing_value_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "bad.fits")

            hdr = fits.Header()
            hdr["VERSION"] = 1

            fits.writeto(file_path, data=np.array([0, 0]), header=hdr, overwrite=True)

            with self.assertRaises(ValueError):
                _BaseScalarValue.restore(file_path)


