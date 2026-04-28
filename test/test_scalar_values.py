import unittest
import tempfile
import os

from specula.scalar_values import IntValue, FloatValue, StringValue


class TestScalarValues(unittest.TestCase):
    """
    Unit tests for scalar value classes:
    IntValue, FloatValue, and StringValue.

    These tests verify:
    - correct type enforcement
    - getter/setter behavior
    - FITS save/restore round-trip
    """

    def test_int_value_creation_and_get_set(self):
        """
        Test IntValue initialization and value access/modification.
        """
        v = IntValue(value=5, description="test int")
        self.assertEqual(v.get_value(), 5)

        v.set_value(10)
        self.assertEqual(v.get_value(), 10)

        with self.assertRaises(AssertionError):
            v.set_value(3.14)

    def test_float_value_creation_and_get_set(self):
        """
        Test FloatValue initialization and value access/modification.
        """
        v = FloatValue(value=2.5, description="test float")
        self.assertEqual(v.get_value(), 2.5)

        v.set_value(1.25)
        self.assertEqual(v.get_value(), 1.25)

        with self.assertRaises(AssertionError):
            v.set_value("not a float")

    def test_string_value_creation_and_get_set(self):
        """
        Test StringValue initialization and value access/modification.
        """
        v = StringValue(value="hello", description="test string")
        self.assertEqual(v.get_value(), "hello")

        v.set_value("world")
        self.assertEqual(v.get_value(), "world")

        with self.assertRaises(AssertionError):
            v.set_value(123)

    def test_fits_save_and_restore_int(self):
        """
        Test FITS serialization and deserialization for IntValue.
        """
        v = IntValue(value=42, description="fitstest")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "int.fits")

            v.save(path, overwrite=True)
            restored = IntValue.restore(path)

            self.assertEqual(restored.get_value(), 42)
            self.assertEqual(restored.description, "fitstest")

    def test_fits_save_and_restore_float(self):
        """
        Test FITS serialization and deserialization for FloatValue.
        """
        v = FloatValue(value=3.14, description="pi")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "float.fits")

            v.save(path, overwrite=True)
            restored = FloatValue.restore(path)

            self.assertEqual(restored.get_value(), 3.14)

    def test_fits_save_and_restore_string(self):
        """
        Test FITS serialization and deserialization for StringValue.
        """
        v = StringValue(value="spectra", description="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "str.fits")

            v.save(path, overwrite=True)
            restored = StringValue.restore(path)

            self.assertEqual(restored.get_value(), "spectra")
            self.assertEqual(restored.description, "test")

    def test_type_enforcement(self):
        """
        Ensure type safety is enforced by set_value().
        """
        self.assertRaises(AssertionError, IntValue, value="bad")
        self.assertRaises(AssertionError, FloatValue, value="bad")
        self.assertRaises(AssertionError, StringValue, value=123)

