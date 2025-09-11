import types
import unittest
from unittest.mock import patch, MagicMock
from specula.lib.utils import import_class


class TestImportClass(unittest.TestCase):
    """
    Unit tests for the `import_class` function in specula.lib.utils.
    
    The tests verify correct behavior in all major scenarios:
    - Successful import from default modules
    - Successful import from additional user-defined modules
    - Raising AttributeError when class is missing
    - Raising ImportError when module is missing
    - Correct fallback behavior when earlier modules fail
    """

    def test_successful_import_from_first_module(self):
        """
        Test that `import_class` correctly imports a class from the first
        default module (specula.processing_objects).

        This test mocks `camelcase_to_snakecase` to return a fixed module name,
        and mocks `importlib.import_module` to return a module containing the
        expected class. The test asserts that the returned class is correct and
        that both mocks are called as expected.
        """
        with patch("specula.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("specula.lib.utils.importlib.import_module") as mock_import:

            # Fake module + class
            mock_module = MagicMock()
            mock_class = type("MyClass", (), {})
            setattr(mock_module, "MyClass", mock_class)
            mock_import.return_value = mock_module

            result = import_class("MyClass")

            self.assertEqual(result, mock_class)
            self.assertTrue(mock_snake.called)
            mock_snake.assert_called_with("MyClass")
            mock_import.assert_called_with("specula.processing_objects.my_class")

    def test_successful_import_from_additional_module(self):
        """
        Test that `import_class` can successfully import a class from a
        user-provided additional module when all default modules fail.

        The test simulates ModuleNotFoundError for the default modules
        (processing_objects, data_objects, display) and provides a mock
        module in `custom.module` that contains the target class. It asserts
        that the correct class is returned and that import_module is called
        for the additional module.
        """
        with patch("specula.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("specula.lib.utils.importlib.import_module") as mock_import:

            def side_effect(module_path):
                if module_path in [
                    "specula.processing_objects.my_class",
                    "specula.data_objects.my_class",
                    "specula.display.my_class",
                ]:
                    raise ModuleNotFoundError
                elif module_path == "custom.module.my_class":
                    mock_module = MagicMock()
                    mock_class = type("MyClass", (), {})
                    setattr(mock_module, "MyClass", mock_class)
                    return mock_module
                else:
                    raise ModuleNotFoundError

            mock_import.side_effect = side_effect

            result = import_class("MyClass", additional_modules=["custom.module"])
            self.assertEqual(result.__name__, "MyClass")
            self.assertTrue(mock_snake.called)
            mock_import.assert_any_call("custom.module.my_class")

    def test_module_not_found_raises_import_error(self):
        """
        Test that `import_class` raises ImportError when the class cannot
        be found in any module.

        This simulates ModuleNotFoundError for all attempts to import modules.
        It asserts that the function raises ImportError and that
        camelcase_to_snakecase is called.
        """
        with patch("specula.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("specula.lib.utils.importlib.import_module") as mock_import:

            mock_import.side_effect = ModuleNotFoundError

            with self.assertRaises(ImportError):
                import_class("MyClass")

            self.assertTrue(mock_snake.called)

    def test_fallback_to_later_module(self):
        """
        Test that `import_class` correctly falls back to later default modules
        when earlier ones are missing.

        The test simulates ModuleNotFoundError for 'processing_objects' and
        'data_objects', then provides a module for 'display'. It verifies that
        the returned class is correct and that import_module was called on
        the fallback module.
        """
        with patch("specula.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("specula.lib.utils.importlib.import_module") as mock_import:

            def side_effect(module_path):
                if module_path in [
                    "specula.processing_objects.my_class",
                    "specula.data_objects.my_class",
                ]:
                    raise ModuleNotFoundError
                mock_module = MagicMock()
                mock_class = type("MyClass", (), {})
                setattr(mock_module, "MyClass", mock_class)
                return mock_module

            mock_import.side_effect = side_effect

            result = import_class("MyClass")
            self.assertEqual(result.__name__, "MyClass")
            self.assertTrue(mock_snake.called)
            mock_import.assert_any_call("specula.display.my_class")

    @patch("importlib.import_module")
    def test_module_not_found_for_dependency_inside_module(self, mock_import):
        """If a dependency inside the module is missing, the error should be re-raised."""
        # Simulate failure due to missing 'numpy' while loading target module
        mock_import.side_effect = ModuleNotFoundError("No module named 'numpy'")

        with self.assertRaises(ModuleNotFoundError) as ctx:
            import_class("FakeClass")

        self.assertIn("numpy", str(ctx.exception))  # ensure dependency error bubbles up

    @patch("importlib.import_module")
    def test_class_not_in_module_raises_attribute_error(self, mock_import):
        """If module loads but class is missing, raise AttributeError."""
        fake_module = types.SimpleNamespace()
        mock_import.return_value = fake_module  # empty module, no classes

        with self.assertRaises(AttributeError) as ctx:
            import_class("FakeClass")

        self.assertIn("Class FakeClass not found", str(ctx.exception))

    @patch("importlib.import_module")
    def test_successful_import(self, mock_import):
        """Successfully imports class if it exists in module."""
        class Dummy:
            pass
        fake_module = types.SimpleNamespace(Dummy=Dummy)
        mock_import.return_value = fake_module

        result = import_class("Dummy")
        self.assertIs(result, Dummy)
