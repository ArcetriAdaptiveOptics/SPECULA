import types
import unittest
from unittest.mock import patch, MagicMock

import specula
specula.init(0)  # Default target device

from specula.lib.utils import import_class


class TestImportClass(unittest.TestCase):
    """
    Unit tests for the `import_class` function in specula.lib.utils
    (defined in iteras.lib.utils, which searches iteras's packages first,
    then the ones registered by plugins like specula).
    
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
        default module (iteras.processing_objects).

        This test mocks `camelcase_to_snakecase` to return a fixed module name,
        and mocks `importlib.import_module` to return a module containing the
        expected class. The test asserts that the returned class is correct and
        that both mocks are called as expected.
        """
        with patch("iteras.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("iteras.lib.utils.importlib.import_module") as mock_import:

            # Fake module + class
            mock_module = MagicMock()
            mock_class = type("MyClass", (), {})
            setattr(mock_module, "MyClass", mock_class)
            mock_import.return_value = mock_module

            result = import_class("MyClass")

            self.assertEqual(result, mock_class)
            self.assertTrue(mock_snake.called)
            mock_snake.assert_called_with("MyClass")
            # All packages are searched, starting from the first one. The same
            # class found in several packages is not ambiguous.
            self.assertEqual(mock_import.call_args_list[0].args, ("iteras.processing_objects.my_class",))
            mock_import.assert_any_call("specula.processing_objects.my_class")

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
        with patch("iteras.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("iteras.lib.utils.importlib.import_module") as mock_import:

            def side_effect(module_path):
                if module_path == "custom.module.my_class":
                    mock_module = MagicMock()
                    mock_class = type("MyClass", (), {})
                    setattr(mock_module, "MyClass", mock_class)
                    return mock_module
                else:
                    raise ModuleNotFoundError(f"No module named '{module_path}'", name=module_path)

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
        with patch("iteras.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("iteras.lib.utils.importlib.import_module") as mock_import:

            mock_import.side_effect = ModuleNotFoundError

            with self.assertRaises(ImportError):
                import_class("MyClass")

            self.assertTrue(mock_snake.called)

    def test_fallback_to_later_module(self):
        """
        Test that `import_class` correctly falls back to later default modules
        when earlier ones are missing.

        The test simulates ModuleNotFoundError for all packages except
        specula's 'display', which provides the class. It verifies that
        the returned class is correct and that import_module was called on
        the fallback module.
        """
        with patch("iteras.lib.utils.camelcase_to_snakecase", return_value="my_class") as mock_snake, \
             patch("iteras.lib.utils.importlib.import_module") as mock_import:

            def side_effect(module_path):
                if module_path != "specula.display.my_class":
                    raise ModuleNotFoundError(f"No module named '{module_path}'", name=module_path)
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


class TestClassLookup(unittest.TestCase):
    """
    Tests of import_class() with real packages: name collisions, qualified
    names and automatic loading of the plugins declared with entry points.
    """

    def setUp(self):
        import os
        import sys
        import tempfile
        import textwrap
        from iteras import registry

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name

        def write(path, text):
            full = os.path.join(root, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, 'w') as f:
                f.write(textwrap.dedent(text))

        # Two packages defining different classes called Widget, and a third one
        # re-exporting the class of the first one (like the specula alias modules)
        write('lookup_a/__init__.py', '')
        write('lookup_a/widget.py', 'class Widget:\n    pass\n')
        write('lookup_a/gadget.py', 'class Gadget:\n    pass\n')
        write('lookup_b/__init__.py', '')
        write('lookup_b/widget.py', 'class Widget:\n    pass\n')
        write('lookup_alias/__init__.py', '')
        write('lookup_alias/gadget.py', 'from lookup_a.gadget import Gadget\n')
        write('lookup_plugin/__init__.py', '''
            from iteras import registry
            registry.register_plugin('lookup_plugin', packages=['lookup_plugin'])
            ''')
        write('lookup_plugin/sprocket.py', 'class Sprocket:\n    pass\n')

        sys.path.insert(0, root)
        self.addCleanup(sys.path.remove, root)
        self.addCleanup(lambda: [sys.modules.pop(m) for m in list(sys.modules)
                                 if m.split('.')[0].startswith('lookup_')])

        # Registrations made by a test are undone at the end
        for name in ('_packages', '_plugin_roots'):
            p = patch.object(registry, name, list(getattr(registry, name)))
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(registry, '_entry_points_loaded', True)
        p.start()
        self.addCleanup(p.stop)

    def test_ambiguous_name_raises(self):
        from iteras.lib.utils import AmbiguousClassError
        with self.assertRaises(AmbiguousClassError) as ctx:
            import_class('Widget', ['lookup_a', 'lookup_b'])
        self.assertIn('lookup_a.widget.Widget', str(ctx.exception))
        self.assertIn('lookup_b.widget.Widget', str(ctx.exception))
        # AmbiguousClassError is an ImportError
        self.assertIsInstance(ctx.exception, ImportError)

    def test_same_class_in_two_packages_is_not_ambiguous(self):
        from lookup_a.gadget import Gadget
        self.assertIs(import_class('Gadget', ['lookup_a', 'lookup_alias']), Gadget)

    def test_qualified_name_with_package(self):
        from lookup_a.widget import Widget as WidgetA
        from lookup_b.widget import Widget as WidgetB
        self.assertIs(import_class('lookup_a.Widget', ['lookup_a', 'lookup_b']), WidgetA)
        self.assertIs(import_class('lookup_b.Widget', ['lookup_a', 'lookup_b']), WidgetB)

    def test_qualified_name_with_plugin_root(self):
        # specula registers specula.processing_objects, specula.data_objects, ...
        from specula.processing_objects.integrator import Integrator
        from iteras.data_objects.sim_params import SimParams
        self.assertIs(import_class('specula.Integrator'), Integrator)
        self.assertIs(import_class('iteras.SimParams'), SimParams)
        # A qualified name restricts the search to that plugin
        with self.assertRaises(ImportError):
            import_class('iteras.Integrator')
        # Registered plugins can be used as prefix too
        __import__('lookup_plugin')
        self.assertEqual(import_class('lookup_plugin.Sprocket').__module__, 'lookup_plugin.sprocket')

    def test_qualified_name_with_module_path(self):
        from lookup_b.widget import Widget as WidgetB
        self.assertIs(import_class('lookup_b.widget.Widget'), WidgetB)
        with self.assertRaises(AttributeError):
            import_class('lookup_b.widget.NotThere')
        with self.assertRaises(ImportError):
            import_class('lookup_missing.module.Widget')

    def test_missing_packages_are_skipped(self):
        from lookup_a.gadget import Gadget
        self.assertIs(import_class('Gadget', ['lookup_missing', 'lookup_a']), Gadget)

    def test_entry_point_plugins_loaded_when_class_not_found(self):
        from iteras import registry
        ep = MagicMock()
        ep.load.side_effect = lambda: __import__('lookup_plugin')
        with patch.object(registry, '_entry_points_loaded', False), \
             patch('importlib.metadata.entry_points', return_value=[ep]):
            klass = import_class('Sprocket')
        self.assertEqual(klass.__module__, 'lookup_plugin.sprocket')
        ep.load.assert_called_once_with()

    def test_entry_point_plugins_not_loaded_when_class_found(self):
        from iteras import registry
        with patch.object(registry, '_entry_points_loaded', False), \
             patch('importlib.metadata.entry_points') as mock_eps:
            import_class('Gadget', ['lookup_a'])
        mock_eps.assert_not_called()

    def test_simul_checks_add_modules(self):
        from specula.simul import Simul
        params = {'main': {'class': 'SimParams', 'root_dir': self.tmp.name,
                           'add_modules': ['lookup_missing']}}
        with self.assertRaisesRegex(ValueError, 'lookup_missing'):
            Simul('dummy.yml').build_objects(params)

    def test_simul_with_qualified_names(self):
        from specula.simul import Simul
        from iteras.processing_objects.data_store import DataStore
        params = {'main': {'class': 'iteras.SimParams', 'root_dir': self.tmp.name},
                  'store': {'class': 'iteras.processing_objects.data_store.DataStore',
                            'store_dir': self.tmp.name}}
        simul = Simul('dummy.yml')
        self.assertTrue(simul.is_main_params(params, 'main'))
        simul.build_objects(params)
        self.assertIsInstance(simul.objs['store'], DataStore)
