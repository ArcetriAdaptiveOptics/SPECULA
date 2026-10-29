'''
Compatibility of specula with the iteras framework: the modules moved to
iteras are still importable from their old specula paths, the values set
by init() are visible through specula, and SPECULA is registered as an
iteras plugin.
'''
import specula
specula.init(0)  # Default target device

import importlib
import numpy as np
import os
import unittest
from unittest.mock import patch, MagicMock

import iteras
from iteras import registry
from iteras.lib.utils import import_class

# Old specula module -> iteras module
MOVED_MODULES = {
    'specula.base_time_obj': 'iteras.base_time_obj',
    'specula.base_data_obj': 'iteras.base_data_obj',
    'specula.base_processing_obj': 'iteras.base_processing_obj',
    'specula.base_value': 'iteras.base_value',
    'specula.scalar_values': 'iteras.scalar_values',
    'specula.connections': 'iteras.connections',
    'specula.loop_control': 'iteras.loop_control',
    'specula.simul': 'iteras.simul',
    'specula.simul_diagram': 'iteras.simul_diagram',
    'specula.calib_manager': 'iteras.calib_manager',
    'specula.tracing': 'iteras.tracing',
    'specula.template_processing_obj': 'iteras.template_processing_obj',
    'specula.base_replay_analyser': 'iteras.base_replay_analyser',
    'specula.processing_objects.data_store': 'iteras.processing_objects.data_store',
    'specula.processing_objects.data_source': 'iteras.processing_objects.data_source',
    'specula.processing_objects.data_buffer': 'iteras.processing_objects.data_buffer',
    'specula.processing_objects.data_print': 'iteras.processing_objects.data_print',
    'specula.processing_objects.display_server': 'iteras.processing_objects.display_server',
    'specula.processing_objects.terminal_input': 'iteras.processing_objects.terminal_input',
    'specula.display.base_display': 'iteras.display.base_display',
    'specula.display.display_process': 'iteras.display.display_process',
    'specula.display.data_plotter': 'iteras.display.data_plotter',
    'specula.display.plot_display': 'iteras.display.plot_display',
    'specula.display.plot_vector_display': 'iteras.display.plot_vector_display',
    'specula.display.display_recorder': 'iteras.display.display_recorder',
    'specula.lib.display_server_api': 'iteras.lib.display_server_api',
}


class TestMovedModules(unittest.TestCase):

    def test_old_paths_are_the_iteras_modules(self):
        for old, new in MOVED_MODULES.items():
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_classes_are_the_same_objects(self):
        from specula.base_processing_obj import BaseProcessingObj as OldBase
        from iteras.base_processing_obj import BaseProcessingObj as NewBase
        from specula.base_value import BaseValue
        self.assertIs(OldBase, NewBase)
        self.assertIsInstance(BaseValue(), importlib.import_module('iteras.base_value').BaseValue)

    def test_patching_old_path_patches_iteras(self):
        with patch('specula.connections.cpuArray', 'patched'):
            self.assertEqual(importlib.import_module('iteras.connections').cpuArray, 'patched')

    def test_synonyms(self):
        from specula.processing_objects.specula_input import SpeculaInput
        from iteras.processing_objects.interactive_input import InteractiveInput
        from specula.log import get_specula_logger, SpeculaLogAdapter
        from iteras.log import get_iteras_logger, IterasLogAdapter
        self.assertIs(SpeculaInput, InteractiveInput)
        self.assertIs(get_specula_logger, get_iteras_logger)
        self.assertIs(SpeculaLogAdapter, IterasLogAdapter)

    def test_generic_utils_reexported(self):
        from specula.lib import utils as specula_utils
        from iteras.lib import utils as iteras_utils
        for name in ['camelcase_to_snakecase', 'import_class', 'get_type_hints',
                     'make_tn', 'flatten', 'resolve_type']:
            self.assertIs(getattr(specula_utils, name), getattr(iteras_utils, name))


class TestRuntimeValues(unittest.TestCase):

    def test_values_set_by_init_are_forwarded(self):
        for name in ['cp', 'xp', 'global_precision', 'default_target_device_idx',
                     'float_dtype', 'complex_dtype', 'array_types', 'process_rank',
                     'process_comm', 'gpuEnabled']:
            with self.subTest(name=name):
                self.assertIs(getattr(specula, name), getattr(iteras, name))

    def test_forwarded_values_follow_init(self):
        orig_device, orig_precision = iteras.default_target_device_idx, iteras.global_precision
        new_precision = 1 - orig_precision
        try:
            specula.init(orig_device, precision=new_precision)
            self.assertEqual(specula.global_precision, new_precision)
            from specula import global_precision
            self.assertEqual(global_precision, new_precision)
        finally:
            specula.init(orig_device, precision=orig_precision)
        self.assertEqual(specula.global_precision, orig_precision)

    def test_functions(self):
        self.assertIs(specula.init, iteras.init)
        self.assertIs(specula.cpuArray, iteras.cpuArray)
        self.assertIs(specula.main_simul, iteras.main_simul)

    def test_fuse_decorated_before_init(self):
        # The fused kernel is created at the first GPU call, so functions can
        # be decorated before init() (here, with cupy not yet set)
        with patch('iteras.cp', None):
            @iteras.fuse(kernel_name="add_one")
            def add_one(x, xp):
                return x + 1
        x = np.arange(3.0)
        np.testing.assert_array_equal(add_one(x, xp=np), x + 1)
        if iteras.cp is not None:
            y = iteras.cp.arange(3.0)
            np.testing.assert_array_equal(iteras.cp.asnumpy(add_one(y, xp=iteras.cp)), x + 1)

    def test_unknown_attribute(self):
        with self.assertRaises(AttributeError):
            _ = specula.not_an_attribute


class TestSpeculaPlugin(unittest.TestCase):

    def test_registered(self):
        self.assertIn('specula', registry.plugin_roots())
        self.assertIn('specula.processing_objects', registry.search_packages())

    def test_import_class_finds_iteras_and_specula_classes(self):
        self.assertEqual(import_class('DataStore').__module__, 'iteras.processing_objects.data_store')
        self.assertEqual(import_class('SimParams').__module__, 'iteras.data_objects.sim_params')
        self.assertEqual(import_class('SimulParams').__module__, 'specula.data_objects.simul_params')
        self.assertIs(import_class('SpeculaInput'), import_class('InteractiveInput'))

    def test_import_class_unknown(self):
        with self.assertRaises(ImportError):
            import_class('NotAnExistingClass')

    def test_logger_names(self):
        from specula.data_objects.simul_params import SimulParams
        from iteras.data_objects.sim_params import SimParams
        from iteras.base_data_obj import BaseDataObj

        class LocalObj(BaseDataObj):
            pass

        self.assertEqual(SimulParams().logger.logger.name, 'specula.SimulParams')
        self.assertEqual(SimParams().logger.logger.name, 'iteras.SimParams')
        # Classes outside registered plugins log under their own top-level package
        root = LocalObj.__module__.partition('.')[0]
        self.assertEqual(LocalObj().logger.logger.name, f'{root}.LocalObj')

    def test_calib_subdirs(self):
        from iteras.calib_manager import CalibManager
        cm = CalibManager('/root')
        self.assertEqual(cm.filename('data', 'x'), os.path.join('/root', 'data/', 'x.fits'))
        self.assertEqual(cm.filename('Recmat', 'x'), os.path.join('/root', 'rec/', 'x.fits'))

    def test_env_aliases(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('ITERAS_DISABLE_GPU', 'SPECULA_DISABLE_GPU')}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(registry.getenv('ITERAS_DISABLE_GPU', 'FALSE'), 'FALSE')
            os.environ['SPECULA_DISABLE_GPU'] = '1'
            self.assertEqual(registry.getenv('ITERAS_DISABLE_GPU', 'FALSE'), '1')
            os.environ['ITERAS_DISABLE_GPU'] = '2'
            self.assertEqual(registry.getenv('ITERAS_DISABLE_GPU', 'FALSE'), '2')


class TestRegistry(unittest.TestCase):

    def setUp(self):
        # Registrations made by a test are undone at the end
        patches = [patch.object(registry, name, list(getattr(registry, name)))
                   for name in ('_packages', '_plugin_roots')]
        patches += [patch.object(registry, '_calib_subdirs', dict(registry._calib_subdirs)),
                    patch.object(registry, '_env_aliases', dict(registry._env_aliases))]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_register_plugin(self):
        registry.register_plugin('myplugin', packages=['myplugin.objects', 'myplugin.objects'],
                                 calib_subdirs={'MyData': 'mydata/'},
                                 env_aliases={'ITERAS_PORT': 'MYPLUGIN_PORT'})
        self.assertEqual(registry.search_packages().count('myplugin.objects'), 1)
        self.assertEqual(registry.search_packages()[-1], 'myplugin.objects')
        self.assertIn('myplugin', registry.plugin_roots())
        self.assertEqual(registry.calib_subdirs()['MyData'], 'mydata/')
        with patch.dict(os.environ, {'MYPLUGIN_PORT': '1234'}):
            os.environ.pop('ITERAS_PORT', None)
            os.environ.pop('SPECULA_PORT', None)
            self.assertEqual(registry.getenv('ITERAS_PORT'), '1234')

    def test_load_entry_point_plugins(self):
        ep = MagicMock()
        with patch.object(registry, '_entry_points_loaded', False), \
             patch('importlib.metadata.entry_points', return_value=[ep]) as mock_eps:
            registry.load_entry_point_plugins()
            registry.load_entry_point_plugins()     # loaded only once
        mock_eps.assert_called_once_with(group='iteras.plugins')
        ep.load.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
