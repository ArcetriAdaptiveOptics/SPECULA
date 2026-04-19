import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'docs' / 'scripts' / 'generate_objects_summary.py'
SPEC = importlib.util.spec_from_file_location('generate_objects_summary', MODULE_PATH)
generate_objects_summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_objects_summary)


class TestGenerateObjectsSummary(unittest.TestCase):

    def _write_file(self, path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content), encoding='utf-8')

    def test_extract_classes_resolves_super_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / 'pkg_inherited'
            base_file = pkg / 'base_obj.py'
            child_file = pkg / 'child_obj.py'

            self._write_file(pkg / '__init__.py', '')

            self._write_file(
                base_file,
                """
                from collections import namedtuple

                InputDesc = namedtuple('InputDesc', 'type desc')
                OutputDesc = namedtuple('OutputDesc', 'type desc')

                class BaseValue:
                    pass

                class BaseObj:
                    @classmethod
                    def input_names(cls):
                        return {
                            'in_value': InputDesc(BaseValue, 'Main input'),
                            'in_optional': InputDesc(BaseValue, 'Optional input (optional)'),
                        }

                    @classmethod
                    def output_names(cls):
                        return {
                            'out_base': OutputDesc(BaseValue, 'Base output'),
                        }
                """,
            )

            self._write_file(
                child_file,
                """
                from .base_obj import BaseObj, BaseValue, OutputDesc

                class ChildObj(BaseObj):
                    @classmethod
                    def input_names(cls):
                        return super().input_names()

                    @classmethod
                    def output_names(cls):
                        result = super().output_names()
                        result.update({
                            'out_modes_{sensor_idx}': OutputDesc(BaseValue, 'Dynamic output pattern'),
                        })
                        return result
                """,
            )

            sys.path.insert(0, str(root))
            self.addCleanup(lambda: sys.path.remove(str(root)) if str(root) in sys.path else None)

            registry = {}
            registry.update(generate_objects_summary.extract_classes_from_file(
                base_file,
                module_name='pkg_inherited.base_obj',
            ))
            registry.update(generate_objects_summary.extract_classes_from_file(
                child_file,
                module_name='pkg_inherited.child_obj',
            ))

            info = registry['ChildObj']
            inputs = info['named_inputs']
            outputs = info['named_outputs']

            self.assertEqual(inputs['in_value'], False)
            self.assertEqual(inputs['in_optional'], True)
            self.assertIn('out_base', outputs)
            self.assertIn('out_modes_{sensor_idx}', outputs)

    def test_generate_rst_table_includes_resolved_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / 'pkg_simple'
            file_path = pkg / 'child_obj.py'

            self._write_file(pkg / '__init__.py', '')

            self._write_file(
                file_path,
                """
                from collections import namedtuple

                InputDesc = namedtuple('InputDesc', 'type desc')
                OutputDesc = namedtuple('OutputDesc', 'type desc')

                class BaseValue:
                    pass

                class ChildObj:
                    @classmethod
                    def input_names(cls):
                        return {
                            'in_sig': InputDesc(BaseValue, 'Signal input'),
                            'gain_mod': InputDesc(BaseValue, 'Optional gain (optional)'),
                        }

                    @classmethod
                    def output_names(cls):
                        return {
                            'out_modes_{sensor_idx}': OutputDesc(BaseValue, 'Dynamic output pattern'),
                        }
                """,
            )

            modules = [('pkg_simple.child_obj', file_path)]

            sys.path.insert(0, str(root))
            self.addCleanup(lambda: sys.path.remove(str(root)) if str(root) in sys.path else None)

            rst = generate_objects_summary.generate_rst_table(
                'Processing Objects',
                modules,
                description='Synthetic test table.',
                include_io=True,
            )

            self.assertIn('in_sig', rst)
            self.assertIn('gain_mod *(opt)*', rst)
            self.assertIn('out_modes_[sensor_idx]', rst)
            self.assertIn('     - -\n     - -', rst)

    def test_generate_rst_table_raises_on_empty_processing_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / 'pkg_broken'
            file_path = pkg / 'broken_obj.py'

            self._write_file(pkg / '__init__.py', '')
            self._write_file(
                file_path,
                """
                import definitely_missing_dependency

                class BrokenObj:
                    pass
                """,
            )

            modules = [('pkg_broken.broken_obj', file_path)]

            sys.path.insert(0, str(root))
            self.addCleanup(lambda: sys.path.remove(str(root)) if str(root) in sys.path else None)

            with self.assertRaises(RuntimeError) as cm:
                generate_objects_summary.generate_rst_table(
                    'Processing Objects',
                    modules,
                    description='Synthetic failing table.',
                    include_io=True,
                )

            self.assertIn('No classes extracted', str(cm.exception))
