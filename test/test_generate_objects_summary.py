import importlib.util
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

    def test_inherited_io_resolution_with_super_update(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_file = root / 'base_obj.py'
            child_file = root / 'child_obj.py'

            self._write_file(
                base_file,
                """
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

            registry = {}
            registry.update(generate_objects_summary.extract_classes_from_file(base_file))
            registry.update(generate_objects_summary.extract_classes_from_file(child_file))

            inputs, outputs = generate_objects_summary.get_inherited_io('ChildObj', registry)

            self.assertEqual(inputs['in_value'], False)
            self.assertEqual(inputs['in_optional'], True)
            self.assertIn('out_base', outputs)
            self.assertIn('out_modes_{sensor_idx}', outputs)

    def test_generate_rst_table_includes_resolved_io(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pkg = root / 'pkg'
            file_path = pkg / 'child_obj.py'

            self._write_file(
                file_path,
                """
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

            registry = generate_objects_summary.build_global_registry(root)
            modules = [('pkg.child_obj', file_path)]

            rst = generate_objects_summary.generate_rst_table(
                'Processing Objects',
                modules,
                registry,
                description='Synthetic test table.',
                include_io=True,
            )

            self.assertIn('in_sig', rst)
            self.assertIn('gain_mod *(opt)*', rst)
            self.assertIn('out_modes_[sensor_idx]', rst)
            self.assertNotIn('     - -\n     - -', rst)
