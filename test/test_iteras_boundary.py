'''
iteras must not depend on specula: it will become a separate package,
of which specula is a plugin. These tests check that no iteras module
imports specula, and that a simulation made of iteras objects only runs
in a process where specula cannot be imported.
'''
import ast
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import iteras

ITERAS_ROOT = Path(iteras.__file__).resolve().parent


def _imported_modules(path):
    '''Names of the modules imported by a Python source file'''
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


class TestIteraBoundary(unittest.TestCase):

    def test_iteras_does_not_import_specula(self):
        offending = []
        for path in sorted(ITERAS_ROOT.rglob('*.py')):
            for module in _imported_modules(path):
                if module == 'specula' or module.startswith('specula.'):
                    offending.append(f'{path.relative_to(ITERAS_ROOT.parent)}: {module}')
        self.assertEqual(offending, [], 'iteras modules must not import specula')

    def test_simulation_without_specula(self):
        '''
        Run a simulation with a feedback loop (delayed input) using iteras objects
        and processing objects from an additional module, in a subprocess
        where any import of specula fails.
        '''
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            pkg = tmp / 'iteras_boundary_objs'
            pkg.mkdir()
            (pkg / '__init__.py').write_text('')
            (pkg / 'step.py').write_text(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue

                class Step(BaseProcessingObj):
                    def __init__(self, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.ones(1, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
                '''))
            (pkg / 'accumulator.py').write_text(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue
                from iteras.connections import InputValue

                class Accumulator(BaseProcessingObj):
                    def __init__(self, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out
                        self.inputs['in_step'] = InputValue(type=BaseValue)
                        self.inputs['in_prev'] = InputValue(type=BaseValue)

                    def trigger_code(self):
                        self.out.value[:] = self.local_inputs['in_prev'].value + \\
                                            self.local_inputs['in_step'].value

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
                '''))
            (pkg / 'copy.py').write_text(textwrap.dedent('''
                from iteras.base_processing_obj import BaseProcessingObj
                from iteras.base_value import BaseValue
                from iteras.connections import InputValue

                class Copy(BaseProcessingObj):
                    def __init__(self, target_device_idx=None, precision=None):
                        super().__init__(target_device_idx=target_device_idx, precision=precision)
                        self.out = BaseValue(value=self.xp.zeros(1, dtype=self.dtype),
                                             target_device_idx=target_device_idx)
                        self.outputs['out_value'] = self.out
                        self.inputs['in_value'] = InputValue(type=BaseValue)

                    def trigger_code(self):
                        self.out.value[:] = self.local_inputs['in_value'].value

                    def post_trigger(self):
                        super().post_trigger()
                        self.out.generation_time = self.current_time
                '''))
            store_dir = tmp / 'store'
            (tmp / 'params.yml').write_text(textwrap.dedent(f'''
                main:
                  class: 'SimParams'
                  root_dir: '{tmp}'
                  total_time: 0.010
                  time_step: 0.001
                  add_modules: ['iteras_boundary_objs']

                step:
                  class: 'Step'

                acc:
                  class: 'Accumulator'
                  inputs:
                    in_step: 'step.out_value'
                    in_prev: 'copy.out_value:-1'

                copy:
                  class: 'Copy'
                  inputs:
                    in_value: 'acc.out_value'

                store:
                  class: 'DataStore'
                  store_dir: '{store_dir}'
                  create_tn: False
                  inputs:
                    input_list: ['total-acc.out_value']
                '''))
            script = textwrap.dedent(f'''
                import sys
                sys.modules['specula'] = None   # any import of specula fails
                import iteras
                iteras.init(-1, precision=0)
                from iteras.simul import Simul
                Simul({str(tmp / 'params.yml')!r}, speed_report=False).run()
                loaded = [m for m, mod in sys.modules.items()
                          if m.split('.')[0] == 'specula' and mod is not None]
                assert not loaded, loaded
                from astropy.io import fits
                print('RESULT', fits.getdata({str(store_dir / 'total.fits')!r}).ravel().tolist())
                ''')
            env = os.environ.copy()
            env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ITERAS_ROOT.parent), str(tmp),
                                                               env.get('PYTHONPATH')]))
            env['MPLBACKEND'] = 'Agg'
            r = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                               env=env, cwd=tmpdir)
            self.assertEqual(r.returncode, 0, r.stderr)
            result = [line for line in r.stdout.splitlines() if line.startswith('RESULT')]
            self.assertEqual(result, ['RESULT ' + str([float(i) for i in range(1, 11)])], r.stdout)


if __name__ == '__main__':
    unittest.main()
