'''
Modules can be imported before init(): no module copies the values set by
init() when it is imported, and GPU kernels are created at the first GPU call.
'''
import ast
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import iteras
import specula
from iteras import registry

# Values set by init(): copying them at import time (e.g. "from specula import cp")
# freezes the value present when the module is imported
INIT_VALUES = {'cp', 'xp', 'gpuEnabled', 'global_precision', 'float_dtype_list',
               'complex_dtype_list', 'gpu_float_dtype_list', 'gpu_complex_dtype_list',
               'array_types', 'float_dtype', 'complex_dtype', 'default_target_device_idx',
               'default_target_device', 'process_comm', 'process_rank'}

ROOTS = [Path(iteras.__file__).resolve().parent, Path(specula.__file__).resolve().parent]

try:
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# The GPU can also be disabled with ITERAS_DISABLE_GPU or SPECULA_DISABLE_GPU
GPU_AVAILABLE = CUPY_AVAILABLE and registry.getenv('ITERAS_DISABLE_GPU', 'FALSE') == 'FALSE'


class TestImportBeforeInit(unittest.TestCase):

    def test_no_module_copies_init_values(self):
        offending = []
        for root in ROOTS:
            for path in sorted(root.rglob('*.py')):
                # Bytes: decoded as Python does (UTF-8 by default), whatever the platform encoding
                tree = ast.parse(path.read_bytes(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.level == 0 \
                            and node.module in ('iteras', 'specula'):
                        names = {alias.name for alias in node.names} & INIT_VALUES
                        if names:
                            offending.append(f'{path.relative_to(root.parent)}:{node.lineno} '
                                             f'imports {sorted(names)}')
        self.assertEqual(offending, [], 'Read these values when they are used '
                         '(e.g. self.xp, iteras.cp, specula.cp), not at import')

    @unittest.skipUnless(GPU_AVAILABLE, 'GPU not available')
    def test_gpu_functions_imported_before_init(self):  # pragma: no cover
        '''
        Import the modules with GPU code before init(0), then check that their
        GPU paths give the same results as on the CPU.
        '''
        script = textwrap.dedent('''
            import numpy as np
            from specula.lib.interp2d import Interp2D
            from specula.lib.toccd import toccd
            from specula.lib.affine_transform import affine_transform
            from specula.processing_objects.slopec import sum_product
            from specula.lib.mask import CircularMask
            import specula
            specula.init(0, precision=1)
            import cupy as cp

            a = np.random.default_rng(1).random((16, 16), dtype=np.float32)

            interp_cpu = Interp2D((16, 16), (8, 8), rotInDeg=10, dtype=np.float32, xp=np)
            interp_gpu = Interp2D((16, 16), (8, 8), rotInDeg=10, dtype=np.float32, xp=cp)
            np.testing.assert_allclose(cp.asnumpy(interp_gpu.interpolate(cp.asarray(a))),
                                       interp_cpu.interpolate(a), rtol=1e-4, atol=1e-5)

            np.testing.assert_allclose(cp.asnumpy(toccd(cp.asarray(a), (8, 8), xp=cp)),
                                       toccd(a, (8, 8), xp=np), rtol=1e-5)

            matrix = np.array([[1, 0, 0.5], [0, 1, 0.25]], dtype=np.float64)
            out_cpu = np.zeros_like(a)
            out_gpu = cp.zeros_like(cp.asarray(a))
            affine_transform(a, matrix, out_cpu, xp=np)
            affine_transform(cp.asarray(a), cp.asarray(matrix), out_gpu, xp=cp)
            np.testing.assert_allclose(cp.asnumpy(out_gpu), out_cpu, rtol=1e-5, atol=1e-6)

            np.testing.assert_allclose(cp.asnumpy(sum_product(cp.asarray(a), cp.asarray(a), xp=cp)),
                                       sum_product(a, a, xp=np), rtol=1e-5)

            # The default array module of the masks is the one set by init()
            assert isinstance(CircularMask((8, 8)).mask(), cp.ndarray)
            print('OK')
            ''')
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOTS[0].parent), env.get('PYTHONPATH')]))
        r = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('OK', r.stdout)


if __name__ == '__main__':
    unittest.main()
