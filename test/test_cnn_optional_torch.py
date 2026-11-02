import os
import subprocess
import sys
import textwrap
import unittest


class TestCnnOptionalTorch(unittest.TestCase):
    """torch is an optional dependency: the CNN processing objects must be
    importable without it (SPECULA's test helpers import every processing
    object module), and only their construction must fail, clearly."""

    def test_import_without_torch_and_clear_error_on_construction(self):
        # A separate process, so that torch can be hidden even when installed.
        code = textwrap.dedent('''
            import sys
            sys.modules['torch'] = None        # any "import torch" now raises ImportError
            import specula
            specula.init(-1)
            from iteras.processing_objects.conv2d_net_trainer import Conv2dNetTrainer
            from iteras.processing_objects.conv2d_net_tester import Conv2dNetTester
            for cls, kwargs in [(Conv2dNetTrainer, {'network_filename': 'x.pth'}),
                                (Conv2dNetTester, {'network_filename': 'x.pth'})]:
                try:
                    cls(**kwargs)
                except ImportError as e:
                    assert 'needs torch' in str(e), str(e)
                else:
                    raise AssertionError(f'{cls.__name__} built without torch')
            print('OK')
        ''')
        # From the repository root, so that the modules of this checkout are imported
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=repo)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('OK', out.stdout)


if __name__ == '__main__':
    unittest.main()
