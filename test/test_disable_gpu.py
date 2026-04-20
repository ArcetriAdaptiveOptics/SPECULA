import specula
specula.init(0)  # Default target device

import os
import sys
import numpy as np
import unittest
import importlib

from unittest import mock


class TestDisableGpu(unittest.TestCase):

    def test_disable_gpu(self):
        '''Test that SPECULA_DISABLE_GPU always results in numpy being loaded instead of cupy'''

        orig_disable = os.environ.get('SPECULA_DISABLE_GPU', None)
        orig_cupy = sys.modules.get('cupy', None)

        try:
            with mock.patch.dict("sys.modules", {
                "cupy": mock.Mock()
            }):
                os.environ['SPECULA_DISABLE_GPU'] = '1'
                importlib.reload(specula)
                specula.init(0)
                assert specula.xp == np
        finally:
            if orig_disable is not None:
                os.environ['SPECULA_DISABLE_GPU'] = orig_disable
            else:
                del os.environ['SPECULA_DISABLE_GPU']
            if orig_cupy is not None:
                sys.modules['cupy'] = orig_cupy
            elif 'cupy' in sys.modules:
                del sys.modules['cupy']
            print(f'{orig_disable=}')
            importlib.reload(specula)
            specula.init(0)
