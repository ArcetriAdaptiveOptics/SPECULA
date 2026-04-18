import specula
specula.init(0)  # Default target device

import os
import numpy as np
import unittest
import importlib

from unittest import mock


class TestDisableGpu(unittest.TestCase):

    def test_disable_gpu(self):
        '''Test that SPECULA_DISABLE_GPU always results in numpy being loaded instead of cupy'''

        previous = os.environ.get('SPECULA_DISABLE_GPU', None)

        try:
            with mock.patch.dict("sys.modules", {
                "cupy": mock.Mock()
            }):
                os.environ['SPECULA_DISABLE_GPU'] = '1'
                importlib.reload(specula)
                specula.init(0)
                assert specula.xp == np
        finally:
            if previous is not None:
                os.environ['SPECULA_DISABLE_GPU'] = previous