
import os
import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat


class TestModalrec(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_modalrec_wrong_size_gpu(self):
        self._test_modalrec_wrong_size(target_device_idx=0, xp=cp)

    def test_modalrec_wrong_size_cpu(self):
        self._test_modalrec_wrong_size(target_device_idx=-1, xp=np)
        
    def _test_modalrec_wrong_size(self, target_device_idx, xp):
        
        rec = Modalrec(recmat = Recmat(xp.arange(12).reshape((3,4))),
                       target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            rec.compute_modes(rec._recmat, xp.arange(5))

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_modalrec_uses_own_slopes_gpu(self):
        self._test_modalrec_uses_own_slopes(target_device_idx=0, xp=cp)

    def test_modalrec_uses_own_slopes_cpu(self):
        self._test_modalrec_uses_own_slopes(target_device_idx=-1, xp=np)

    def _test_modalrec_uses_own_slopes(self, target_device_idx, xp):
        
        rec = Modalrec(recmat = Recmat(xp.arange(12).reshape((3,4)), target_device_idx=target_device_idx),
                       target_device_idx=target_device_idx)

        rec._slopes = Slopes(slopes=xp.arange(4), target_device_idx=target_device_idx)
        print(type(rec._slopes.slopes))
        print(len(rec._slopes.slopes))

        res1 = rec.compute_modes(rec._recmat)
        res2 = rec.compute_modes(rec._recmat, None)
        np.testing.assert_array_equal(cpuArray(res1), cpuArray(res2))

if __name__ == '__main__':
    unittest.main()