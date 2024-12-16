
import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat

from test.specula_testlib import cpu_and_gpu

class TestModalrec(unittest.TestCase):

    @cpu_and_gpu
    def test_modalrec_wrong_size(self, target_device_idx, xp):
        rec = Modalrec(recmat = Recmat(xp.arange(12).reshape((3,4))),
                       target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            rec.compute_modes(rec._recmat, xp.arange(5))

    @cpu_and_gpu
    def test_modalrec_uses_own_slopes(self, target_device_idx, xp):
        
        rec = Modalrec(recmat = Recmat(xp.arange(12).reshape((3,4)), target_device_idx=target_device_idx),
                       target_device_idx=target_device_idx)

        rec._slopes = Slopes(slopes=xp.arange(4), target_device_idx=target_device_idx)

        res1 = rec.compute_modes(rec._recmat)
        res2 = rec.compute_modes(rec._recmat, None)
        np.testing.assert_array_equal(cpuArray(res1), cpuArray(res2))
