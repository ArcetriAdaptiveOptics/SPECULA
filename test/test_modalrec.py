
import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.modalrec import Modalrec
from specula.data_objects.slopes import Slopes
from specula.data_objects.recmat import Recmat

from test.specula_testlib import cpu_and_gpu

class TestModalrec(unittest.TestCase):

    @cpu_and_gpu
    def test_modalrec_wrong_size(self, target_device_idx, xp):
        
        recmat = Recmat(xp.arange(12).reshape((3,4)), target_device_idx=target_device_idx)
        rec = Modalrec(recmat=recmat, target_device_idx=target_device_idx)

        slopes = Slopes(slopes=xp.arange(5), target_device_idx=target_device_idx)
        rec.inputs['in_slopes'].set(slopes)

        t = 1
        slopes.generation_time = t
        rec.prepare_trigger(t)
        with self.assertRaises(ValueError):
            rec.trigger_code()
