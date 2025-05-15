
import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.modalrec import Modalrec
from specula.processing_objects.modalrec_implicit_polc import ModalrecImplicitPolc
from specula.data_objects.recmat import Recmat
from specula.data_objects.intmat import Intmat
from specula.data_objects.slopes import Slopes
from specula.base_value import BaseValue

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

    @cpu_and_gpu
    def test_modalrec_vs_implicit_polc(self, target_device_idx, xp):
        target_device_idx = None

        # intmat: 6x4 random array
        intmat_arr = xp.array([
                            [1, 0,  1,  1],
                            [0, 1, -1,  1],
                            [1, 0, -1,  1],
                            [0, 1,  1, -1],
                            [1, 0,  1, -1],
                            [0, 1, -1, -1]
                        ])
        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)

        # recmat: pseudo-inverse or intmat (shape 4x6)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # projmat: 2x4 with a diagonal of 2
        projmat_arr = xp.eye(4) * 2
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # slopes: 10 random values
        slopes_list = [3,  1.5,  3,  -0.5,  1,  1.5]
        slopes    = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)
        slopes_ip = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)

        # commands: 4 random values
        commands_list = [0.1, 0.2, 0.3, 0.4]
        commands    = BaseValue('commands', value=xp.array(commands_list), target_device_idx=target_device_idx)
        commands_ip = BaseValue('commands', value=xp.array(commands_list), target_device_idx=target_device_idx)

        # Modalrec standard (POLC)
        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            target_device_idx=target_device_idx
        )
        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.prepare_trigger(0)
        rec.trigger_code()
        out1 = rec.modes.value.copy()

        # ModalrecImplicitPolc
        rec2 = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            target_device_idx=target_device_idx
        )
        rec2.inputs['in_slopes'].set(slopes_ip)
        rec2.inputs['in_commands'].set(commands_ip)
        rec2.prepare_trigger(0)
        rec2.trigger_code()
        out2 = rec2.modes.value.copy()

        xp.testing.assert_allclose(out1, out2, rtol=1e-10, atol=1e-12)