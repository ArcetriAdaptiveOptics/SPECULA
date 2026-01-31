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
        rec.setup()
        rec.prepare_trigger(t)
        with self.assertRaises(ValueError):
            rec.trigger_code()

    @cpu_and_gpu
    def test_modalrec_vs_implicit_polc(self, target_device_idx, xp):

        # intmat (shape 6x4)
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

        # slopes:
        slopes_list = [3,  1.5,  3,  -0.5,  1,  1.5]
        slopes    = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)
        slopes_ip = Slopes(slopes=xp.array(slopes_list), target_device_idx=target_device_idx)
        slopes.generation_time = 0
        slopes_ip.generation_time = 0

        # commands:
        commands_list = [0.1, 0.2, 0.3, 0.4]
        commands    = BaseValue('commands', value=xp.array(commands_list), target_device_idx=target_device_idx)
        commands_ip = BaseValue('commands', value=xp.array(commands_list), target_device_idx=target_device_idx)
        commands.generation_time = 0
        commands_ip.generation_time = 0

        # Modalrec standard (POLC)
        rec = Modalrec(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            polc=True,
            in_commands_size=4,
            target_device_idx=target_device_idx
        )
        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)
        rec.setup()
        rec.prepare_trigger(0)
        rec.trigger_code()
        out1 = rec.modes.value.copy()

        # ModalrecImplicitPolc
        rec2 = ModalrecImplicitPolc(
            recmat=recmat,
            projmat=projmat,
            intmat=intmat,
            in_commands_size=4,
            target_device_idx=target_device_idx
        )
        rec2.inputs['in_slopes'].set(slopes_ip)
        rec2.inputs['in_commands'].set(commands_ip)
        rec2.setup()
        rec2.prepare_trigger(0)
        rec2.trigger_code()
        out2 = rec2.modes.value.copy()

        xp.testing.assert_allclose(out1, out2, rtol=1e-10, atol=1e-12)

    @cpu_and_gpu
    def test_modalrec_polc_wrong_size(self, target_device_idx, xp):
        
        # intmat which expects 4 commands and produces 6 slopes
        intmat_arr = xp.array([
                            [1, 0,  1,  1],
                            [0, 1, -1,  1],
                            [1, 0, -1,  1],
                            [0, 1,  1, -1],
                            [1, 0,  1, -1],
                            [0, 1, -1, -1]
                        ])
        intmat = Intmat(intmat_arr, target_device_idx=target_device_idx)
        
        # recmat: pseudo-inverse of intmat (shape 4x6)
        recmat_arr = xp.linalg.pinv(intmat_arr)
        recmat = Recmat(recmat_arr, target_device_idx=target_device_idx)

        # projmat: 2x4 with a diagonal of 2
        projmat_arr = xp.eye(4) * 2
        projmat = Recmat(projmat_arr, target_device_idx=target_device_idx)

        # Create a Modalrec which expects 6 slopes and 4 commands
        rec = Modalrec(
            nmodes=4,
            recmat=recmat,
            intmat=intmat,
            projmat=projmat,
            polc=True,
            in_commands_size=4,
            target_device_idx=target_device_idx
        )

        # Slopes with wrong size (5 instead of 6)
        slopes = Slopes(slopes=xp.arange(5), target_device_idx=target_device_idx)
        commands = BaseValue('commands', value=xp.array([0.1, 0.2, 0.3, 0.4]), target_device_idx=target_device_idx)

        rec.inputs['in_slopes'].set(slopes)
        rec.inputs['in_commands'].set(commands)

        t = 1
        slopes.generation_time = t
        commands.generation_time = t

        rec.setup()

        # We expect a ValueError during prepare_trigger due to size mismatch
        with self.assertRaises(ValueError) as cm:
            rec.prepare_trigger(t)

        # Verify that the error message is as expected
        self.assertIn("Dimension mismatch in POLC mode", str(cm.exception))
        self.assertIn("intmat @ commands will produce 6 slopes", str(cm.exception))
        self.assertIn("but input slopes has size 5", str(cm.exception))