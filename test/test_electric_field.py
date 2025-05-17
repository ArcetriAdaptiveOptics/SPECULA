
import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.electric_field_combinator import ElectricFieldCombinator

from test.specula_testlib import cpu_and_gpu

class TestElectricField(unittest.TestCase):

    @cpu_and_gpu
    def test_reset_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)

        ef.reset()

        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)

        assert id_A_before == id_A_after
        assert id_p_before == id_p_after

    @cpu_and_gpu
    def test_set_value_does_not_reallocate(self, target_device_idx, xp):

        ef = ElectricField(10,10, 0.1, S0=1, target_device_idx=target_device_idx)

        id_A_before = id(ef.A)
        id_p_before = id(ef.phaseInNm)

        ef.set_value([xp.ones(100).reshape(10,10), xp.zeros(100).reshape(10,10)])

        id_A_after = id(ef.A)
        id_p_after = id(ef.phaseInNm)

        assert id_A_before == id_A_after
        assert id_p_before == id_p_after

    @cpu_and_gpu
    def test_ef_combinator(self, target_device_idx, xp):
        pixel_pitch = 0.1
        simulParams = SimulParams(pixel_pitch=pixel_pitch)
        ef1 = ElectricField(10,10, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef2 = ElectricField(10,10, pixel_pitch, S0=2, target_device_idx=target_device_idx)

        A1 = xp.ones((10, 10))
        ef1.A = A1
        ef1.phaseInNm = 1 * xp.ones((10, 10))
        A2 = xp.ones((10, 10))
        A2[0, 0] = 0
        A2[9, 9] = 0
        ef2.A = A2
        ef2.phaseInNm = 3 * xp.ones((10, 10))

        ef_combinator = ElectricFieldCombinator(
            simul_params=simulParams,
            target_device_idx=target_device_idx
        )

        ef_combinator.inputs['in_ef1'].set(ef1)
        ef_combinator.inputs['in_ef2'].set(ef2)

        t = 1
        ef1.generation_time = t
        ef2.generation_time = t

        ef_combinator.check_ready(t)
        ef_combinator.setup()
        ef_combinator.trigger()
        ef_combinator.post_trigger()

        out_ef = ef_combinator.outputs['out_ef']

        assert np.allclose(out_ef.A, ef1.A * ef2.A)
        assert np.allclose(out_ef.phaseInNm, ef1.phaseInNm + ef2.phaseInNm)
        assert np.allclose(out_ef.S0, ef1.S0 + ef2.S0)