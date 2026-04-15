import unittest

import specula
specula.init(0)

from specula.base_processing_obj import InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from test.specula_testlib import cpu_and_gpu


class TestPowerLossSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.data_objects.simul_params import SimulParams
        from specula.processing_objects.power_loss import PowerLoss
        sp = SimulParams(pixel_pupil=20, pixel_pitch=0.05)
        obj = PowerLoss(simul_params=sp, wavelengthInNm=500.0, nd=2,
                        prop_distance=400e3, receiver_diam=0.1,
                        target_device_idx=target_device_idx)
        obj.sanity_check()


class TestElectricFieldReflectionSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.electric_field_reflection import ElectricFieldReflection
        obj = ElectricFieldReflection(target_device_idx=target_device_idx)
        obj.sanity_check()


class TestElectricFieldCombinatorSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.electric_field_combinator import ElectricFieldCombinator
        obj = ElectricFieldCombinator(target_device_idx=target_device_idx)
        obj.sanity_check()


class TestPhaseFlatteningSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.phase_flattening import PhaseFlattening
        obj = PhaseFlattening(target_device_idx=target_device_idx)
        obj.sanity_check()


class TestBaseInserterSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.base_inserter import BaseInserter
        obj = BaseInserter(output_size=10, slice_args=[[0, 3], [2, 5]],
                           target_device_idx=target_device_idx)
        obj.sanity_check()


class TestDataPrintSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.data_print import DataPrint
        obj = DataPrint(target_device_idx=target_device_idx)
        obj.sanity_check()


class TestAvcSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.processing_objects.avc import AVC
        obj = AVC(target_device_idx=target_device_idx)
        obj.sanity_check()


class TestDataBufferSanityCheck(unittest.TestCase):

    def test_sanity_check_with_dynamic_inputs(self):
        from specula.processing_objects.data_buffer import DataBuffer
        obj = DataBuffer(buffer_size=5)
        obj.inputs['some_dynamic_input'] = InputValue(type=BaseValue)
        obj.sanity_check()  # Should not raise thanks to check_input_names override

    def test_check_input_names_is_noop(self):
        from specula.processing_objects.data_buffer import DataBuffer
        obj = DataBuffer()
        obj.inputs['dynamic'] = InputValue(type=BaseValue)
        obj.check_input_names()  # Should pass silently

    def test_check_output_names_is_noop(self):
        from specula.processing_objects.data_buffer import DataBuffer
        obj = DataBuffer()
        obj.outputs['dynamic_out'] = BaseValue()
        obj.check_output_names()  # Should pass silently


if __name__ == '__main__':
    unittest.main()