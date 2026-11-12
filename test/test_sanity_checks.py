import unittest

import specula
specula.init(0)

from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.loop_control import LoopControl
from test.specula_testlib import cpu_and_gpu


class TestPowerLossSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check(self, target_device_idx, xp):
        from specula.data_objects.simul_params import SimulParams
        from specula.processing_objects.power_loss import PowerLoss
        from specula.processing_objects.atmo_propagation import AtmoPropagation
        from specula.data_objects.source import Source
        sp = SimulParams(pixel_pupil=20, pixel_pitch=0.05)
        on_axis_source = Source(polar_coordinates=[0,0], magnitude=0.0, height=600000, wavelengthInNm=500.0)
        prop = AtmoPropagation(simul_params=sp, source_dict={'on_axis': on_axis_source}, doFresnel=True, upwards=True,
                               wavelengthInNm=500.0, padding_factor=2)
        obj = PowerLoss(simul_params=sp, prop=prop, target_device_idx=target_device_idx)
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
        from specula.data_objects.simul_params import SimulParams
        from specula.processing_objects.avc import AVC
        sp = SimulParams(time_step=0.001)
        obj = AVC(simul_params=sp, freq=[[30.0, 60.0], [45.0]],
                  gx=0.05, gomega=0.001, k=1.0,
                  target_device_idx=target_device_idx)
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


class TestDataSourceSanityCheck(unittest.TestCase):

    def test_sanity_check_with_dynamic_outputs(self):
        from specula.processing_objects.data_source import DataSource

        obj = DataSource(outputs=[], store_dir='/tmp')
        obj.outputs['dynamic_out'] = BaseValue()

        # check_output_names is intentionally custom/no-op for dynamic file-driven outputs.
        obj.sanity_check()


class TestAtmoPropagationSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_sanity_check_with_dynamic_outputs(self, target_device_idx, xp):
        """
        With mergeLayersContrib=True (default) the outputs are checked as for any
        object: a wrong-type output raises. With mergeLayersContrib=False each
        output is a list of fields created in setup(), and the static output
        check is skipped.
        """
        from specula.data_objects.simul_params import SimulParams
        from specula.data_objects.source import Source
        from specula.processing_objects.atmo_propagation import AtmoPropagation

        simul_params = SimulParams(pixel_pupil=16, pixel_pitch=0.1)
        source = Source(polar_coordinates=[0.0, 0.0], magnitude=8, wavelengthInNm=750)

        for merge in (True, False):
            obj = AtmoPropagation(
                simul_params,
                source_dict={'src': source},
                mergeLayersContrib=merge,
                target_device_idx=target_device_idx,
            )
            obj.sanity_check()
            obj.outputs['out_extra_ef'] = BaseValue(target_device_idx=target_device_idx)
            if merge:
                with self.assertRaises(TypeError):
                    obj.sanity_check()
            else:
                obj.sanity_check()


class _OptionalInputObj(BaseProcessingObj):
    """Declares an input described as "(optional)", created only if create_input."""

    @classmethod
    def input_names(cls):
        return {'opt_in': InputDesc(BaseValue, 'An input that may stay unconnected (optional)')}

    @classmethod
    def output_names(cls):
        return {}

    def __init__(self, create_input, target_device_idx=-1):
        super().__init__(target_device_idx=target_device_idx)
        if create_input:
            self.inputs['opt_in'] = InputValue(type=BaseValue, optional=True)


class _LateOutputObj(BaseProcessingObj):
    """Declares an output, created in the constructor or (wrongly) only in setup()."""

    @classmethod
    def input_names(cls):
        return {}

    @classmethod
    def output_names(cls):
        return {'out_value': OutputDesc(BaseValue, 'A value')}

    def __init__(self, create_in_init, target_device_idx=-1):
        super().__init__(target_device_idx=target_device_idx)
        self.create_in_init = create_in_init
        self.setup_called = False
        if create_in_init:
            self.outputs['out_value'] = BaseValue(target_device_idx=target_device_idx)

    def setup(self):
        super().setup()
        self.setup_called = True
        if not self.create_in_init:
            self.outputs['out_value'] = BaseValue(target_device_idx=self.target_device_idx)


class TestOptionalInputsMustExist(unittest.TestCase):
    """Issue #805: every declared input must exist, "(optional)" is documentation only."""

    def test_optional_description_but_input_missing_raises(self):
        """
        Expected: sanity_check() raises ValueError, even if the description of
        the declared input ends with "(optional)" (before #805 it was skipped).
        """
        obj = _OptionalInputObj(create_input=False)
        with self.assertRaisesRegex(ValueError, 'declared in input_names but not present'):
            obj.sanity_check()

    def test_optional_input_created_passes_and_runs_unconnected(self):
        """
        Expected: with the input created as InputValue(optional=True), sanity_check()
        passes and the object runs in a LoopControl with the input unconnected
        (setup() does not complain about it).
        """
        obj = _OptionalInputObj(create_input=True)
        obj.sanity_check()
        self.assertTrue(obj.inputs['opt_in'].optional)

        loop = LoopControl()
        loop.add(obj, idx=0)
        loop.run(run_time=1, dt=1)


class TestSanityCheckBeforeSetup(unittest.TestCase):
    """Issue #805: LoopControl calls sanity_check() before setup()."""

    def test_output_created_only_in_setup_raises_before_setup(self):
        """
        Expected: running the loop raises ValueError from sanity_check() and
        setup() has not run yet (before #805 setup() ran first and created the output).
        """
        obj = _LateOutputObj(create_in_init=False)
        loop = LoopControl()
        loop.add(obj, idx=0)
        with self.assertRaisesRegex(ValueError, 'declared in output_names but not present'):
            loop.run(run_time=1, dt=1)
        self.assertFalse(obj.setup_called)

    def test_output_created_in_constructor_runs(self):
        """Expected: the same object with the output created in the constructor runs fine."""
        obj = _LateOutputObj(create_in_init=True)
        loop = LoopControl()
        loop.add(obj, idx=0)
        loop.run(run_time=1, dt=1)
        self.assertTrue(obj.setup_called)


class TestSHSodiumInputsSanityCheck(unittest.TestCase):

    @cpu_and_gpu
    def test_ngs_sh_has_optional_sodium_inputs(self, target_device_idx, xp):
        """
        Issue #805. Expected: an NGS SH (no laser launch telescope) has the
        sodium_altitude and sodium_intensity inputs, both optional, and
        sanity_check() passes.
        """
        from specula.processing_objects.sh import SH
        sh = SH(wavelengthInNm=500, subap_wanted_fov=3, sensor_pxscale=0.5,
                subap_on_diameter=4, subap_npx=6, target_device_idx=target_device_idx)
        for name in ('sodium_altitude', 'sodium_intensity'):
            self.assertIn(name, sh.inputs)
            self.assertTrue(sh.inputs[name].optional)
        sh.sanity_check()
