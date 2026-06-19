import specula
specula.init(0)  # Default target device
from specula.loop_control import LoopControl

import unittest

from specula import cpuArray, np
from specula.processing_objects.phase_extractor import PhaseExtractor
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.layer import Layer

from test.specula_testlib import cpu_and_gpu


class TestPhaseExtractor(unittest.TestCase):

    @cpu_and_gpu
    def test_extract_from_electric_field(self, target_device_idx, xp):
        ef = ElectricField(4, 4, 0.1, target_device_idx=target_device_idx)
        ef.phaseInNm = xp.arange(16, dtype=ef.dtype).reshape(4, 4)
        ef.generation_time = ef.seconds_to_t(1)

        extractor = PhaseExtractor(target_device_idx=target_device_idx)
        extractor.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(extractor, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        result = cpuArray(extractor.outputs['out_phase'].value)
        expected = np.arange(16, dtype=float).reshape(4, 4)
        np.testing.assert_array_almost_equal(result, expected)

    @cpu_and_gpu
    def test_extract_from_layer(self, target_device_idx, xp):
        layer = Layer(4, 4, 0.1, height=1000.0, target_device_idx=target_device_idx)
        layer.phaseInNm = xp.ones((4, 4), dtype=layer.dtype) * 42.0
        layer.generation_time = layer.seconds_to_t(1)

        extractor = PhaseExtractor(target_device_idx=target_device_idx)
        extractor.inputs['in_ef'].set(layer)

        loop = LoopControl()
        loop.add(extractor, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        result = cpuArray(extractor.outputs['out_phase'].value)
        assert result.shape == (4, 4)
        np.testing.assert_array_almost_equal(result, np.full((4, 4), 42.0))

    @cpu_and_gpu
    def test_output_shape(self, target_device_idx, xp):
        ef = ElectricField(8, 6, 0.05, target_device_idx=target_device_idx)
        ef.generation_time = ef.seconds_to_t(1)

        extractor = PhaseExtractor(target_device_idx=target_device_idx)
        extractor.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(extractor, idx=0)
        loop.run(run_time=2, dt=1, t0=1)

        assert extractor.outputs['out_phase'].value.shape == (6, 8)
