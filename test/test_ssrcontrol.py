import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula import cpuArray
from specula.data_objects.ssr_filter_data import SsrFilterData
from specula.processing_objects.ssr_filter import SsrFilter
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula.processing_objects.schedule_generator import ScheduleGenerator
from test.specula_testlib import cpu_and_gpu

class TestSsrFilter(unittest.TestCase):
    """Test suite for SsrFilter processing object"""

    @cpu_and_gpu
    def test_instantiation(self, target_device_idx, xp):
        """Test basic instantiation"""
        simul_params = SimulParams(time_step=0.001)
        ssr_data = SsrFilterData.from_gain([1.0], target_device_idx=target_device_idx)

        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        self.assertEqual(ssr_filter._nfilter, 1)
        self.assertIsNotNone(ssr_filter.outputs['out_comm'])

    @cpu_and_gpu
    def test_pure_gain_response(self, target_device_idx, xp):
        """Test pure gain (no dynamics)"""
        simul_params = SimulParams(time_step=0.001)
        gains = [0.5, 2.0]
        ssr_data = SsrFilterData.from_gain(gains, target_device_idx=target_device_idx)

        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create input
        input_value = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()

        # Trigger filter
        t0 = 0
        input_value.generation_time = t0
        ssr_filter.check_ready(t0)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output = cpuArray(ssr_filter.outputs['out_comm'].value)
        expected = np.array([0.5, 2.0])

        np.testing.assert_array_almost_equal(output, expected)

    @cpu_and_gpu
    def test_integrator_response(self, target_device_idx, xp):
        """Test integrator with constant input"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step
        gain = 0.5

        ssr_data = SsrFilterData.from_integrator([gain],
                                                target_device_idx=target_device_idx)

        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create constant input
        input_value = BaseValue(value=xp.array([1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()

        # Run for 3 steps
        expected_outputs = [gain, 2 * gain, 3 * gain]

        for step, expected in enumerate(expected_outputs):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            output = cpuArray(ssr_filter.outputs['out_comm'].value)
            np.testing.assert_almost_equal(output[0], expected, decimal=6)

    @cpu_and_gpu
    def test_with_gain_modulation(self, target_device_idx, xp):
        """Test filter with gain_mod input"""
        simul_params = SimulParams(time_step=0.001)
        gains = [1.0, 1.0]
        ssr_data = SsrFilterData.from_gain(gains, target_device_idx=target_device_idx)

        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create inputs
        input_value = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)
        gain_mod = BaseValue(value=xp.array([0.5, 2.0], dtype=xp.float32),
                            target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.inputs['gain_mod'].set(gain_mod)
        ssr_filter.setup()

        # Trigger
        t0 = 0
        input_value.generation_time = t0
        gain_mod.generation_time = t0
        ssr_filter.check_ready(t0)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output = cpuArray(ssr_filter.outputs['out_comm'].value)
        expected = np.array([0.5, 2.0])  # gain * input * gain_mod

        np.testing.assert_array_almost_equal(output, expected)

    @cpu_and_gpu
    def test_with_schedule_generator_gain_mod(self, target_device_idx, xp):
        """Test integrator with VALUE_SCHEDULE gain_mod"""
        simul_params = SimulParams(time_step=0.001)

        # Create integrator
        gains = [0.5, 0.3]
        ssr_data = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create VALUE_SCHEDULE gain_mod
        gain_mod_generator = ScheduleGenerator(
            scheduled_values=[
                [1.0, 1.0],  # t < 0.002s
                [2.0, 0.5]   # t >= 0.002s
            ],
            scheduled_times=[0.002],
            modes_per_group=[1, 1],
            target_device_idx=target_device_idx
        )

        # Create constant input
        constant_input = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                                  target_device_idx=target_device_idx)

        # Connect
        ssr_filter.inputs['delta_comm'].set(constant_input)
        ssr_filter.inputs['gain_mod'].set(gain_mod_generator.outputs['output'])

        gain_mod_generator.setup()
        ssr_filter.setup()

        # Frame 0: gain_mod=[1.0, 1.0]
        # output = [0.5*1.0*1.0, 0.3*1.0*1.0]
        t0 = 0
        constant_input.generation_time = t0
        gain_mod_generator.check_ready(t0)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        ssr_filter.check_ready(t0)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output_0 = cpuArray(ssr_filter.outputs['out_comm'].value)
        expected_0 = np.array([0.5, 0.3])
        np.testing.assert_allclose(output_0, expected_0, rtol=1e-5)

        # Frame 1: t=0.001, gain_mod=[1.0, 1.0]
        t1 = ssr_filter.seconds_to_t(0.001)
        constant_input.generation_time = t1
        gain_mod_generator.check_ready(t1)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        ssr_filter.check_ready(t1)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output_1 = cpuArray(ssr_filter.outputs['out_comm'].value)
        expected_1 = np.array([2 * 0.5, 2 * 0.3])
        np.testing.assert_allclose(output_1, expected_1, rtol=1e-5)

        # Frame 2: t=0.002, gain_mod=[2.0, 0.5]
        t2 = ssr_filter.seconds_to_t(0.002)
        constant_input.generation_time = t2
        gain_mod_generator.check_ready(t2)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        ssr_filter.check_ready(t2)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output_2 = cpuArray(ssr_filter.outputs['out_comm'].value)
        # Previous state + new contribution with new gain_mod
        expected_2 = np.array([2*0.5 + 0.5*2.0, 2*0.3 + 0.3*0.5])
        np.testing.assert_allclose(output_2, expected_2, rtol=1e-5)

    @cpu_and_gpu
    def test_delay_implementation(self, target_device_idx, xp):
        """Test delay buffer implementation"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step
        delay = 2.0  # 2 frames delay

        ssr_data = SsrFilterData.from_gain([1.0], target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, delay=delay,
                              target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.array([1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()

        # First 2 frames should output 0 (due to delay)
        for step in range(2):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            output = cpuArray(ssr_filter.outputs['out_comm'].value)
            np.testing.assert_almost_equal(output[0], 0.0)

        # Third frame should output 1.0
        t = ssr_filter.seconds_to_t(2 * dt)
        input_value.generation_time = t
        ssr_filter.check_ready(t)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output = cpuArray(ssr_filter.outputs['out_comm'].value)
        np.testing.assert_almost_equal(output[0], 1.0, decimal=5)

    @cpu_and_gpu
    def test_reset_states(self, target_device_idx, xp):
        """Test reset_states functionality"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step

        ssr_data = SsrFilterData.from_integrator([1.0],
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.array([1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()

        # Run for 2 steps
        for step in range(2):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

        output_before = cpuArray(ssr_filter.outputs['out_comm'].value)
        self.assertGreater(output_before[0], 0)

        # Reset
        ssr_filter.reset_states()

        # Output should be zero after reset
        t = ssr_filter.seconds_to_t(2 * dt)
        input_value.value = xp.array([0.0], dtype=xp.float32)
        input_value.generation_time = t
        ssr_filter.check_ready(t)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output_after = cpuArray(ssr_filter.outputs['out_comm'].value)
        np.testing.assert_almost_equal(output_after[0], 0.0)

    @cpu_and_gpu
    def test_multi_mode_different_dynamics(self, target_device_idx, xp):
        """Test multiple modes with different state-space dimensions"""
        simul_params = SimulParams(time_step=0.001)

        # First filter: simple gain (no state)
        A1 = xp.array([[0.0]])
        B1 = xp.array([[0.0]])
        C1 = xp.array([[0.0]])
        D1 = xp.array([[2.0]])

        # Second filter: integrator (1 state)
        A2 = xp.array([[1.0]])
        B2 = xp.array([[0.5]])
        C2 = xp.array([[1.0]])
        D2 = xp.array([[0.0]])

        ssr_data = SsrFilterData([A1, A2], [B1, B2], [C1, C2], [D1, D2],
                                target_device_idx=target_device_idx)

        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        ssr_filter.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()

        # First step: gain outputs immediately, integrator starts accumulating
        t0 = 0
        input_value.generation_time = t0
        ssr_filter.check_ready(t0)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        output = cpuArray(ssr_filter.outputs['out_comm'].value)
        np.testing.assert_almost_equal(output[0], 2.0)  # Gain
        np.testing.assert_almost_equal(output[1], 0.5)  # Integrator
