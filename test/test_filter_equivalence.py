import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula import cpuArray
from specula.data_objects.ssr_filter_data import SsrFilterData
from specula.processing_objects.ssr_filter import SsrFilter
from specula.processing_objects.integrator import Integrator
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from test.specula_testlib import cpu_and_gpu


class TestSsrIirEquivalence(unittest.TestCase):
    """Test equivalence between SSR and IIR integrator implementations"""

    @cpu_and_gpu
    def test_integrator_equivalence_constant_input(self, target_device_idx, xp):
        """Test that SSR and IIR integrators produce same results with constant input"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step

        # Parameters
        gains = [0.5, 0.3, 0.7]
        n_modes = len(gains)
        n_steps = 10

        # Create SSR integrator
        ssr_data = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create IIR integrator (ff=1.0 for pure integration, no forgetting)
        iir_integrator = Integrator(simul_params, int_gain=gains, ff=None,
                                   target_device_idx=target_device_idx)

        # Create constant input
        input_value = BaseValue(value=xp.ones(n_modes, dtype=xp.float32),
                               target_device_idx=target_device_idx)

        # Connect inputs
        ssr_filter.inputs['delta_comm'].set(input_value)
        iir_integrator.inputs['delta_comm'].set(input_value)

        ssr_filter.setup()
        iir_integrator.setup()

        # Run simulation
        for step in range(n_steps):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t

            # SSR filter
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            # IIR filter
            iir_integrator.check_ready(t)
            iir_integrator.trigger()
            iir_integrator.post_trigger()

            # Compare outputs
            ssr_output = cpuArray(ssr_filter.outputs['out_comm'].value)
            iir_output = cpuArray(iir_integrator.outputs['out_comm'].value)

            np.testing.assert_allclose(ssr_output, iir_output, rtol=1e-6,
                                      err_msg=f"Outputs differ at step {step}")

    @cpu_and_gpu
    def test_integrator_equivalence_varying_input(self, target_device_idx, xp):
        """Test equivalence with time-varying input"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step

        # Parameters
        gains = [0.5, 1.0]
        n_modes = len(gains)
        n_steps = 20

        # Create SSR integrator
        ssr_data = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create IIR integrator
        iir_integrator = Integrator(simul_params, int_gain=gains, ff=None,
                                   target_device_idx=target_device_idx)

        # Create input (will vary over time)
        input_value = BaseValue(value=xp.zeros(n_modes, dtype=xp.float32),
                               target_device_idx=target_device_idx)

        # Connect inputs
        ssr_filter.inputs['delta_comm'].set(input_value)
        iir_integrator.inputs['delta_comm'].set(input_value)

        ssr_filter.setup()
        iir_integrator.setup()

        # Run simulation with varying input (sinusoidal)
        for step in range(n_steps):
            t = ssr_filter.seconds_to_t(step * dt)

            # Vary input: sin wave with different frequencies per mode
            input_array = np.array([
                np.sin(2 * np.pi * 10 * step * dt),  # 10 Hz
                np.sin(2 * np.pi * 20 * step * dt)   # 20 Hz
            ], dtype=np.float32)

            input_value.value = xp.array(input_array)
            input_value.generation_time = t

            # SSR filter
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            # IIR filter
            iir_integrator.check_ready(t)
            iir_integrator.trigger()
            iir_integrator.post_trigger()

            # Compare outputs
            ssr_output = cpuArray(ssr_filter.outputs['out_comm'].value)
            iir_output = cpuArray(iir_integrator.outputs['out_comm'].value)

            np.testing.assert_allclose(ssr_output, iir_output, rtol=1e-6,
                                      err_msg=f"Outputs differ at step {step}")

    @cpu_and_gpu
    def test_integrator_with_forgetting_factor(self, target_device_idx, xp):
        """Test integrator with forgetting factor (leaky integrator)"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step

        # Parameters
        gains = [0.5]
        ff = [0.99]  # Forgetting factor < 1.0 (leaky integrator)
        n_steps = 50

        # Create SSR leaky integrator using from_integrator with ff
        ssr_data = SsrFilterData.from_integrator(gains, ff=ff,
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create IIR leaky integrator
        iir_integrator = Integrator(simul_params, int_gain=gains, ff=ff,
                                   target_device_idx=target_device_idx)

        # Create constant input
        input_value = BaseValue(value=xp.ones(1, dtype=xp.float32),
                               target_device_idx=target_device_idx)

        # Connect inputs
        ssr_filter.inputs['delta_comm'].set(input_value)
        iir_integrator.inputs['delta_comm'].set(input_value)

        ssr_filter.setup()
        iir_integrator.setup()

        # Run simulation
        for step in range(n_steps):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t

            # SSR filter
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            # IIR filter
            iir_integrator.check_ready(t)
            iir_integrator.trigger()
            iir_integrator.post_trigger()

            # Compare outputs
            ssr_output = cpuArray(ssr_filter.outputs['out_comm'].value)
            iir_output = cpuArray(iir_integrator.outputs['out_comm'].value)

            np.testing.assert_allclose(ssr_output, iir_output, rtol=1e-5,
                                      err_msg=f"Outputs differ at step {step}")

    @cpu_and_gpu
    def test_multi_mode_integrator_equivalence(self, target_device_idx, xp):
        """Test equivalence with multiple modes (like SOUL params)"""
        simul_params = SimulParams(time_step=0.000588)  # SOUL time step
        dt = simul_params.time_step

        # SOUL-like parameters (simplified)
        int_gains = [0.55, 0.45, 0.40, 0.35]
        ffs = [1.0, 0.999996, 0.999365, 0.977895]
        n_modes_list = [5, 5, 5, 5]  # Simplified for testing (total 20 modes)

        # Expand gains and ff according to n_modes
        gains_expanded = []
        ff_expanded = []
        for i, n in enumerate(n_modes_list):
            gains_expanded.extend([int_gains[i]] * n)
            ff_expanded.extend([ffs[i]] * n)

        n_modes_total = sum(n_modes_list)
        n_steps = 15

        # Create SSR filters - use factory method instead of manual construction
        ssr_data = SsrFilterData.from_integrator(
            gains_expanded, 
            ff=ff_expanded,
            target_device_idx=target_device_idx
        )
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        # Create IIR integrator
        iir_integrator = Integrator(simul_params,
                                   int_gain=int_gains,
                                   ff=ffs,
                                   n_modes=n_modes_list,
                                   target_device_idx=target_device_idx)

        # Create random input
        np.random.seed(42)

        input_value = BaseValue(value=xp.zeros(n_modes_total, dtype=xp.float32),
                               target_device_idx=target_device_idx)

        # Connect inputs
        ssr_filter.inputs['delta_comm'].set(input_value)
        iir_integrator.inputs['delta_comm'].set(input_value)

        ssr_filter.setup()
        iir_integrator.setup()

        # Run simulation with random input
        for step in range(n_steps):
            t = ssr_filter.seconds_to_t(step * dt)

            # Random input for each mode
            random_input = np.random.randn(n_modes_total).astype(np.float32)
            input_value.value = xp.array(random_input)
            input_value.generation_time = t

            # SSR filter
            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            # IIR filter
            iir_integrator.check_ready(t)
            iir_integrator.trigger()
            iir_integrator.post_trigger()

            # Compare outputs
            ssr_output = cpuArray(ssr_filter.outputs['out_comm'].value)
            iir_output = cpuArray(iir_integrator.outputs['out_comm'].value)

            # Use slightly relaxed tolerance for accumulated errors
            np.testing.assert_allclose(ssr_output, iir_output, rtol=1e-5, atol=1e-8,
                                      err_msg=f"Outputs differ at step {step}")

    @cpu_and_gpu
    def test_reset_equivalence(self, target_device_idx, xp):
        """Test that reset works identically for both implementations"""
        simul_params = SimulParams(time_step=0.001)
        dt = simul_params.time_step

        gains = [1.0]

        # Create filters
        ssr_data = SsrFilterData.from_integrator(gains,
                                                target_device_idx=target_device_idx)
        ssr_filter = SsrFilter(simul_params, ssr_data, target_device_idx=target_device_idx)

        iir_integrator = Integrator(simul_params, int_gain=gains, ff=None,
                                   target_device_idx=target_device_idx)

        # Create input
        input_value = BaseValue(value=xp.ones(1, dtype=xp.float32),
                               target_device_idx=target_device_idx)

        # Connect and setup
        ssr_filter.inputs['delta_comm'].set(input_value)
        iir_integrator.inputs['delta_comm'].set(input_value)
        ssr_filter.setup()
        iir_integrator.setup()

        # Run for a few steps
        for step in range(5):
            t = ssr_filter.seconds_to_t(step * dt)
            input_value.generation_time = t

            ssr_filter.check_ready(t)
            ssr_filter.trigger()
            ssr_filter.post_trigger()

            iir_integrator.check_ready(t)
            iir_integrator.trigger()
            iir_integrator.post_trigger()

        # Verify they have accumulated state
        ssr_before = cpuArray(ssr_filter.outputs['out_comm'].value)
        iir_before = cpuArray(iir_integrator.outputs['out_comm'].value)
        self.assertGreater(ssr_before[0], 0)
        self.assertGreater(iir_before[0], 0)

        # Reset both
        ssr_filter.reset_states()
        iir_integrator.reset_states()

        # Run one more step with zero input
        t = ssr_filter.seconds_to_t(5 * dt)
        input_value.value = xp.zeros(1, dtype=xp.float32)
        input_value.generation_time = t

        ssr_filter.check_ready(t)
        ssr_filter.trigger()
        ssr_filter.post_trigger()

        iir_integrator.check_ready(t)
        iir_integrator.trigger()
        iir_integrator.post_trigger()

        # Both should output zero
        ssr_after = cpuArray(ssr_filter.outputs['out_comm'].value)
        iir_after = cpuArray(iir_integrator.outputs['out_comm'].value)

        np.testing.assert_almost_equal(ssr_after[0], 0.0)
        np.testing.assert_almost_equal(iir_after[0], 0.0)
        np.testing.assert_allclose(ssr_after, iir_after, rtol=1e-10)
