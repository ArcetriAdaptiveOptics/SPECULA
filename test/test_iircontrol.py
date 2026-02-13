import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.iir_filter_data import IirFilterData
from specula.processing_objects.iir_filter import IirFilter
from specula.processing_objects.integrator import Integrator
from specula.processing_objects.schedule_generator import ScheduleGenerator
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu

class TestIirFilter(unittest.TestCase):

    # We just check that it goes through.
    @cpu_and_gpu
    def test_iir_filter_instantiation(self, target_device_idx, xp):
        iir_filter = IirFilterData(ordnum=(1,1), ordden=(1,1),
                                   num=xp.ones((2,2)), den=xp.ones((2,2)),
                                   target_device_idx=target_device_idx)
        simulParams = SimulParams(time_step=0.001)
        iir_control = IirFilter(simulParams, iir_filter)

    @cpu_and_gpu
    def test_integrator_instantiation(self, target_device_idx, xp):
        simulParams = SimulParams(time_step=0.001)
        integrator = Integrator(simulParams,
                                int_gain=[0.5,0.4,0.3],
                                ff=[0.99,0.95,0.90],
                                n_modes= [2,3,4],
                                   target_device_idx=target_device_idx)
        # check that the iir_filter_data is set up correctly by comparing gain
        # and [0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3]
        self.assertEqual(np.sum(np.abs(cpuArray(integrator.iir_filter_data.gain) \
                         - np.array([0.5,0.5,0.4,0.4,0.4,0.3,0.3,0.3,0.3]))),0)

    @cpu_and_gpu
    def test_integrator_with_value_schedule_gain_mod(self, target_device_idx, xp):
        """
        Test integrator with VALUE_SCHEDULE gain_mod:
        - Create an integrator with int_gain=[0.5, 0.3] and modes_per_group=[1, 1] 
        - Create a VALUE_SCHEDULE that changes gain_mod from [1.0, 1.0]
          to [2.0, 0.5] at 3rd step (0.002s)
        - Apply constant input of 1.0 for 3 frames
        - Verify correct integration with varying gain_mod
        """
        verbose = False

        simulParams = SimulParams(time_step=0.001)

        # Create integrator: 2 modes with gains [0.5, 0.3]
        integrator = Integrator(simulParams, int_gain=[0.5, 0.3], n_modes=[1, 1],
                            target_device_idx=target_device_idx)

        # Create VALUE_SCHEDULE gain_mod that changes after 0.001s
        gain_mod_generator = ScheduleGenerator(
            scheduled_values=[
                [1.0, 1.0],  # gain_mod for t < 0.001s
                [2.0, 0.5]   # gain_mod for t >= 0.001s
            ],
            scheduled_times=[0.002],  # change at 0.002s
            modes_per_group=[1, 1],  # 1 mode per value
            target_device_idx=target_device_idx
        )

        # Create constant input of 1.0 for both modes
        constant_input = BaseValue(value=xp.array([1.0, 1.0], dtype=xp.float32),
                                target_device_idx=target_device_idx)

        # Connect inputs
        integrator.inputs['delta_comm'].set(constant_input)
        integrator.inputs['gain_mod'].set(gain_mod_generator.outputs['output'])

        # Setup objects
        gain_mod_generator.setup()
        integrator.setup()

        # Frame 0: t=0, gain_mod=[1.0, 1.0], int_gain=[0.5, 0.3]
        # Expected output: [0.5*1.0*1.0, 0.3*1.0*1.0] = [0.5, 0.3]
        t0 = 0
        constant_input.generation_time = t0
        gain_mod_generator.check_ready(t0)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        integrator.check_ready(t0)
        integrator.trigger()
        integrator.post_trigger()

        output_frame0 = cpuArray(integrator.outputs['out_comm'].value)
        expected_frame0 = np.array([0.5, 0.3])

        if verbose:
            print("input at t=0:", constant_input.value)
            print("Output at t=0:", integrator.outputs['out_comm'].value)
            print("Expected output at t=0:", expected_frame0)

        np.testing.assert_allclose(output_frame0, expected_frame0, rtol=1e-5)

        # Frame 1: t=0.001, gain_mod=[1.0, 1.0] (still first interval)
        # Previous state: [0.5, 0.3], new input: [0.5*0.5*1.0, 0.3*1.0*1.0] = [0.5, 0.3]
        # Expected output: [0.5+0.5, 0.3+0.3] = [1.0, 0.6]
        t1 = integrator.seconds_to_t(0.001)
        constant_input.generation_time = t1
        gain_mod_generator.check_ready(t1)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        integrator.check_ready(t1)
        integrator.trigger()
        integrator.post_trigger()

        output_frame1 = cpuArray(integrator.outputs['out_comm'].value)
        expected_frame1 = np.array([1.0, 0.6])

        if verbose:
            print("input at t=0.001:", constant_input.value)
            print("Output at t=0.001:", integrator.outputs['out_comm'].value)
            print("Expected output at t=0.001:", expected_frame1)

        np.testing.assert_allclose(output_frame1, expected_frame1, rtol=1e-5)

        # Frame 2: t=0.002, gain_mod=[2.0, 0.5] (second interval)
        # Previous state: [1.0, 0.6], new input: [0.5*1.0*2.0, 0.3*1.0*0.5] = [1.0, 0.15]
        # Expected output: [1.0+1.0, 0.6+0.15] = [2.0, 0.75]
        t2 = integrator.seconds_to_t(0.002)
        constant_input.generation_time = t2
        gain_mod_generator.check_ready(t2)
        gain_mod_generator.trigger()
        gain_mod_generator.post_trigger()

        integrator.check_ready(t2)
        integrator.trigger()
        integrator.post_trigger()

        output_frame2 = cpuArray(integrator.outputs['out_comm'].value)
        expected_frame2 = np.array([2.0, 0.75])

        if verbose:
            print("input at t=0.002:", constant_input.value)
            print("Output at t=0.002:", integrator.outputs['out_comm'].value)
            print("Expected output at t=0.002:", expected_frame2)

        np.testing.assert_allclose(output_frame2, expected_frame2, rtol=1e-5)

    @cpu_and_gpu
    def test_integrator_integration_disabled(self, target_device_idx, xp):
        """
        Test Integrator with integration=False (FIR mode):
        - Create an integrator with integration=False
        - Apply constant input
        - Verify output is constant (no accumulation)
        """
        simulParams = SimulParams(time_step=0.001)
        dt = simulParams.time_step

        # Create integrator with integration disabled
        integrator = Integrator(simulParams, int_gain=[0.5, 0.3], n_modes=[1, 1],
                               integration=False,
                               target_device_idx=target_device_idx)

        # Create constant input
        constant_input = BaseValue(value=xp.ones(2, dtype=xp.float32),
                                  target_device_idx=target_device_idx)

        integrator.inputs['delta_comm'].set(constant_input)
        integrator.setup()

        # Run for 5 steps
        outputs = []
        for step in range(5):
            t = integrator.seconds_to_t(step * dt)
            constant_input.generation_time = t

            integrator.check_ready(t)
            integrator.trigger()
            integrator.post_trigger()

            outputs.append(cpuArray(integrator.outputs['out_comm'].value))

        # With integration=False (FIR mode), output should be constant
        # Output = gain * input (no accumulation)
        expected = np.array([0.5, 0.3])

        # All outputs should be the same
        for i, output in enumerate(outputs):
            np.testing.assert_allclose(output, expected, rtol=1e-6,
                                      err_msg=f"Output differs at step {i}")

        # Verify no accumulation: last output should equal first
        np.testing.assert_allclose(outputs[-1], outputs[0], rtol=1e-10)

    @cpu_and_gpu
    def test_integrator_no_delay_output(self, target_device_idx, xp):
        """Test that out_comm_no_delay provides synchronous output for POLC"""
        simulParams = SimulParams(time_step=0.001)
        dt = simulParams.time_step
        delay = 1.0  # 1 frame delay

        integrator = Integrator(simulParams, int_gain=[0.5], n_modes=[1],
                               delay=delay,
                               target_device_idx=target_device_idx)

        constant_input = BaseValue(value=xp.array([1.0], dtype=xp.float32),
                                  target_device_idx=target_device_idx)

        integrator.inputs['delta_comm'].set(constant_input)
        integrator.setup()

        # Track both outputs over 3 frames
        no_delay_outputs = []
        delayed_outputs = []

        for step in range(3):
            t = integrator.seconds_to_t(step * dt)
            constant_input.generation_time = t
            integrator.check_ready(t)
            integrator.trigger()
            integrator.post_trigger()

            no_delay_outputs.append(cpuArray(integrator.outputs['out_comm_no_delay'].value)[0])
            delayed_outputs.append(cpuArray(integrator.outputs['out_comm'].value)[0])

        # No-delay: immediate accumulation [0.5, 1.0, 1.5]
        expected_no_delay = [0.5, 1.0, 1.5]
        np.testing.assert_allclose(no_delay_outputs, expected_no_delay, rtol=1e-6)

        # Delayed: outputs shifted by 1 frame [0.0, 0.5, 1.0]
        expected_delayed = [0.0, 0.5, 1.0]
        np.testing.assert_allclose(delayed_outputs, expected_delayed, rtol=1e-6)

    @cpu_and_gpu
    def test_integrator_no_delay_vs_delayed_zero_delay(self, target_device_idx, xp):
        """Test that both outputs are identical when delay=0"""
        simulParams = SimulParams(time_step=0.001)

        integrator = Integrator(simulParams, int_gain=[1.0], n_modes=[1],
                               delay=0.0,
                               target_device_idx=target_device_idx)

        constant_input = BaseValue(value=xp.array([2.0], dtype=xp.float32),
                                  target_device_idx=target_device_idx)

        integrator.inputs['delta_comm'].set(constant_input)
        integrator.setup()

        # Run for 2 steps
        for step in range(2):
            t = integrator.seconds_to_t(step * 0.001)
            constant_input.generation_time = t
            integrator.check_ready(t)
            integrator.trigger()
            integrator.post_trigger()

            delayed = cpuArray(integrator.outputs['out_comm'].value)[0]
            no_delay = cpuArray(integrator.outputs['out_comm_no_delay'].value)[0]

            # Both should be identical when delay=0
            np.testing.assert_almost_equal(delayed, no_delay, decimal=10)

    @cpu_and_gpu
    def test_integrator_fractional_delay_no_delay_independence(self, target_device_idx, xp):
        """Test that no_delay output is independent of fractional delay interpolation"""
        simulParams = SimulParams(time_step=0.001)
        dt = simulParams.time_step
        delay = 1.5  # Fractional delay

        integrator = Integrator(simulParams, int_gain=[1.0], n_modes=[1],
                               delay=delay,
                               target_device_idx=target_device_idx)

        # Apply sequence: 10, 20, 30
        inputs = [10.0, 20.0, 30.0]
        input_value = BaseValue(value=xp.array([0.0], dtype=xp.float32),
                               target_device_idx=target_device_idx)

        integrator.inputs['delta_comm'].set(input_value)
        integrator.setup()

        expected_no_delay = [10.0, 30.0, 60.0]  # Accumulation: 10, 10+20, 10+20+30

        for step, inp in enumerate(inputs):
            t = integrator.seconds_to_t(step * dt)
            input_value.value = xp.array([inp], dtype=xp.float32)
            input_value.generation_time = t
            integrator.check_ready(t)
            integrator.trigger()
            integrator.post_trigger()

            no_delay = cpuArray(integrator.outputs['out_comm_no_delay'].value)[0]

            # No-delay should reflect current accumulated state
            np.testing.assert_almost_equal(no_delay, expected_no_delay[step], decimal=5,
                                          err_msg=f"Step {step}: no_delay output mismatch")

        # Verify delayed output uses interpolation
        delayed = cpuArray(integrator.outputs['out_comm'].value)[0]
        # At step 2: delay=1.5 interpolates between buffer[1] and buffer[2]
        # buffer[2] = 10 (step 0), buffer[1] = 30 (step 1)
        # output = 0.5 * 10 + 0.5 * 30 = 20
        np.testing.assert_almost_equal(delayed, 20.0, decimal=5)
