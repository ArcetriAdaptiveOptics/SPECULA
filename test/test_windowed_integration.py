import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.windowed_integration import WindowedIntegration
from specula.processing_objects.wave_generator import WaveGenerator
from specula.data_objects.simul_params import SimulParams
from specula.base_value import BaseValue
from specula import cpuArray, np

from test.specula_testlib import cpu_and_gpu


class TestWindowedIntegration(unittest.TestCase):

    @cpu_and_gpu
    def test_windowed_integration_wrong_dt(self, target_device_idx, xp):
        """Test that non-multiple dt raises ValueError"""
        simul_params = SimulParams(time_step=2)

        # A non-multiple of time_step raises ValueError
        with self.assertRaises(ValueError):
            WindowedIntegration(simul_params, n_elem=10, dt=5,
                              target_device_idx=target_device_idx)

        # A multiple of time_step does not raise
        _ = WindowedIntegration(simul_params, n_elem=10, dt=4,
                              target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_windowed_integration_zero_dt(self, target_device_idx, xp):
        """Test that zero or negative dt raises ValueError"""
        simul_params = SimulParams(time_step=1)

        with self.assertRaises(ValueError):
            WindowedIntegration(simul_params, n_elem=10, dt=0,
                              target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            WindowedIntegration(simul_params, n_elem=10, dt=-1,
                              target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_windowed_integration_raises_on_missing_input(self, target_device_idx, xp):
        """Test that setup raises if input is not connected"""
        simul_params = SimulParams(time_step=1)
        integrator = WindowedIntegration(simul_params, n_elem=10, dt=1,
                                        target_device_idx=target_device_idx)

        # Raises because of missing input
        with self.assertRaises(ValueError):
            integrator.setup()

        # Connect input
        input_value = BaseValue(value=xp.zeros(10), target_device_idx=target_device_idx)
        integrator.inputs['input'].set(input_value)

        # Does not raise anymore
        integrator.setup()

    @cpu_and_gpu
    def test_windowed_integration_basic(self, target_device_idx, xp):
        """Test basic integration with constant input"""
        time_step = 0.1
        dt = 0.5  # Integration window
        n_elem = 5
        simul_params = SimulParams(time_step=time_step)

        # Create integrator
        integrator = WindowedIntegration(simul_params, n_elem=n_elem, dt=dt,
                                        target_device_idx=target_device_idx)

        # Create constant input
        input_value = BaseValue(value=xp.ones(n_elem), target_device_idx=target_device_idx)
        input_value.generation_time = 0
        integrator.inputs['input'].set(input_value)
        integrator.setup()

        # Run simulation for one integration window
        steps = int(dt / time_step)  # 5 steps
        for step in range(steps):
            current_time = integrator.seconds_to_t(step * time_step)
            input_value.generation_time = current_time

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

        # After 5 steps of dt=0.1 each, with input=1, integrated value should be 1
        # (sum of 5 * 0.1 * 1 / 0.5 = 1)
        expected = np.ones(n_elem)
        np.testing.assert_allclose(cpuArray(integrator.output.value), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_windowed_integration_multiple_windows(self, target_device_idx, xp):
        """Test multiple integration windows"""
        time_step = 0.1
        dt = 0.3  # Integration window
        n_elem = 3
        simul_params = SimulParams(time_step=time_step)

        integrator = WindowedIntegration(simul_params, n_elem=n_elem, dt=dt,
                                        target_device_idx=target_device_idx)

        # Create input that increments at each step
        input_value = BaseValue(value=xp.zeros(n_elem), target_device_idx=target_device_idx)
        integrator.inputs['input'].set(input_value)
        integrator.setup()

        # Run for 2 complete windows (6 steps)
        steps = 6
        outputs = []

        for step in range(steps):
            current_time = integrator.seconds_to_t(step * time_step)
            input_value.value[:] = step + 1  # Input = 1, 2, 3, 4, 5, 6
            input_value.generation_time = current_time

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

            # Store output at end of each window (steps 2 and 5)
            if step in [2, 5]:
                outputs.append(integrator.output.value.copy())

        # First window: (1*0.1 + 2*0.1 + 3*0.1) / 0.3 = 6*0.1/0.3 = 2
        # Second window: (4*0.1 + 5*0.1 + 6*0.1) / 0.3 = 15*0.1/0.3 = 5
        expected_first = np.ones(n_elem) * 2.0
        expected_second = np.ones(n_elem) * 5.0

        np.testing.assert_allclose(cpuArray(outputs[0]), expected_first, rtol=1e-6)
        np.testing.assert_allclose(cpuArray(outputs[1]), expected_second, rtol=1e-6)

    @cpu_and_gpu
    def test_windowed_integration_with_start_time(self, target_device_idx, xp):
        """Test integration with delayed start_time"""
        time_step = 0.1
        dt = 0.2
        start_time = 0.3  # Start after 3 steps
        n_elem = 2
        simul_params = SimulParams(time_step=time_step)

        integrator = WindowedIntegration(simul_params, n_elem=n_elem, dt=dt,
                                        start_time=start_time,
                                        target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.ones(n_elem), target_device_idx=target_device_idx)
        integrator.inputs['input'].set(input_value)
        integrator.setup()

        # Run for 5 steps
        for step in range(5):
            current_time = integrator.seconds_to_t(step * time_step)
            input_value.generation_time = current_time

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

        # Integration should start at step 3, so only steps 3,4 contribute
        # (1*0.1 + 1*0.1) / 0.2 = 1
        expected = np.ones(n_elem)
        np.testing.assert_allclose(cpuArray(integrator.output.value), expected, rtol=1e-6)

    @cpu_and_gpu
    def test_windowed_integration_update_time_on_dt_true(self, target_device_idx, xp):
        """Test that generation_time is updated only at window completion when update_time_on_dt=True"""
        time_step = 0.1
        dt = 0.2
        n_elem = 2
        simul_params = SimulParams(time_step=time_step)

        integrator = WindowedIntegration(simul_params, n_elem=n_elem, dt=dt,
                                        update_time_on_dt=True,
                                        target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.ones(n_elem), target_device_idx=target_device_idx)
        integrator.inputs['input'].set(input_value)
        integrator.setup()

        integrator.output.generation_time = integrator.seconds_to_t(0)

        # Run for 3 steps
        generation_times = []
        for step in range(3):
            current_time = integrator.seconds_to_t(step * time_step)
            input_value.generation_time = current_time

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

            generation_times.append(float(integrator.output.generation_time))

        # With update_time_on_dt=True, generation_time should only update at step 1
        # (when window completes at t=0.2)
        expected_times = [0, float(integrator.seconds_to_t(0.1)), float(integrator.seconds_to_t(0.1))]
        self.assertEqual(generation_times, expected_times)

    @cpu_and_gpu
    def test_windowed_integration_update_time_on_dt_false(self, target_device_idx, xp):
        """Test that generation_time is updated every step when update_time_on_dt=False"""
        time_step = 0.1
        dt = 0.2
        n_elem = 2
        simul_params = SimulParams(time_step=time_step)

        integrator = WindowedIntegration(simul_params, n_elem=n_elem, dt=dt,
                                        target_device_idx=target_device_idx)

        input_value = BaseValue(value=xp.ones(n_elem), target_device_idx=target_device_idx)
        integrator.inputs['input'].set(input_value)
        integrator.setup()

        # Run for 3 steps
        generation_times = []
        for step in range(3):
            current_time = integrator.seconds_to_t(step * time_step)
            input_value.generation_time = current_time

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

            generation_times.append(float(integrator.output.generation_time))

        # With update_time_on_dt=False, generation_time should update every step
        expected_times = [float(integrator.seconds_to_t(i * time_step)) for i in range(3)]
        self.assertEqual(generation_times, expected_times)

    @cpu_and_gpu
    def test_windowed_integration_with_wave_generator(self, target_device_idx, xp):
        """Integration test with WaveGenerator"""
        time_step = 0.1
        dt = 0.5
        simul_params = SimulParams(time_step=time_step)

        # Create wave generator with constant output (freq=0)
        generator = WaveGenerator(target_device_idx=target_device_idx, constant=2, freq=0)
        generator.setup()

        # Create integrator
        integrator = WindowedIntegration(simul_params, n_elem=1, dt=dt,
                                         update_time_on_dt=True,
                                         target_device_idx=target_device_idx)
        integrator.inputs['input'].set(generator.outputs['output'])
        integrator.setup()

        # Run for one complete window (5 steps)
        for step in range(5):
            current_time = integrator.seconds_to_t(step * time_step)

            generator.current_time = current_time
            generator.check_ready(current_time)
            generator.trigger()
            generator.post_trigger()

            integrator.check_ready(current_time)
            integrator.trigger()
            integrator.post_trigger()

        # With constant input of 2, integrated over 0.5s: 2 * 0.5 / 0.5 = 2
        expected = np.array([2.0])
        np.testing.assert_allclose(cpuArray(integrator.output.value), expected, rtol=1e-6)
