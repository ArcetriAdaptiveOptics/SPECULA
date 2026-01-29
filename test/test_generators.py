import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.simul_params import SimulParams
from specula.lib.modal_pushpull_signal import modal_pushpull_signal
from specula.processing_objects.push_pull_generator import PushPullGenerator
from specula.processing_objects.random_generator import RandomGenerator
from specula.processing_objects.schedule_generator import ScheduleGenerator
from specula.data_objects.time_history import TimeHistory
from specula.processing_objects.time_history_generator import TimeHistoryGenerator
from specula.processing_objects.vibration_generator import VibrationGenerator
from specula.processing_objects.wave_generator import WaveGenerator

from test.specula_testlib import cpu_and_gpu

class TestGenerators(unittest.TestCase):

    @cpu_and_gpu
    def test_func_generator_constant(self, target_device_idx, xp):
        constant = [4,3]
        f = WaveGenerator('SIN', target_device_idx=target_device_idx, constant=constant)
        f.setup()

        for t in [f.seconds_to_t(x) for x in [0.1, 0.2, 0.3, 0.4, 0.5]]:
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            np.testing.assert_allclose(value, constant)

    @cpu_and_gpu
    def test_func_generator_sin(self, target_device_idx, xp):
        amp = 1
        freq = 2
        offset = 3
        constant = 4
        f = WaveGenerator('SIN', amp=amp, freq=freq, offset=offset, constant=constant, target_device_idx=target_device_idx)
        f.setup()

        # Test twice in order to test streams capture, if enabled
        for t in [f.seconds_to_t(x) for x in [0.1, 0.2, 0.3]]:
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            np.testing.assert_almost_equal(value, amp * np.sin(freq*2 * np.pi*f.t_to_seconds(t) + offset) + constant)

    @cpu_and_gpu
    def test_wave_generator_square(self, target_device_idx, xp):
        amp = 2.0
        freq = 1.0
        f = WaveGenerator('SQUARE', amp=amp, freq=freq, target_device_idx=target_device_idx)
        f.setup()

        # Test at different phases
        value = []
        for t_sec in [0.0, 0.25, 0.5, 0.75]:
            t = f.seconds_to_t(t_sec)
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value.append(cpuArray(f.outputs['output'].value).copy())

        # Square wave should be +amp or -amp
        np.testing.assert_almost_equal(max(value), amp)
        np.testing.assert_almost_equal(min(value), -amp)

    @cpu_and_gpu
    def test_wave_generator_linear(self, target_device_idx, xp):
        """Test WaveGenerator linear functionality"""
        slope = 2.0
        constant = 1.0
        f = WaveGenerator('SIN', slope=slope, constant=constant,
                        target_device_idx=target_device_idx)
        f.setup()

        # Test multiple time points
        for t_sec in [0.1, 0.2, 0.3]:
            t = f.seconds_to_t(t_sec)
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            expected = slope * t_sec + constant
            np.testing.assert_almost_equal(value, expected)

    @cpu_and_gpu
    def test_wave_generator_vsize(self, target_device_idx, xp):
        """Test WaveGenerator vsize parameter"""
        slope = 2.0
        constant = 1.0
        f = WaveGenerator('SIN', slope=slope, constant=constant,
                        target_device_idx=target_device_idx)
        assert f.outputs['output'].value.shape == (1,)

        f = WaveGenerator('SIN', slope=slope, vsize=10, constant=constant,
                        target_device_idx=target_device_idx)
        assert f.outputs['output'].value.shape == (10,)

    @cpu_and_gpu
    def test_random_generator_normal(self, target_device_idx, xp):
        amp = 1.0
        constant = 2.0
        seed = 42
        output_size = 100

        f = RandomGenerator(distribution='NORMAL', amp=amp, constant=constant, 
                           seed=seed, output_size=output_size,
                           target_device_idx=target_device_idx)
        f.setup()

        # Generate multiple samples
        samples = []
        for i in range(10):
            f.check_ready(i)
            f.trigger()
            f.post_trigger()
            samples.append(cpuArray(f.outputs['output'].value))

        all_samples = np.concatenate(samples)

        # Check that mean is close to constant
        np.testing.assert_allclose(np.mean(all_samples), constant, atol=0.2)

        # Check that std is close to amp
        np.testing.assert_allclose(np.std(all_samples), amp, atol=0.2)

    @cpu_and_gpu
    def test_random_generator_uniform(self, target_device_idx, xp):
        amp = 2.0
        constant = 1.0
        seed = 123
        output_size = 50

        f = RandomGenerator(distribution='UNIFORM', amp=amp, constant=constant,
                           seed=seed, output_size=output_size,
                           target_device_idx=target_device_idx)
        f.setup()

        f.check_ready(1)
        f.trigger()
        f.post_trigger()

        values = cpuArray(f.outputs['output'].value)
        
        # Uniform distribution should be in [constant - amp/2, constant + amp/2]
        expected_min = constant - amp / 2
        expected_max = constant + amp / 2
        
        self.assertTrue(np.all(values >= expected_min))
        self.assertTrue(np.all(values <= expected_max))

    @cpu_and_gpu
    def test_random_generator_vsize(self, target_device_idx, xp):
        """Test RandomGenerator vsize parameter"""
        amp = 2.0
        constant = 1.0
        f = RandomGenerator(distribution='UNIFORM', amp=amp, constant=constant,
                        target_device_idx=target_device_idx)
        assert f.outputs['output'].value.shape == (1,)

        f = RandomGenerator(distribution='UNIFORM', amp=amp, constant=constant, vsize=10,
                        target_device_idx=target_device_idx)
        assert f.outputs['output'].value.shape == (10,)

    @cpu_and_gpu
    def test_vibration(self, target_device_idx, xp):
        nmodes = 2
        # it is a vector of 500 elements from 1 to 500
        freq = np.linspace(1, 500, 500)
        # there are 2 peaks at 10 and 20 Hz smoothed with a gaussian
        psd = np.zeros((nmodes, len(freq)))
        psd[0, :] = np.exp(-((freq - 10) ** 2) / (2 * (1 ** 2)))
        psd[1, :] = np.exp(-((freq - 20) ** 2) / (2 * (1 ** 2)))
        
        simulParams = SimulParams(time_step=0.001, total_time=1000.0)
        f = VibrationGenerator(simulParams, nmodes=nmodes, psd=psd, freq=freq, seed=1, target_device_idx=target_device_idx)
        f.setup()

        niter = int(simulParams.total_time / simulParams.time_step)
        self.assertEqual(f.time_hist.shape, (niter, nmodes))

        # variance of the signal
        var = np.zeros((nmodes,))
        for i in range(nmodes):
            var[i] = np.var(f.time_hist[:, i])
        # check that the variance is equal to the psd
        np.testing.assert_allclose(var[0], np.sum(psd[0, :]) * (freq[1] - freq[0]), rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(var[1], np.sum(psd[0, :]) * (freq[1] - freq[0]), rtol=2e-2, atol=1e-2)

        display = False
        if display:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(freq, psd[0, :], label='mode 1')
            plt.plot(freq, psd[1, :], label='mode 2')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD')
            plt.legend()
            plt.figure()
            plt.plot(f.time_hist[:, 0], label='mode 1')
            plt.plot(f.time_hist[:, 1], label='mode 2')
            plt.legend()
            plt.show()
        
    @cpu_and_gpu
    def test_vibration_wrong_size(self, target_device_idx, xp):
        
        simulParams = SimulParams(time_step=0.001, total_time=1000.0)

        # PSD array too small
        with self.assertRaises(ValueError):
            _ = VibrationGenerator(simulParams, nmodes=5, psd=np.zeros((3, 10)), freq=np.zeros((10, 5)), target_device_idx=target_device_idx)

        # Freq array too small
        with self.assertRaises(ValueError):
            _ = VibrationGenerator(simulParams, nmodes=5, psd=np.zeros((5, 10)), freq=np.zeros((10, 3)), target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_time_history_2d(self, target_device_idx, xp):

        niters = 3
        data = xp.arange(niters * 4).reshape((niters, 4))
        time_hist = TimeHistory(data, target_device_idx=target_device_idx)

        f = TimeHistoryGenerator(time_hist, target_device_idx=target_device_idx)

        # Test first frame
        for iter in range(niters):
            f.check_ready(iter)
            f.trigger()
            f.post_trigger()
            value = f.outputs['output'].value
            np.testing.assert_allclose(cpuArray(value), cpuArray(data[iter]))

        # Test beyond data (should use last values)
        f.check_ready(iter+1)
        f.trigger()
        f.post_trigger()
        value = f.outputs['output'].value
        np.testing.assert_allclose(cpuArray(value), cpuArray(data[-1]))

    @cpu_and_gpu
    def test_time_history_1d(self, target_device_idx, xp):

        niters = 3
        data = xp.arange(niters)
        time_hist = TimeHistory(data, target_device_idx=target_device_idx)

        f = TimeHistoryGenerator(time_hist, target_device_idx=target_device_idx)

        # Test first frame
        for iter in range(niters):
            f.check_ready(iter)
            f.trigger()
            f.post_trigger()
            value = f.outputs['output'].value
            np.testing.assert_allclose(cpuArray(value), cpuArray(data[iter]))

        # Test beyond data (should use last values)
        f.check_ready(iter+1)
        f.trigger()
        f.post_trigger()
        value = f.outputs['output'].value
        np.testing.assert_allclose(cpuArray(value), cpuArray(data[-1]))

    @cpu_and_gpu
    def test_schedule_generator(self, target_device_idx, xp):
        scheduled_values = [
            [0.1, 0.0],    # Values for t < 0.1s
            [0.5, 0.2],    # Values for 0.1s ≤ t < 0.3s
            [1.0, 0.8]     # Values for t ≥ 0.3s
        ]
        scheduled_times = [0.1, 0.3]
        modes_per_group = [2, 3]  # 2 modes for first value, 3 for second

        f = ScheduleGenerator(
            scheduled_values=scheduled_values,
            scheduled_times=scheduled_times,
            modes_per_group=modes_per_group,
            target_device_idx=target_device_idx
        )
        f.setup()

        # Test t = 0.05s (first interval)
        t1 = f.seconds_to_t(0.05)
        f.check_ready(t1)
        f.trigger()
        f.post_trigger()
        expected1 = [0.1, 0.1, 0.0, 0.0, 0.0]  # Expanded according to modes_per_group
        np.testing.assert_allclose(cpuArray(f.outputs['output'].value), expected1)

        # Test t = 0.2s (second interval)
        t2 = f.seconds_to_t(0.2)
        f.check_ready(t2)
        f.trigger()
        f.post_trigger()
        expected2 = [0.5, 0.5, 0.2, 0.2, 0.2]
        np.testing.assert_allclose(cpuArray(f.outputs['output'].value), expected2)

        # Test t = 0.5s (third interval)
        t3 = f.seconds_to_t(0.5)
        f.check_ready(t3)
        f.trigger()
        f.post_trigger()
        expected3 = [1.0, 1.0, 0.8, 0.8, 0.8]
        np.testing.assert_allclose(cpuArray(f.outputs['output'].value), expected3)

    @cpu_and_gpu
    def test_push_pull_generator(self, target_device_idx, xp):
        nmodes = 3
        amp = 0.5
        ncycles = 2

        f = PushPullGenerator(
            nmodes=nmodes,
            push_pull_type='PUSHPULL',
            amp=amp,
            ncycles=ncycles,
            target_device_idx=target_device_idx
        )
        f.setup()

        # Test multiple frames
        outputs = []
        for i in range(10):
            f.check_ready(i)
            f.trigger()
            f.post_trigger()
            outputs.append(f.outputs['output'].value.copy())

        # Check agains reference signal
        hist = modal_pushpull_signal(n_modes=nmodes, amplitude=amp, ncycles=ncycles, xp=np)
        for i in range(10):
            np.testing.assert_array_equal(cpuArray(outputs[i]), hist[i])

    @cpu_and_gpu
    def test_push_pull_generator_with_first_mode(self, target_device_idx, xp):
        nmodes = 8
        amp = 0.5
        ncycles = 2
        constant_amp = True
        first_mode = 2

        f = PushPullGenerator(
            nmodes=nmodes,
            first_mode=first_mode,
            push_pull_type='PUSHPULL',
            amp=amp,
            constant_amp=constant_amp,
            ncycles=ncycles,
            target_device_idx=target_device_idx
        )
        f.setup()

        # Test multiple frames
        outputs = []
        for i in range(10):
            f.check_ready(i)
            f.trigger()
            f.post_trigger()
            outputs.append(f.outputs['output'].value.copy())

        # Check agains reference signal
        hist = modal_pushpull_signal(n_modes=nmodes, first_mode=first_mode, amplitude=amp, constant=constant_amp, ncycles=ncycles, xp=np)
        for i in range(10):
            np.testing.assert_array_equal(cpuArray(outputs[i]), hist[i])

    @cpu_and_gpu
    def test_push_pull_invalid_type(self, target_device_idx, xp):

        with self.assertRaises(ValueError):
            _ = PushPullGenerator(nmodes=1, push_pull_type='INVALID')

    @cpu_and_gpu
    def test_func_generator_float(self, target_device_idx, xp):
        constant = [4,3]
        f = WaveGenerator('SIN', constant=constant, target_device_idx=target_device_idx, precision=1)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        assert f.outputs['output'].value.dtype == np.float32

    @cpu_and_gpu
    def test_func_generator_double(self, target_device_idx, xp):
        constant = [4,3]
        f = WaveGenerator('SIN', constant=constant, target_device_idx=target_device_idx, precision=0)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        assert f.outputs['output'].value.dtype == np.float64

    @cpu_and_gpu
    def test_output_size_consistency(self, target_device_idx, xp):
        # Test scalar parameters (should give output_size=1)
        wave_gen = WaveGenerator('SIN', amp=1.0, freq=2.0, target_device_idx=target_device_idx)
        wave_gen.setup()
        self.assertEqual(wave_gen.output.value.shape[0], 1)

        # Test array parameters (should give output_size=len(array))
        wave_gen_array = WaveGenerator('SIN', amp=[1.0, 2.0, 3.0], target_device_idx=target_device_idx)
        wave_gen_array.setup()
        self.assertEqual(wave_gen_array.output.value.shape[0], 3)

        # Test explicit output_size
        rand_gen = RandomGenerator(output_size=5, target_device_idx=target_device_idx)
        rand_gen.setup()
        self.assertEqual(rand_gen.output.value.shape[0], 5)

    @cpu_and_gpu
    def test_array_size_validation(self, target_device_idx, xp):
        # This should work (same size arrays)
        try:
            wave_gen = WaveGenerator('SIN', amp=[1.0, 2.0], freq=[3.0, 4.0], 
                                    target_device_idx=target_device_idx)
            wave_gen.setup()
        except ValueError:
            self.fail("Should not raise ValueError for same-size arrays")

        # This should fail (different size arrays)
        with self.assertRaises(ValueError):
            wave_gen = WaveGenerator('SIN', amp=[1.0, 2.0], freq=[3.0, 4.0, 5.0], 
                                    target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_time_progression(self, target_device_idx, xp):
        f = WaveGenerator('SIN', freq=1.0, target_device_idx=target_device_idx)
        f.setup()

        # Check initial state
        self.assertEqual(f.iter_counter, 0)

        # Progress through several frames
        for i in range(5):
            t = f.seconds_to_t(i * 0.001)
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            self.assertEqual(f.iter_counter, i + 1)

    def test_error_conditions(self):       
        # ScheduleGenerator: wrong length of scheduled_values vs scheduled_times
        with self.assertRaises(ValueError):
            ScheduleGenerator(
                scheduled_values=[[1.0], [2.0]],  # 2 value sets
                scheduled_times=[0.1, 0.2, 0.3],  # 3 times (should be 1)
                modes_per_group=[1]
            )
        
        # PushPullGenerator: missing amplitude
        with self.assertRaises(ValueError):
            PushPullGenerator(nmodes=2)  # No amp or vect_amplitude
        
        # RandomGenerator: invalid distribution
        with self.assertRaises(ValueError):
            gen = RandomGenerator(distribution='invalid')
            gen.trigger_code()
        
        # WaveGenerator: invalid wave type
        with self.assertRaises(ValueError):
            gen = WaveGenerator(wave_type='invalid')
            gen.trigger_code()
