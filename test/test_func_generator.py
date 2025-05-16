
import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.processing_objects.func_generator import FuncGenerator
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestFuncGenerator(unittest.TestCase):

    @cpu_and_gpu
    def test_func_generator_constant(self, target_device_idx, xp):
        simulParams = SimulParams(time_step=0.001)
        constant = [4,3]
        f = FuncGenerator(simulParams, 'SIN', target_device_idx=target_device_idx, constant=constant)
        f.check_ready(1)
        f.trigger()
        f.post_trigger()
        value = cpuArray(f.outputs['output'].value)
        np.testing.assert_allclose(value, constant)

    @cpu_and_gpu
    def test_func_generator_sin(self, target_device_idx, xp):
        simulParams = SimulParams(time_step=0.001)
        amp = 1
        freq = 2
        offset = 3
        constant = 4
        f = FuncGenerator(simulParams, 'SIN', target_device_idx=target_device_idx, amp=amp, freq=freq, offset=offset, constant=constant)
        f.setup()

        # Test twice in order to test streams capture, if enabled
        for t in [f.seconds_to_t(x) for x in [0.1, 0.2, 0.3]]:
            f.check_ready(t)
            f.trigger()
            f.post_trigger()
            value = cpuArray(f.outputs['output'].value)
            np.testing.assert_almost_equal(value, amp * np.sin(freq*2 * np.pi*f.t_to_seconds(t) + offset) + constant)

    @cpu_and_gpu
    def test_vibration(self, target_device_idx, xp):
        nmodes = 2
        # it is a vector of 500 elements from 1 to 500
        fr_psd = np.linspace(1, 500, 500)
        # there are 2 peaks at 10 and 20 Hz smoothed with a gaussian
        psd = np.zeros((nmodes, len(fr_psd)))
        psd[0, :] = np.exp(-((fr_psd - 10) ** 2) / (2 * (1 ** 2)))
        psd[1, :] = np.exp(-((fr_psd - 20) ** 2) / (2 * (1 ** 2)))
        simulParams = SimulParams(time_step=0.001, total_time=1000.0)
        f = FuncGenerator(simulParams, 'VIB_PSD', target_device_idx=target_device_idx, fr_psd=fr_psd, psd=psd, nmodes=nmodes, seed=1)
        f.setup()

        niter = int(simulParams.total_time / simulParams.time_step)
        self.assertEqual(f.time_hist.shape, (niter, nmodes))

        # variance of the signal
        var = np.zeros((nmodes,))
        for i in range(nmodes):
            var[i] = np.var(f.time_hist[:, i])
        # check that the variance is equal to the psd
        np.testing.assert_allclose(var[0], np.sum(psd[0, :]) * (fr_psd[1] - fr_psd[0]), rtol=2e-2, atol=1e-2)
        np.testing.assert_allclose(var[1], np.sum(psd[0, :]) * (fr_psd[1] - fr_psd[0]), rtol=2e-2, atol=1e-2)

        display = False
        if display:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(fr_psd, psd[0, :], label='mode 1')
            plt.plot(fr_psd, psd[1, :], label='mode 2')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('PSD')
            plt.legend()
            plt.figure()
            plt.plot(f.time_hist[:, 0], label='mode 1')
            plt.plot(f.time_hist[:, 1], label='mode 2')
            plt.legend()
            plt.show()

