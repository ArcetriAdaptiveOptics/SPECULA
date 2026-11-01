import specula
specula.init(0)  # Default target device
from specula.loop_control import LoopControl

import unittest

import numpy as np

from specula.processing_objects.ccd import CCD
from specula.data_objects.intensity import Intensity
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestCCD(unittest.TestCase):

    def _accelerated_run(self, accelerate, steps=6, **noise):
        '''CCD reading out at each step a changing intensity; return its frames and the CCD'''
        from iteras.accelerator import Accelerator
        from iteras.data_objects.accelerator import Accelerator as AcceleratorOptions
        xp = specula.cp
        simul_params = SimulParams(time_step=0.001)
        ccd = CCD(simul_params, size=(16, 16), dt=0.001, bandw=300, target_device_idx=0, **noise)
        i = Intensity(dimx=16, dimy=16, target_device_idx=0)
        ccd.inputs['in_i'].set(i)
        if accelerate:
            Accelerator(AcceleratorOptions()).apply([ccd])
        ccd.setup()
        frames = []
        for step in range(steps):
            t = ccd.seconds_to_t(0.001 * step)
            i.i[:] = xp.arange(256, dtype=xp.float32).reshape(16, 16) * (1 + step)
            i.generation_time = t
            ccd.check_ready(t)
            ccd.trigger()
            ccd.post_trigger()
            frames.append(specula.cpuArray(ccd._pixels.pixels).copy())
        return frames, ccd

    @unittest.skipIf(specula.cp is None, 'Cupy not found')
    def test_accelerated_with_readout_noise(self):
        # The readout noise comes from a generator whose state is on the GPU: the
        # accelerated CCD gives the same frames, with new noise at each step
        ref, _ = self._accelerated_run(False, readout_noise=True, readout_level=5.0)
        out, ccd = self._accelerated_run(True, readout_noise=True, readout_level=5.0)
        self.assertEqual(ccd._accel.status, 'accelerated')
        for a, b in zip(ref, out):
            np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-4)
        # The frames are not just the input scaled: there is noise
        self.assertFalse(np.allclose(out[4] / 5, out[5] / 6))

    @unittest.skipIf(specula.cp is None, 'Cupy not found')
    def test_not_accelerated_with_photon_noise(self):
        # Poisson random numbers cannot be captured: the CCD says so, and runs as Python code
        out, ccd = self._accelerated_run(True, photon_noise=True)
        self.assertIsNone(ccd._accel.graph)
        self.assertIn('photon noise', ccd._accel.status)

    @cpu_and_gpu
    def test_ccd_negative_dt(self, target_device_idx, xp):
        simul_params = SimulParams(time_step = 2)

        with self.assertRaises(ValueError):
            ccd = CCD(simul_params, size=(2,2), dt=-1, bandw=300,
                      target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_ccd_wrong_dt(self, target_device_idx, xp):
        simul_params = SimulParams(time_step = 2)

        # A non-multiple of time_step raises ValueError
        with self.assertRaises(ValueError):
            ccd = CCD(simul_params, size=(2,2), dt=5, bandw=300,
                      target_device_idx=target_device_idx)

        # A multiple of time_step does not raise
        _ = CCD(simul_params, size=(2,2), dt=4, bandw=300,
                      target_device_idx=target_device_idx)


    @cpu_and_gpu
    def test_time_conversion_corner_cases1(self, target_device_idx, xp):
        '''
        Test that modulus operation work on the result of seconds_to_t(),
        even when the original data would fail in floating point
        '''
        simul_params = SimulParams(time_step = 0.002)

        # naive floating-point would fail: 1.0 % 0.002 == 0.0019999999999999792
        # This code must not fail
        _ = CCD(simul_params, size=(2,2), dt=1.0, bandw=300,
                      target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_time_conversion_corner_cases2(self, target_device_idx, xp):
        '''
        Test that modulus operation work on the result of seconds_to_t(),
        even when the original data would fail in floating point
        '''
        simul_params = SimulParams(time_step = 0.0001)

        # naive floating-point would fail: 0.0005 % 0.0001 == 9.999999999999999e-05
        # This code must not fail
        _ = CCD(simul_params, size=(2,2), dt=0.0005, bandw=300,
                      target_device_idx=target_device_idx)


    @cpu_and_gpu
    def test_ccd_raises_on_missing_input(self, target_device_idx, xp):

        simul_params = SimulParams(time_step = 2)
        ccd = CCD(simul_params, size=(2,2), dt=2, bandw=300,
                       target_device_idx=target_device_idx)

        i = Intensity(dimx=2, dimy=2, target_device_idx=target_device_idx)

        # Raises because of missing input
        with self.assertRaises(ValueError):
            ccd.setup()

        ccd.inputs['in_i'].set(i)

        # Does not raise anymore
        ccd.setup()

    @cpu_and_gpu
    def test_ccd_excess_noise_trigger(self, target_device_idx, xp):
        t_seconds = 1

        average_i = 10.0
        emccd_gain = 400.0
        simul_params = SimulParams(time_step=t_seconds)
        ccd = CCD(
            simul_params,
            size=(10, 10),
            dt=t_seconds,
            bandw=1,
            excess_noise=True,
            emccd_gain=emccd_gain,
            target_device_idx=target_device_idx
        )
        i = Intensity(dimx=10, dimy=10, target_device_idx=target_device_idx)
        # Set up the input intensity with a known value
        i.i[:] = average_i
        i.generation_time = i.seconds_to_t(t_seconds)
        ccd.inputs['in_i'].set(i)

        loop = LoopControl()
        loop.add(ccd, idx=0)
        loop.run(run_time=t_seconds*2, dt=t_seconds, t0=t_seconds)
        
        # Check that the average pixel value is close to the expected value
        actual = float(xp.mean(ccd._pixels.pixels))
        expected = average_i * emccd_gain
        rel_tol = 0.1
        self.assertTrue(
            abs(actual - expected) / expected < rel_tol,
            f"Relative difference {abs(actual - expected) / expected:.2%} exceeds {rel_tol:.2%}"
        )

    @cpu_and_gpu
    def test_ccd_noise_trigger(self, target_device_idx, xp):
        t_seconds = 1

        average_i = 10.0
        emccd_gain = 400.0
        readout_level = 1.0
        ADU_gain = 1.0
        simul_params = SimulParams(time_step=t_seconds)
        ccd = CCD(
            simul_params,
            size=(10, 10),
            dt=t_seconds,
            bandw=1,
            photon_noise=True,
            excess_noise=True,
            readout_noise=True,
            readout_level=readout_level,
            emccd_gain=emccd_gain,
            ADU_gain=ADU_gain,
            target_device_idx=target_device_idx
        )
        i = Intensity(dimx=10, dimy=10, target_device_idx=target_device_idx)
        # Set up the input intensity with a known value
        i.i[:] = average_i
        i.generation_time = i.seconds_to_t(t_seconds)
        ccd.inputs['in_i'].set(i)

        loop = LoopControl()
        loop.add(ccd, idx=0)
        loop.run(run_time=t_seconds*2, dt=t_seconds, t0=t_seconds)

        # Check that the average pixel value is close to the expected value
        actual = float(xp.mean(ccd._pixels.pixels))
        expected = average_i
        rel_tol = 0.20
        self.assertTrue(
            abs(actual - expected) / expected < rel_tol,
            f"Relative difference {abs(actual - expected) / expected:.2%} exceeds {rel_tol:.2%}"
        )
