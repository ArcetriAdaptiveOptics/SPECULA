import specula
specula.init(0)  # Default target device

import unittest

from specula.processing_objects.ccd import CCD
from specula.data_objects.intensity import Intensity
from specula.data_objects.simul_params import SimulParams

from test.specula_testlib import cpu_and_gpu

class TestCCD(unittest.TestCase):

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
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step
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
        i.generation_time = t
        ccd.inputs['in_i'].set(i)
        ccd.loop_dt = t
        ccd.setup()
        # Execute the trigger method
        ccd.check_ready(t)
        ccd.trigger()
        ccd.post_trigger()
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
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step
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
        i.generation_time = t
        ccd.inputs['in_i'].set(i)
        ccd.loop_dt = t
        ccd.setup()
        # Execute the trigger method
        ccd.check_ready(t)
        ccd.trigger()
        ccd.post_trigger()
        # Check that the average pixel value is close to the expected value
        actual = float(xp.mean(ccd._pixels.pixels))
        expected = average_i
        rel_tol = 0.20
        self.assertTrue(
            abs(actual - expected) / expected < rel_tol,
            f"Relative difference {abs(actual - expected) / expected:.2%} exceeds {rel_tol:.2%}"
        )

if __name__ == '__main__':
    unittest.main()