import specula
specula.init(0)  # Default target device

import unittest

from specula import cp, np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.lib.zernike_generator import ZernikeGenerator
from specula.processing_objects.zernike_sensor import ZernikeSensor
from test.specula_testlib import cpu_and_gpu


class TestZernikeSensor(unittest.TestCase):

    @cpu_and_gpu
    def test_flat_wavefront_output_size(self, target_device_idx, xp):
        """Test that ZernikeSensor produces correct output dimensions for flat wavefront"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 70
        output_resolution = 80
        spot_radius_lambda = 0.5
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create Zernike sensor
        zernike_sensor = ZernikeSensor(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            spot_radius_lambda=spot_radius_lambda,
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        zernike_sensor.inputs['in_ef'].set(ef)

        # Setup and run
        zernike_sensor.setup()
        zernike_sensor.check_ready(t)
        zernike_sensor.trigger()
        zernike_sensor.post_trigger()

        # Get output intensity
        intensity = zernike_sensor.outputs['out_i']

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.imshow(intensity.i)
            plt.title("Output Intensity")
            plt.colorbar()
            plt.show()

        # Test 1: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match expected {expected_shape}")

        # Test 2: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

    @cpu_and_gpu
    def test_focus(self, target_device_idx, xp):
        """Test focus aberration on ZernikeSensor"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 70
        output_resolution = 80
        spot_radius_lambda = 1.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create Zernike sensor
        zernike_sensor = ZernikeSensor(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            spot_radius_lambda=spot_radius_lambda,
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        # Create Zernike generator for focus
        zg = ZernikeGenerator(ef.size[0], xp=xp, dtype=ef.dtype)
        ef.phaseInNm = zg.getZernike(4)*100.
        ef.generation_time = t

        # Connect input
        zernike_sensor.inputs['in_ef'].set(ef)

        # Setup and run
        zernike_sensor.setup()
        zernike_sensor.check_ready(t)
        zernike_sensor.trigger()
        zernike_sensor.post_trigger()

        # Get output intensity
        intensity = zernike_sensor.outputs['out_i']

        plot_debug = True
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(ef.phaseInNm)
            plt.title("Input Phase")
            plt.colorbar()
            plt.figure()
            plt.imshow(intensity.i)
            plt.title("Output Intensity")
            plt.colorbar()
            plt.show()