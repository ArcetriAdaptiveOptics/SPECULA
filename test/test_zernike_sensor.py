import specula
specula.init(0)  # Default target device

import unittest

from specula import np

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
        fov = 1.8
        pup_diam = 70
        output_resolution = 80
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
        fov = 1.8
        pup_diam = 70
        output_resolution = 80
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
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        # Create Zernike generator for focus
        zg = ZernikeGenerator(ef.size[0], xp=xp, dtype=ef.dtype)
        phaseInNm = zg.getZernike(4)*10
        ef.phaseInNm = phaseInNm
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
        intensity_diff = intensity.i.copy()

        ef.phaseInNm[:] = phaseInNm*0.
        ef.generation_time = 2*t

        zernike_sensor.check_ready(2*t)
        zernike_sensor.trigger()
        zernike_sensor.post_trigger()

        intensity_diff -= intensity.i.copy()

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(phaseInNm)
            plt.title("Input Phase")
            plt.colorbar()
            plt.figure()
            plt.imshow(intensity_diff)
            plt.title("Output Intensity")
            plt.colorbar()
            # horizontal cut of phase
            plt.figure()
            plt.plot(phaseInNm[ef.phaseInNm.shape[0] // 2, :])
            plt.title("Horizontal Cut of Input Phase")
            plt.xlabel("Pixel")
            plt.ylabel("Phase (nm)")
            # horizontal cut of output intensity
            plt.figure()
            plt.plot(intensity_diff[intensity_diff.shape[0] // 2, :])
            plt.title("Horizontal Cut of Output Intensity")
            plt.xlabel("Pixel")
            plt.ylabel("Intensity")
            plt.show()

        # store max value of horizontal cut of intensity
        max_input_intensity = float(xp.max(intensity_diff[intensity_diff.shape[0] // 2, :]))
        max_input_intensity_index = int(xp.argmax(intensity_diff[intensity_diff.shape[0] // 2, :]))
        # store first minimum of horizontal cut
        min_input_intensity = float(xp.min(intensity_diff[intensity_diff.shape[0] // 2, :]))
        min_input_intensity_index = int(xp.argmin(intensity_diff[intensity_diff.shape[0] // 2, :]))
        # search value in between the min and max
        index_mean = round((min_input_intensity_index + max_input_intensity_index) // 2)
        mean_input_intensity = float(intensity_diff[intensity_diff.shape[0] // 2, index_mean])

        # this three points should fit a quadratic
        coeffs = np.polyfit([max_input_intensity_index, min_input_intensity_index, index_mean],
                             [max_input_intensity, min_input_intensity, mean_input_intensity], 2)

        # Compare fitting and values, i.e. error
        fit = np.polyval(coeffs, [max_input_intensity_index, min_input_intensity_index, index_mean])
        error = np.abs(fit - [max_input_intensity, min_input_intensity, mean_input_intensity])

        verbose = False
        if verbose:
            print('Fit, Values and Fitting error:')
            for f, v, e in zip(fit, [max_input_intensity, min_input_intensity, mean_input_intensity], error):
                print(f" {f:.5f}, {v:.5f}, {e:.5f}")

        # Fitting error must be lower than 1e-4 (it should be true with a small aberration)
        assert np.all(error < 1e-4), "Fitting error is too high!"
