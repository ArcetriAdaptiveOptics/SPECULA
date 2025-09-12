import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.processing_objects.modulated_double_roof import ModulatedDoubleRoof
from test.specula_testlib import cpu_and_gpu


class TestModulatedDoubleRoof(unittest.TestCase):

    @cpu_and_gpu
    def test_flat_wavefront_output_size(self, target_device_idx, xp):
        """Test that ModulatedDoubleRoof produces correct output dimensions for flat wavefront"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        pup_dist = 36
        output_resolution = 80
        mod_amp = 10.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedDoubleRoof sensor
        double_roof = ModulatedDoubleRoof(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        double_roof.inputs['in_ef'].set(ef)

        # Setup and run
        double_roof.setup()
        double_roof.check_ready(t)
        double_roof.trigger()
        double_roof.post_trigger()

        # Get output intensity
        intensity = double_roof.outputs['out_i']

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.imshow(intensity.i)
            plt.title("Output Intensity")
            plt.colorbar()
            plt.show()

        # Test 1: Check that both roofs exist
        self.assertTrue(hasattr(double_roof, 'roof1_tlt'), "roof1_tlt should be created")
        self.assertTrue(hasattr(double_roof, 'roof2_tlt'), "roof2_tlt should be created")

        # Test 2: Check roof dimensions are equal
        self.assertEqual(double_roof.roof1_tlt.shape, double_roof.roof2_tlt.shape,
                        "Both roofs should have the same dimensions")

        # Test 3: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match expected {expected_shape}")

        # Test 4: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

        # Test 5: Check flux conservation (total intensity should match input)
        total_flux = xp.sum(intensity.i)
        expected_flux = ref_S0 * ef.masked_area()
        np.testing.assert_allclose(cpuArray(total_flux), cpuArray(expected_flux),
                                 rtol=0.1, atol=1e-6,
                                 err_msg="Total flux is not conserved")

        # Test 6: Check that we have 4 sub-pupils (basic structure test)
        # The intensity should have 4 distinct bright regions
        max_intensity = xp.max(intensity.i)
        threshold = max_intensity * 0.1  # 10% of max intensity
        bright_pixels = float(xp.sum(intensity.i > threshold))

        # Should have a reasonable number of bright pixels for 4 sub-pupils
        min_expected_pixels = 4 * (pup_diam // 4) ** 2  # Very rough estimate
        self.assertGreater(bright_pixels, min_expected_pixels,
                          "Not enough bright pixels for 4 sub-pupils")

        # Test 7:  Test quadrant structure
        h, w = intensity.i.shape
        mid_h, mid_w = h // 2, w // 2

        # Extract quadrants
        q1 = intensity.i[:mid_h, :mid_w]       # Top-left
        q2 = intensity.i[:mid_h, mid_w:]       # Top-right
        q3 = intensity.i[mid_h:, :mid_w]       # Bottom-left
        q4 = intensity.i[mid_h:, mid_w:]       # Bottom-right

        # Each quadrant should have some significant intensity
        total_intensity = cpuArray(xp.sum(intensity.i))
        q1_intensity = cpuArray(xp.sum(q1))
        q2_intensity = cpuArray(xp.sum(q2))
        q3_intensity = cpuArray(xp.sum(q3))
        q4_intensity = cpuArray(xp.sum(q4))

        # Each quadrant should have at least 5% of total intensity
        min_quadrant_fraction = 0.05
        self.assertGreater(q1_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 1 has insufficient intensity")
        self.assertGreater(q2_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 2 has insufficient intensity")
        self.assertGreater(q3_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 3 has insufficient intensity") 
        self.assertGreater(q4_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 4 has insufficient intensity")

        print(f"Test passed: output shape = {intensity.i.shape}, "
              f"total flux = {cpuArray(total_flux):.1f}, "
              f"bright pixels = {cpuArray(bright_pixels)}")

        print(f"Quadrant intensities: Q1={cpuArray(q1_intensity):.1f}, "
              f"Q2={cpuArray(q2_intensity):.1f}, Q3={cpuArray(q3_intensity):.1f}, "
              f"Q4={cpuArray(q4_intensity):.1f}") 