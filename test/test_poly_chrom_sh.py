import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.poly_chrom_sh import PolyChromSH
from test.specula_testlib import cpu_and_gpu

import numpy as np

class TestPolyChromSH(unittest.TestCase):

    @cpu_and_gpu
    def test_polychrom_sh_basic(self, target_device_idx, xp):
        """Test basic functionality of polychromatic SH"""

        t = 1
        ref_S0 = 100

        # Test parameters - use single subaperture for simplicity
        wavelengths = [500, 600, 700]
        flux_factors = [0.8, 1.0, 0.6]
        xy_tilts = [[0, 0], [0.05, 0.02], [-0.03, 0.08]]

        poly_sh = PolyChromSH(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            xy_tilts_in_arcsec=xy_tilts,
            subap_wanted_fov=2,
            sensor_pxscale=0.1,
            subap_on_diameter=1,  # Single subaperture
            subap_npx=20,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_sh.inputs['in_ef'].set(ef)
        poly_sh.setup()
        poly_sh.check_ready(t)
        poly_sh.trigger()
        poly_sh.post_trigger()

        # 1. Verify all intensities have the same dimensions
        output_intensity = poly_sh.outputs['out_i']
        expected_size = 1 * 20  # subap_on_diameter * subap_npx

        self.assertEqual(output_intensity.i.shape[0], expected_size)
        self.assertEqual(output_intensity.i.shape[1], expected_size)

        # Verify individual SH instances have same output size
        for i, sh in enumerate(poly_sh._wfs_instances):
            sh_output = sh.outputs['out_i']
            self.assertEqual(sh_output.i.shape, output_intensity.i.shape)

    @cpu_and_gpu
    def test_flux_factor_ef_consistency(self, target_device_idx, xp):
        """Test that intensity scales with flux_factor and electric fields are equal"""

        t = 1
        wavelengths = [500, 600, 700]
        flux_factors = [0.5, 1.0, 2.0]
        xy_tilts = [[0, 0], [0, 0], [0, 0]]

        poly_sh = PolyChromSH(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            xy_tilts_in_arcsec=xy_tilts,
            subap_wanted_fov=2,
            sensor_pxscale=0.1,
            subap_on_diameter=1,
            subap_npx=20,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_sh.inputs['in_ef'].set(ef)
        poly_sh.setup()
        poly_sh.check_ready(t)
        poly_sh.trigger()
        poly_sh.post_trigger()

        # Get individual wavelength contributions
        contrib_500 = poly_sh.get_wavelength_contribution(0)
        contrib_600 = poly_sh.get_wavelength_contribution(1)
        contrib_700 = poly_sh.get_wavelength_contribution(2)

        # Total intensity for each wavelength
        total_500 = xp.sum(contrib_500)
        total_600 = xp.sum(contrib_600)
        total_700 = xp.sum(contrib_700)

        # Check ratios match flux factor ratios
        # After normalization, contributions should be proportional to flux_factors
        normalized_flux = np.array(flux_factors) / np.sum(flux_factors)

        total_intensity = total_500 + total_600 + total_700

        expected_500 = total_intensity * normalized_flux[0]
        expected_600 = total_intensity * normalized_flux[1]
        expected_700 = total_intensity * normalized_flux[2]

        np.testing.assert_allclose(cpuArray(total_500), cpuArray(expected_500), rtol=1e-10)
        np.testing.assert_allclose(cpuArray(total_600), cpuArray(expected_600), rtol=1e-10)
        np.testing.assert_allclose(cpuArray(total_700), cpuArray(expected_700), rtol=1e-10)

        # Since no tilts are applied and no _has_tilts, all WFS should receive the same EF
        ref_ef = poly_sh._wfs_instances[0].local_inputs['in_ef']

        for i in range(1, len(poly_sh._wfs_instances)):
            test_ef = poly_sh._wfs_instances[i].local_inputs['in_ef']

            # phaseInNm from ef should match
            self.assertTrue(np.array_equal(cpuArray(ref_ef.phaseInNm), cpuArray(test_ef.phaseInNm)))

    @cpu_and_gpu
    def test_spot_shift_with_tilt(self, target_device_idx, xp):
        """Test that spots shift correctly with applied tilts - similar to SH pixel scale test"""

        t = 1
        pxscale_arcsec = 0.01
        pixel_pupil = 120
        pixel_pitch = 0.01
        sh_npix = 100

        # Create two identical wavelengths, one with tilt, one without
        wavelengths = [500, 500]
        flux_factors = [1.0, 1.0]
        xy_tilts = [[0, 0], [10.0*pxscale_arcsec, 0]]  # 10 pixels tilt in x direction

        poly_sh = PolyChromSH(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            xy_tilts_in_arcsec=xy_tilts,
            subap_wanted_fov=sh_npix * pxscale_arcsec,
            sensor_pxscale=pxscale_arcsec,
            subap_on_diameter=1,
            subap_npx=sh_npix,
            target_device_idx=target_device_idx
        )

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_sh.inputs['in_ef'].set(ef)
        poly_sh.setup()
        poly_sh.check_ready(t)
        poly_sh.trigger()
        poly_sh.post_trigger()

        # Get individual contributions
        flat_contrib = poly_sh.get_wavelength_contribution(0)
        tilted_contrib = poly_sh.get_wavelength_contribution(1)

        # Manually shift the flat contribution by 10 pixels to compare
        flat_shifted = xp.roll(flat_contrib, 10, axis=1)

        diff = cpuArray(tilted_contrib) - cpuArray(flat_shifted)

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.title("Flat Contribution")
            plt.imshow(cpuArray(flat_contrib), cmap='gray')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.title("Tilted Contribution")
            plt.imshow(cpuArray(tilted_contrib), cmap='gray')
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.title("Difference")
            plt.imshow(diff, cmap='gray')
            plt.colorbar()
            plt.show()

        # rms of difference divided by rms of flat contribution
        rms_diff = np.sqrt(np.mean(diff**2))
        rms_flat = np.sqrt(np.mean(cpuArray(flat_contrib)**2))
        print(f"RMS difference / RMS flat: {rms_diff / rms_flat if rms_flat != 0 else 0}")

        # this must be less than 2%
        np.testing.assert_allclose(cpuArray(rms_diff / rms_flat), 0, atol=2e-2)

    @cpu_and_gpu
    def test_wavelength_scaling(self, target_device_idx, xp):
        """Test that spot size scales with wavelength"""

        t = 1
        wavelengths = [500, 1000]  # 2x wavelength difference
        flux_factors = [1.0, 1.0]
        xy_tilts = [[0, 0], [0, 0]]

        poly_sh = PolyChromSH(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            xy_tilts_in_arcsec=xy_tilts,
            subap_wanted_fov=1.0,
            sensor_pxscale=0.01,
            subap_on_diameter=1,
            subap_npx=100,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(100, 100, 0.01, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_sh.inputs['in_ef'].set(ef)
        poly_sh.setup()
        poly_sh.check_ready(t)
        poly_sh.trigger()
        poly_sh.post_trigger()

        # Get individual wavelength contributions
        contrib_500 = poly_sh.get_wavelength_contribution(0)
        contrib_1000 = poly_sh.get_wavelength_contribution(1)

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title("500 nm Contribution")
            plt.imshow(cpuArray(contrib_500), cmap='gray')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.title("1000 nm Contribution")
            plt.imshow(cpuArray(contrib_1000), cmap='gray')
            plt.colorbar()
            plt.show()

        # Compare maximum value they should be equal to square of wavelength ratio
        max_ratio = xp.max(contrib_500) / xp.max(contrib_1000)
        expected_ratio = 4
        np.testing.assert_allclose(cpuArray(max_ratio), expected_ratio, rtol=0.1)

    @cpu_and_gpu
    def test_total_flux_conservation(self, target_device_idx, xp):
        """Test that total flux is conserved"""

        t = 1
        ref_S0 = 100
        wavelengths = [500, 600, 700]
        flux_factors = [0.3, 0.4, 0.3]
        xy_tilts = [[0, 0], [0.01, -0.02], [0.02, 0.01]]

        poly_sh = PolyChromSH(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            xy_tilts_in_arcsec=xy_tilts,
            subap_wanted_fov=3,
            sensor_pxscale=0.5,
            subap_on_diameter=1,
            subap_npx=20,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_sh.inputs['in_ef'].set(ef)
        poly_sh.setup()
        poly_sh.check_ready(t)
        poly_sh.trigger()
        poly_sh.post_trigger()

        total_output_flux = xp.sum(poly_sh.outputs['out_i'].i)
        expected_flux = ref_S0 * ef.masked_area()

        np.testing.assert_allclose(
            cpuArray(total_output_flux),
            cpuArray(expected_flux),
            rtol=1e-10
        )