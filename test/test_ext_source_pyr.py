import specula
specula.init(0)

import unittest
import numpy as np

from specula import cpuArray
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.extended_source import ExtendedSource
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.processing_objects.ext_source_pyramid import ExtSourcePyramid
from test.specula_testlib import cpu_and_gpu

class TestExtSourcePyramidComparison(unittest.TestCase):

    @cpu_and_gpu
    def test_compare_modulated_vs_extsource_pyramid_small_ext(self, target_device_idx, xp):
        # Simulation parameters
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 1.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)  # in arcsec

        # Create extended source
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            # diamter of the ring in arcsec to get a ring with radius mod_amp
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=1,             # one ring
            # choose the value to have the same number of points as the modulation
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Flat wavefront
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        # Pyramid 1: ModulatedPyramid
        pyr1 = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            target_device_idx=target_device_idx
        )
        pyr1.inputs['in_ef'].set(ef)
        pyr1.setup()
        pyr1.check_ready(1)
        pyr1.trigger()
        pyr1.post_trigger()
        out1 = cpuArray(pyr1.outputs['out_i'].i)

        # Pyramid 2: ExtSourcePyramid
        pyr2 = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr2.inputs['in_ef'].set(ef)
        pyr2.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr2.setup()
        pyr2.check_ready(1)
        pyr2.trigger()
        pyr2.post_trigger()
        out2 = cpuArray(pyr2.outputs['out_i'].i)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(out1, cmap='viridis')
            plt.colorbar()
            plt.title("ModulatedPyramid Output")
            plt.subplot(1, 3, 2)
            plt.imshow(out2, cmap='viridis')
            plt.colorbar()
            plt.title("ExtSourcePyramid Output")
            plt.subplot(1, 3, 3)
            plt.imshow(out1 - out2, cmap='viridis')
            plt.colorbar()
            plt.title("Difference (small, flat)")
            plt.show()

        # Compare outputs
        np.testing.assert_allclose(out1, out2, rtol=1e-3, atol=1e-3,
            err_msg="ExtSourcePyramid and ModulatedPyramid outputs differ!")

        # non flat wavefront
        ef.phaseInNm = 100 * np.random.randn(pixel_pupil, pixel_pupil)
        ef.generation_time += 1

        pyr1.check_ready(1)
        pyr1.trigger()
        pyr1.post_trigger()

        pyr2.check_ready(1)
        pyr2.trigger()
        pyr2.post_trigger()

        if plot_debug: # pragma: no cover
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(out1, cmap='viridis')
            plt.colorbar()
            plt.title("ModulatedPyramid Output")
            plt.subplot(1, 3, 2)
            plt.imshow(out2, cmap='viridis')
            plt.colorbar()
            plt.title("ExtSourcePyramid Output")
            plt.subplot(1, 3, 3)
            plt.imshow(out1 - out2, cmap='viridis')
            plt.colorbar()
            plt.title("Difference (small, non-flat)")
            plt.show()

        # Compare outputs
        np.testing.assert_allclose(out1, out2, rtol=1e-3, atol=1e-3,
            err_msg="ExtSourcePyramid and ModulatedPyramid outputs differ!")

        print("Comparison test passed: outputs are equal.")

    @cpu_and_gpu
    def test_compare_modulated_vs_extsource_pyramid_big_ext(self, target_device_idx, xp):
        # Simulation parameters
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 10.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)  # in arcsec

        # Create extended source
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            # diamter of the ring in arcsec to get a ring with radius mod_amp
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=1,             # one ring
            # choose the value to have the same number of points as the modulation
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Flat wavefront
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        # Pyramid 1: ModulatedPyramid
        pyr1 = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            target_device_idx=target_device_idx
        )
        pyr1.inputs['in_ef'].set(ef)
        pyr1.setup()
        pyr1.check_ready(1)
        pyr1.trigger()
        pyr1.post_trigger()
        out1 = cpuArray(pyr1.outputs['out_i'].i)

        # Pyramid 2: ExtSourcePyramid
        pyr2 = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            cuda_stream_enable=False,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr2.inputs['in_ef'].set(ef)
        pyr2.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr2.setup()
        pyr2.check_ready(1)
        pyr2.trigger()
        pyr2.post_trigger()
        out2 = cpuArray(pyr2.outputs['out_i'].i)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(out1, cmap='viridis')
            plt.colorbar()
            plt.title("ModulatedPyramid Output")
            plt.subplot(1, 3, 2)
            plt.imshow(out2, cmap='viridis')
            plt.colorbar()
            plt.title("ExtSourcePyramid Output")
            plt.subplot(1, 3, 3)
            plt.imshow(out1 - out2, cmap='viridis')
            plt.colorbar()
            plt.title("Difference (big, flat)")
            plt.show()

        # Compare outputs
        np.testing.assert_allclose(out1, out2, rtol=1e-3, atol=1e-3,
            err_msg="ExtSourcePyramid and ModulatedPyramid outputs differ!")

        # non flat wavefront
        ef.phaseInNm = 100 * np.random.randn(pixel_pupil, pixel_pupil)
        ef.generation_time += 1

        pyr1.check_ready(1)
        pyr1.trigger()
        pyr1.post_trigger()

        pyr2.check_ready(1)
        pyr2.trigger()
        pyr2.post_trigger()

        if plot_debug: # pragma: no cover
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(out1, cmap='viridis')
            plt.colorbar()
            plt.title("ModulatedPyramid Output")
            plt.subplot(1, 3, 2)
            plt.imshow(out2, cmap='viridis')
            plt.colorbar()
            plt.title("ExtSourcePyramid Output")
            plt.subplot(1, 3, 3)
            plt.imshow(out1 - out2, cmap='viridis')
            plt.colorbar()
            plt.title("Difference (big, non-flat)")
            plt.show()

        # Compare outputs
        np.testing.assert_allclose(out1, out2, rtol=1e-4, atol=1e-4,
            err_msg="ExtSourcePyramid and ModulatedPyramid outputs differ!")

        print("Comparison test passed: outputs are equal.")


    @cpu_and_gpu
    def test_batch_size_independence(self, target_device_idx, xp):
        """Test that different batch sizes produce identical results"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)

        # Create extended source with many points
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=5,
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Wavefront with aberrations
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.phaseInNm = 50 * np.random.randn(pixel_pupil, pixel_pupil)
        ef.generation_time = 1

        # Test with different batch sizes
        outputs = []
        for batch_size in [64, 128, 256]:
            pyr = ExtSourcePyramid(
                simul_params=simul_params,
                wavelengthInNm=wavelength_nm,
                fov=fov,
                pup_diam=pup_diam,
                output_resolution=output_resolution,
                mod_amp=mod_amp,
                max_batch_size=batch_size,
                target_device_idx=target_device_idx
            )
            pyr.inputs['in_ef'].set(ef)
            pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
            pyr.setup()
            pyr.check_ready(1)
            pyr.trigger()
            pyr.post_trigger()
            outputs.append(cpuArray(pyr.outputs['out_i'].i))

        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_allclose(outputs[0], outputs[i], rtol=1e-10, atol=1e-10,
                err_msg=f"Batch size independence failed for batch_size={[64, 128, 256][i]}")

        print("Batch size independence test passed.")


    @cpu_and_gpu
    def test_flux_conservation(self, target_device_idx, xp):
        """Test that total flux is conserved through transmission"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 1.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)

        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=2,
            sampling_lambda_over_d=np.pi/6,
            target_device_idx=target_device_idx,
        )
        src.compute()

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        pyr = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr.inputs['in_ef'].set(ef)
        pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr.setup()
        pyr.check_ready(1)
        pyr.trigger()
        pyr.post_trigger()

        # Total flux in pyramid image (after normalization by factor)
        flux_pyr = float(np.sum(cpuArray(pyr.outputs['out_i'].i)))

        # Expected: normalized flux (=1 after factor normalization) * transmission
        phot = float(ef.S0 * ef.masked_area())
        transmission = cpuArray(pyr.transmission.value)
        if transmission.ndim > 0:
            transmission = transmission[0]
        expected_flux = phot * transmission

        np.testing.assert_allclose(flux_pyr, expected_flux, rtol=0.01,
            err_msg=f"Flux conservation failed! pyr={flux_pyr:.3e}, expected={expected_flux:.3e}")

        print(f"Flux conservation test passed: pyr={flux_pyr:.3e},"
              f" transmission={expected_flux:.3e}")


    @cpu_and_gpu
    def test_zero_flux_points_ignored(self, target_device_idx, xp):
        """Test that points with zero flux don't contribute"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 1.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create source and manually set some fluxes to zero
        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=1,
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Set half the points to zero flux
        coeff = src.outputs['coeff'].value.copy()
        coeff[::2, 3] = 0  # Set every other point's flux to zero
        src.outputs['coeff'].value[:] = coeff
        src.outputs['coeff'].generation_time = 2

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        pyr = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            cuda_stream_enable=False,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr.inputs['in_ef'].set(ef)
        pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr.setup()

        # Check that valid_idx only includes non-zero flux points
        n_nonzero = np.sum(coeff[:, 3] > 0)
        self.assertEqual(len(cpuArray(pyr.valid_idx)), n_nonzero,
            "valid_idx should only contain points with non-zero flux")

        print("Zero flux points test passed.")


    @cpu_and_gpu
    def test_flux_additivity(self, target_device_idx, xp):
        """Test that flux contributions are additive: full = half1 + half2"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)

        # Create source with multiple points
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=2,
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.phaseInNm = 50 * np.random.randn(pixel_pupil, pixel_pupil)
        ef.generation_time = 1

        # Case 1: Full source
        pyr_full = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr_full.inputs['in_ef'].set(ef)
        pyr_full.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr_full.setup()
        pyr_full.check_ready(1)
        pyr_full.trigger()
        pyr_full.post_trigger()
        out_full = cpuArray(pyr_full.outputs['out_i'].i)

        # Case 2: First half of points only
        coeff_original = src.outputs['coeff'].value.copy()
        coeff_half1 = src.outputs['coeff'].value.copy()
        n_points = coeff_half1.shape[0]
        coeff_half1[:n_points//2, 3] = coeff_original[:n_points//2, 3] # Restore first half
        coeff_half1[n_points//2:, 3] = coeff_original[n_points//2:, 3] * 1e-6  # Zero out second half
        src.outputs['coeff'].value[:] = coeff_half1
        src.outputs['coeff'].generation_time = 2

        pyr_half1 = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr_half1.inputs['in_ef'].set(ef)
        pyr_half1.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr_half1.setup()
        pyr_half1.check_ready(1)
        pyr_half1.trigger()
        pyr_half1.post_trigger()
        out_half1 = cpuArray(pyr_half1.outputs['out_i'].i)

        # Case 3: Second half of points only
        coeff_half2 = src.outputs['coeff'].value.copy()
        coeff_half2[:n_points//2, 3] = coeff_original[:n_points//2, 3] * 1e-6  # Zero out first half
        coeff_half2[n_points//2:, 3] = coeff_original[n_points//2:, 3]  # Restore second half
        src.outputs['coeff'].value[:] = coeff_half2
        src.outputs['coeff'].generation_time = 3

        pyr_half2 = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr_half2.inputs['in_ef'].set(ef)
        pyr_half2.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr_half2.setup()
        pyr_half2.check_ready(1)
        pyr_half2.trigger()
        pyr_half2.post_trigger()
        out_half2 = cpuArray(pyr_half2.outputs['out_i'].i)

        # Verify additivity: full = 0.5 * (half1 + half2)
        out_sum = 0.5 * (out_half1 + out_half2)
        np.testing.assert_allclose(out_full, out_sum, rtol=5e-2, atol=5e-4,
            err_msg="Flux additivity failed: full != 0.5 * (half1 + half2)")

        print("Flux additivity test passed: full = 0.5 * (half1 + half2)")


    @cpu_and_gpu
    def test_flux_threshold_filtering(self, target_device_idx, xp):
        """Test that max_flux_ratio_thr correctly filters low-flux points"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)

        # Create source with varying fluxes
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=3,
            sampling_lambda_over_d=np.pi/6,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Manually create flux distribution with known range
        coeff = src.outputs['coeff'].value.copy()
        n_points = coeff.shape[0]

        # Set fluxes: 10 strong points + 90 weak points
        coeff[:10, 3] = 1.0  # Strong flux
        coeff[10:, 3] = 1e-5  # Weak flux (below threshold)

        src.outputs['coeff'].value[:] = coeff
        src.outputs['coeff'].generation_time = 2

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        # Test 1: With threshold enabled (stream disabled)
        threshold = 1e-3
        pyr_filtered = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_flux_ratio_thr=threshold,
            cuda_stream_enable=False,  # Enable threshold
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr_filtered.inputs['in_ef'].set(ef)
        pyr_filtered.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr_filtered.setup()

        # Check that only strong points are in valid_idx
        # plus four pixels where the flux has been redistributed
        n_valid_filtered = len(cpuArray(pyr_filtered.valid_idx))
        self.assertEqual(n_valid_filtered, 14,
            f"Expected 14 valid points with threshold, got {n_valid_filtered}")

        # Verify that flux_factor_vector has zeros for filtered points
        ffv_filtered = cpuArray(pyr_filtered.flux_factor_vector)
        n_nonzero = np.sum(ffv_filtered > 0)
        self.assertEqual(n_nonzero, 14,
            f"Expected 14 non-zero flux values, got {n_nonzero}")

        # Test 2: Without threshold (stream enabled)
        pyr_all = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_flux_ratio_thr=threshold,
            cuda_stream_enable=True,  # Disable threshold
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr_all.inputs['in_ef'].set(ef)
        pyr_all.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr_all.setup()

        # Check that all points are processed
        # Four extra points where the flux has been redistributed
        n_valid_all = len(cpuArray(pyr_all.valid_idx))
        self.assertEqual(n_valid_all, n_points+4,
            f"Expected {n_points+4} valid points without threshold, got {n_valid_all}")

        # Test 3: Verify outputs are similar (weak flux has negligible effect)
        pyr_filtered.check_ready(1)
        pyr_filtered.trigger()
        pyr_filtered.post_trigger()
        out_filtered = cpuArray(pyr_filtered.outputs['out_i'].i)

        pyr_all.check_ready(1)
        pyr_all.trigger()
        pyr_all.post_trigger()
        out_all = cpuArray(pyr_all.outputs['out_i'].i)

        # Outputs should be very similar since weak flux contributes < 0.1%
        np.testing.assert_allclose(out_filtered, out_all, rtol=0.01, atol=1e-6,
            err_msg="Filtered and unfiltered outputs differ more than expected")

        print(f"Flux threshold test passed: {n_valid_filtered} points kept out of {n_points}")


    @cpu_and_gpu
    def test_threshold_effect_on_performance(self, target_device_idx, xp):
        """Test that threshold actually reduces computational load"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)

        # Create large extended source
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=10,  # Many points
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        # Create flux distribution with 80% weak points
        coeff = src.outputs['coeff'].value.copy()
        n_points = coeff.shape[0]
        n_strong = n_points // 5  # 20% strong

        coeff[:n_strong, 3] = 1.0
        coeff[n_strong:, 3] = 1e-6  # Very weak
        src.outputs['coeff'].value[:] = coeff
        src.outputs['coeff'].generation_time = 2

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        # Aggressive threshold should keep only ~20% of points
        pyr = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            max_flux_ratio_thr=1e-4,
            cuda_stream_enable=False,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr.inputs['in_ef'].set(ef)
        pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr.setup()

        n_valid = len(cpuArray(pyr.valid_idx))
        n_chunks = pyr._n_chunks

        # Verify significant reduction
        reduction_factor = n_points / n_valid
        self.assertGreater(reduction_factor, 3.0,
            f"Expected at least 3x reduction, got {reduction_factor:.1f}x")

        # Verify buffer size matches filtered points
        expected_chunks = (n_valid + pyr.max_batch_size - 1) // pyr.max_batch_size
        self.assertEqual(n_chunks, expected_chunks,
            f"Buffer chunks ({n_chunks}) doesn't match expected ({expected_chunks})")

        print(f"Threshold performance test passed: {n_points} -> {n_valid} points "
              f"({reduction_factor:.1f}x reduction, {n_chunks} chunks)")


    @cpu_and_gpu
    def test_face_centers_geometry(self, target_device_idx, xp):
        """Test that face center points are positioned at pyramid face centers"""
        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        batch_size = 64

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        l_o_d = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            size_obj=mod_amp * 4 * l_o_d,
            sampling_type='RINGS',
            n_rings=1,
            sampling_lambda_over_d=np.pi/4,
            target_device_idx=target_device_idx,
        )
        src.compute()

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1

        pyr = ExtSourcePyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            cuda_stream_enable=False,
            max_batch_size=batch_size,
            target_device_idx=target_device_idx
        )
        pyr.inputs['in_ef'].set(ef)
        pyr.inputs['ext_source_coeff'].set(src.outputs['coeff'])
        pyr.setup()

        # Get face center points
        face_angles_ttf, mean_radius = pyr._get_pyramid_face_angles_at_fov_radius()
        face_angles_ttf = cpuArray(face_angles_ttf)
        mean_radius = float(cpuArray(mean_radius))

        tip_vals = face_angles_ttf[:, 0]
        tilt_vals = face_angles_ttf[:, 1]
        focus_vals = face_angles_ttf[:, 2]

        # Test 1: Verify focus is zero
        np.testing.assert_allclose(focus_vals, 0, atol=1e-10,
            err_msg="Focus component should be zero for face center points")

        # Test 2: Verify all points are at the same radius
        radii = np.sqrt(tip_vals**2 + tilt_vals**2)
        np.testing.assert_allclose(radii, mean_radius, rtol=1e-10,
            err_msg=f"All points should be at radius {mean_radius}, got {radii}")

        # Test 3: Verify angles are at 0°, 90°, 180°, 270° (FACE CENTERS)
        angles_rad = np.arctan2(tilt_vals, tip_vals)
        angles_deg = np.rad2deg(angles_rad)
        angles_deg = np.mod(angles_deg, 360)

        expected_angles = np.array([0, 90, 180, 270])
        angles_deg_sorted = np.sort(angles_deg)

        np.testing.assert_allclose(angles_deg_sorted, expected_angles, atol=1e-6,
            err_msg=f"Expected face center angles {expected_angles}, got {angles_deg_sorted}")

        # Test 4: Verify radius matches FoV/2
        expected_radius = float((fov / 2.0) / (pyr.fov_res * pyr.fft_res))
        np.testing.assert_allclose(mean_radius, expected_radius, rtol=1e-10,
            err_msg=f"Radius should match FoV/2: expected {expected_radius}, got {mean_radius}")

        # Test 5: Verify points are at FACE CENTERS (perpendicular to axes)
        for i, (tx, ty) in enumerate(zip(tip_vals, tilt_vals)):
            # Each point should be aligned with either tip OR tilt axis
            ratio = min(abs(tx), abs(ty)) / max(abs(tx), abs(ty))
            self.assertLess(ratio, 0.01,
                f"Point {i} at angle {angles_deg[i]}° should be near an axis "
                f"(face center), not diagonal. Got ratio {ratio:.3f}")

        # Test 6: Verify points are in different quadrants
        quadrants = []
        for tx, ty in zip(tip_vals, tilt_vals):
            if abs(tx) > abs(ty):
                if tx > 0:
                    quadrants.append(1)  # +X axis (0°)
                else:
                    quadrants.append(3)  # -X axis (180°)
            else:
                if ty > 0:
                    quadrants.append(2)  # +Y axis (90°)
                else:
                    quadrants.append(4)  # -Y axis (270°)

        self.assertEqual(len(set(quadrants)), 4,
            "Face center points should be on 4 different axes")

        print(f"Face centers geometry test passed:")
        print(f"  - Radius: {mean_radius:.4f} lambda/D (FoV/2 = {fov/2:.2f} arcsec)")
        print(f"  - Angles: {angles_deg_sorted}° (face centers)")
        print(f"  - Points correctly positioned at pyramid FACE CENTERS")
