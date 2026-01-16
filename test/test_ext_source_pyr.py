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
            sampling_lambda_over_d=np.pi/8,
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
        for batch_size in [50, 100, 500]:
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
                err_msg=f"Batch size independence failed for batch_size={[100, 500, 2000][i]}")

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
        mod_amp = 2.0

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
