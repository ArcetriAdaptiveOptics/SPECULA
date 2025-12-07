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

        loD = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)  # in arcsec

        # Create extended source
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            # diamter of the ring in arcsec to get a ring with radius mod_amp
            size_obj=mod_amp * 4 * loD,
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

        loD = (wavelength_nm * 1e-9) / (pixel_pupil * pixel_pitch) * (206265)  # in arcsec

        # Create extended source
        src = ExtendedSource(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            source_type='TOPHAT',
            # diamter of the ring in arcsec to get a ring with radius mod_amp
            size_obj=mod_amp * 4 * loD,
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
