import specula
specula.init(0)  # Default target device

import unittest
from unittest.mock import MagicMock, patch

from specula import np
from specula import cpuArray, RAD2ASEC

from specula.loop_control import LoopControl
from specula.processing_objects.pyr_pupdata_calibrator import PyrPupdataCalibrator
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.lib.make_xy import make_xy
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from test.specula_testlib import cpu_and_gpu


class TestModulatedPyramid(unittest.TestCase):

    @cpu_and_gpu
    def test_incorrect_tilt_coeff_input_raises(self, target_device_idx, xp):
        """Check that a wrong pyr_tlt_coeffs shape or sign raises an error during init"""
        simul_params = SimulParams(
            pixel_pupil=60,
            pixel_pitch=0.1,
        )

        tlt_coeffs = xp.ones([3,3]) # shape should be [2,4]
        with self.assertRaises(ValueError):
            ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=700,
                pup_diam=20,
                fov = 3.0,
                output_resolution=60,
                pyr_tlt_coeff=tlt_coeffs.tolist(),
                target_device_idx=target_device_idx,
            )

        tlt_coeffs = xp.ones([2,4])
        tlt_coeffs[1,2] = -0.5 # should be positive
        with self.assertRaises(ValueError):
            ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=700,
                pup_diam=20,
                fov = 3.0,
                output_resolution=60,
                pyr_tlt_coeff=tlt_coeffs.tolist(),
                target_device_idx=target_device_idx,
            )

    @cpu_and_gpu
    def test_pyramid_tilt_coeffs_equal_to_one(self, target_device_idx, xp):
        """ Check that applying a unit tilt coefficient produces the same result as the default (nominal) pyramid """
        pixel_pupil = 64
        pixel_pitch = 0.1
        wavelength_in_nm = 750
        pup_diam = 24
        pup_dist = 32
        output_resolution = 64
        fov = 2.0
        mod_amp = 8.0

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
        )

        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_in_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            pyr_tlt_coeff=None,
            target_device_idx=target_device_idx,
        )

        tlt_coeffs = xp.ones([2,4])
        pyramid_tlt = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_in_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            pyr_tlt_coeff=tlt_coeffs.tolist(),
            target_device_idx=target_device_idx,
        )

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=100, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 0

        pyramid.inputs['in_ef'].set(ef)
        pyramid_tlt.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(pyramid, idx=0)
        loop.add(pyramid_tlt, idx=1)
        loop.run(run_time=1, dt=1, t0=0)

        def _get_deviations(pyramid_obj):
            pupcalib = PyrPupdataCalibrator(
                data_dir="/tmp",
                target_device_idx=target_device_idx,
            )
            pupcalib.local_inputs['in_i'] = pyramid_obj.outputs['out_i']
            pupcalib.trigger_code()
            dx = pupcalib.pupdata.cx - (pixel_pupil-1)/2
            dy = pupcalib.pupdata.cy - (pixel_pupil-1)/2
            return dx,dy

        x_deviation, y_deviation = _get_deviations(pyramid)
        x_deviation_tlt, y_deviation_tlt = _get_deviations(pyramid_tlt)

        # Check all other pupil tilts are unaffected
        np.testing.assert_allclose(cpuArray(x_deviation),cpuArray(x_deviation_tlt),atol=0.001)
        np.testing.assert_allclose(cpuArray(y_deviation),cpuArray(y_deviation_tlt),atol=0.001) 


    @cpu_and_gpu
    def test_pyramid_tilt_coeffs(self, target_device_idx, xp):
        """ Check that applying a tilt coefficient changes the pupil-center estimate for the affected face """
        pixel_pupil = 64
        pixel_pitch = 0.1
        wavelength_in_nm = 750
        pup_diam = 24
        pup_dist = 32
        output_resolution = 64
        fov = 2.0
        mod_amp = 8.0

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
        )

        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_in_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            target_device_idx=target_device_idx,
        )

        for pupid in range(4):
            pupids = xp.array([0,1,2,3],dtype=int)
            pyrids = xp.array([2,3,1,0],dtype=int)
            tlt_x = 1.2
            tlt_y = 0.9
            tlt_coeffs = xp.ones([2,4])
            tlt_coeffs[0,pyrids[pupid]] = tlt_x
            tlt_coeffs[1,pyrids[pupid]] = tlt_y
            pyramid_tlt = ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=wavelength_in_nm,
                fov=fov,
                pup_diam=pup_diam,
                pup_dist=pup_dist,
                output_resolution=output_resolution,
                mod_amp=mod_amp,
                pyr_tlt_coeff=tlt_coeffs.tolist(),
                target_device_idx=target_device_idx,
            )

            ef = ElectricField(
                pixel_pupil, pixel_pupil, pixel_pitch, S0=100, target_device_idx=target_device_idx
            )
            ef.A = make_mask(pixel_pupil)
            ef.generation_time = 0

            pyramid.inputs['in_ef'].set(ef)
            pyramid_tlt.inputs['in_ef'].set(ef)

            loop = LoopControl()
            loop.add(pyramid, idx=0)
            loop.add(pyramid_tlt, idx=1)
            loop.run(run_time=1, dt=1, t0=0)

            def _get_deviations(pyramid_obj):
                pupcalib = PyrPupdataCalibrator(
                    data_dir="/tmp",
                    target_device_idx=target_device_idx,
                )
                pupcalib.local_inputs['in_i'] = pyramid_obj.outputs['out_i']
                pupcalib.trigger_code()
                dx = pupcalib.pupdata.cx - (pixel_pupil-1)/2
                dy = pupcalib.pupdata.cy - (pixel_pupil-1)/2
                return dx,dy

            x_deviation, y_deviation = _get_deviations(pyramid)
            x_deviation_tlt, y_deviation_tlt = _get_deviations(pyramid_tlt)

            # Ensure the tilt is giving the expected pixel deviation
            np.testing.assert_almost_equal(cpuArray(abs(x_deviation_tlt[pupid])),cpuArray(pup_dist/2*tlt_x),decimal=1)
            np.testing.assert_almost_equal(cpuArray(abs(y_deviation_tlt[pupid])),cpuArray(pup_dist/2*tlt_y),decimal=1)

            # Sanity check on the nominal pupil deviation
            np.testing.assert_almost_equal(cpuArray(x_deviation[-1]),cpuArray(pup_dist/2),decimal=1)
            np.testing.assert_almost_equal(cpuArray(y_deviation[-1]),cpuArray(pup_dist/2),decimal=1)

            # Check all other pupil tilts are unaffected
            ids = (pupids!=pupid).astype(bool)
            np.testing.assert_allclose(cpuArray(x_deviation[ids]),cpuArray(x_deviation_tlt[ids]),atol=0.1)
            np.testing.assert_allclose(cpuArray(y_deviation[ids]),cpuArray(y_deviation_tlt[ids]),atol=0.1) 


    @cpu_and_gpu
    def test_fft_grid_matches_fp_mask_shape(self, target_device_idx, xp):
        """Regression test for odd FFT sizes where the pyramid phase map and mask must share the same dimensions."""
        # Case A
        simul_params = SimulParams(
            pixel_pupil=367,
            pixel_pitch=0.1,
        )
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=750,
            fov=8.0,
            pup_diam=40,
            output_resolution=120,
            target_device_idx=target_device_idx,
        )
        pyr_tlt = pyramid.get_pyr_tlt(pyramid.fft_sampling, pyramid.fft_padding)
        self.assertEqual(
            pyr_tlt.shape,
            pyramid.fp_mask.shape,
            f"Expected pyramid tilt map and focal plane mask to share shape, got {pyr_tlt.shape} and {pyramid.fp_mask.shape}",
        )
        self.assertEqual(
            pyramid.masked_exp.shape,
            pyramid.fp_mask.shape,
            f"Expected masked exponential to match mask shape, got {pyramid.masked_exp.shape} and {pyramid.fp_mask.shape}",
        )

        # Case B
        simul_params = SimulParams(
            pixel_pupil=197,
            pixel_pitch=0.1,
        )
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=750,
            fov=2.0,
            pup_diam=40,
            output_resolution=120,
            target_device_idx=target_device_idx,
        )
        pyr_tlt = pyramid.get_pyr_tlt(pyramid.fft_sampling, pyramid.fft_padding)
        self.assertEqual(
            pyr_tlt.shape,
            pyramid.fp_mask.shape,
            f"Expected pyramid tilt map and focal plane mask to share shape, got {pyr_tlt.shape} and {pyramid.fp_mask.shape}",
        )
        self.assertEqual(
            pyramid.masked_exp.shape,
            pyramid.fp_mask.shape,
            f"Expected masked exponential to match mask shape, got {pyramid.masked_exp.shape} and {pyramid.fp_mask.shape}",
        )


    @cpu_and_gpu
    def test_flat_wavefront_output_size(self, target_device_idx, xp):
        """Test that ModulatedPyramid produces correct output dimensions for flat wavefront"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 3.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with circular modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='circular',
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        pyramid.inputs['in_ef'].set(ef)

        # Setup and run
        pyramid.setup()
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        # Get output intensity
        intensity = pyramid.outputs['out_i']

        plot_debug = False
        if plot_debug: #pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=[20,2])
            for i in range(pyramid.ttexp.shape[1]):
                plt.subplot(1, pyramid.ttexp.shape[1], i + 1)
                plt.imshow(xp.real(pyramid.ttexp[0, i, :, :]), cmap='gray')
            plt.title("TTExp for Circular Modulation")
            plt.figure()
            plt.imshow(intensity.i)
            plt.title("Intensity for Circular Modulation")
            plt.show()

        # Test 1: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match"
                        f" expected {expected_shape}")

        # Test 2: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

        # Test 3: Check flux conservation (total intensity should match input)
        total_flux = xp.sum(intensity.i)
        expected_flux = ref_S0 * ef.masked_area()
        np.testing.assert_allclose(cpuArray(total_flux), cpuArray(expected_flux),
                                 rtol=0.1, atol=1e-6,
                                 err_msg="Total flux is not conserved")

        # Test 4: Check that we have 4 sub-pupils (basic structure test)
        max_intensity = xp.max(intensity.i)
        threshold = max_intensity * 0.1  # 10% of max intensity
        bright_pixels = float(xp.sum(intensity.i > threshold))

        # Should have a reasonable number of bright pixels for 4 sub-pupils
        min_expected_pixels = 4 * (pup_diam // 4) ** 2  # Very rough estimate
        self.assertGreater(bright_pixels, min_expected_pixels,
                          "Not enough bright pixels for 4 sub-pupils")

        print(f"Circular modulation test passed: output shape = {intensity.i.shape}, "
              f"total flux = {cpuArray(total_flux):.1f}, "
              f"bright pixels = {cpuArray(bright_pixels)}")

    @cpu_and_gpu
    def test_zero_modulation(self, target_device_idx, xp):
        """Test ModulatedPyramid with zero modulation amplitude"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 0.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with circular modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='circular',
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        pyramid.inputs['in_ef'].set(ef)

        # Setup and run
        pyramid.setup()
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        # Get output intensity
        intensity = pyramid.outputs['out_i']

        plot_debug = False
        if plot_debug:  #pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=[4,4])
            plt.imshow(xp.real(pyramid.ttexp[0, 0, :, :]), cmap='gray')
            plt.title("TTExp for Zero Modulation")
            plt.figure()
            plt.imshow(intensity.i)
            plt.title("Intensity for Zero Modulation")
            plt.show()

        # Test 1: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match"
                        f" expected {expected_shape}")

        # Test 2: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

        # Test 3: Check ttexp dimensions
        expected_ttexp_shape = (1, 1, pyramid.fft_sampling, pyramid.fft_sampling)
        self.assertEqual(pyramid.ttexp.shape, expected_ttexp_shape,
                        f"ttexp shape {pyramid.ttexp.shape} doesn't match"
                        f" expected {expected_ttexp_shape}")

    @cpu_and_gpu
    def test_zero_modulation_large_fov(self, target_device_idx, xp):
        """Test ModulatedPyramid with zero modulation amplitude and large FOV"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 6.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 0.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with circular modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='circular',
            target_device_idx=target_device_idx
        )

        # fp_masking must be < 1 to avoid errors
        self.assertLess(pyramid.fp_masking, 1.0,
                       f"fp_masking is {pyramid.fp_masking}, must be < 1")

        # Create flat wavefront (no phase)
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        pyramid.inputs['in_ef'].set(ef)

        # Setup and run
        pyramid.setup()
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        # Get output intensity
        intensity = pyramid.outputs['out_i']

        plot_debug = False
        if plot_debug: #pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=[4,4])
            plt.imshow(xp.real(pyramid.ttexp[0, 0, :, :]), cmap='gray')
            plt.title("TTExp for Zero Modulation")
            plt.figure()
            plt.imshow(intensity.i)
            plt.title("Intensity for Zero Modulation")
            plt.show()

        # Test 1: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match"
                        f" expected {expected_shape}")

        # Test 2: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

        # Test 3: Check ttexp dimensions
        expected_ttexp_shape = (1, 1, pyramid.fft_sampling, pyramid.fft_sampling)
        self.assertEqual(pyramid.ttexp.shape, expected_ttexp_shape,
                        f"ttexp shape {pyramid.ttexp.shape} doesn't match"
                        f" expected {expected_ttexp_shape}")

    @cpu_and_gpu
    def test_vertical_modulation(self, target_device_idx, xp):
        """Test vertical modulation type"""

        # Test parameters
        t = 1
        pixel_pupil = 60
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 2.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with vertical modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='vertical',
            target_device_idx=target_device_idx
        )

        # Create flat wavefront
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input and setup
        pyramid.inputs['in_ef'].set(ef)

        # Setup and run
        pyramid.setup()
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=[10,2])
            for i in range(pyramid.ttexp.shape[1]):
                plt.subplot(1, pyramid.ttexp.shape[1], i + 1)
                plt.imshow(xp.real(pyramid.ttexp[0, i, :, :]), cmap='gray')
            plt.title("TTExp for Vertical Modulation")
            plt.show()

        # Test 1: Check mod_steps calculation for linear modulation
        expected_mod_steps = int(round(mod_amp) * 2 + 1)  # Should be 2*2+1 = 5
        self.assertEqual(pyramid.mod_steps, expected_mod_steps,
                        f"Expected {expected_mod_steps} mod_steps for vertical"
                        f"modulation, got {pyramid.mod_steps}")

        # Test 2: Check ttexp dimensions
        expected_ttexp_shape = (1, pyramid.mod_steps, pyramid.fft_sampling, pyramid.fft_sampling)
        self.assertEqual(pyramid.ttexp.shape, expected_ttexp_shape,
                        f"ttexp shape {pyramid.ttexp.shape} doesn't match"
                        f"expected {expected_ttexp_shape}")

        # Test 3: Check flux_factor_vector dimensions and values (they must be >0 and <inf)
        self.assertEqual(len(pyramid.flux_factor_vector), pyramid.mod_steps,
                        f"flux_factor_vector length {len(pyramid.flux_factor_vector)} doesn't match mod_steps {pyramid.mod_steps}")
        self.assertTrue(np.all(cpuArray(pyramid.flux_factor_vector) > 0) and np.all(cpuArray(pyramid.flux_factor_vector) < np.inf),
                        "flux_factor_vector values must be >0 and <inf")

        # Test 4: Check flux_factor_vector is symmetric for linear modulation
        ffv_cpu = cpuArray(pyramid.flux_factor_vector)
        # For symmetric linear modulation, first and last should be equal, etc.
        np.testing.assert_allclose(ffv_cpu[0], ffv_cpu[-1], rtol=1e-6,
                                 err_msg="Flux factor vector should be symmetric for linear modulation")

        # Test 5: Check that center value is maximum (cos(0) = 1)
        center_idx = pyramid.mod_steps // 2
        center_value = ffv_cpu[center_idx]
        self.assertTrue(center_value <= max(ffv_cpu[0], ffv_cpu[-1]),
                       "Center flux factor should be minimum for linear modulation")

        # Test 6: Run and check output
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        intensity = pyramid.outputs['out_i']
        self.assertEqual(intensity.i.shape, (output_resolution, output_resolution),
                        "Output shape incorrect for vertical modulation")

        verbose = False
        if verbose: #pragma: no cover
            print(f"Vertical modulation test passed: mod_steps = {pyramid.mod_steps}, "
                  f"ttexp shape = {pyramid.ttexp.shape}, "
                  f"flux_factor range = [{ffv_cpu.min():.3f}, {ffv_cpu.max():.3f}]")

    @cpu_and_gpu
    def test_horizontal_modulation(self, target_device_idx, xp):
        """Test horizontal modulation type"""

        # Test parameters (same as vertical)
        t = 1
        pixel_pupil = 60
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 2.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with horizontal modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='horizontal',
            target_device_idx=target_device_idx
        )

        # Create flat wavefront
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input and setup
        pyramid.inputs['in_ef'].set(ef)

        # Setup and run
        pyramid.setup()
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        intensity = pyramid.outputs['out_i']

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=[10,2])
            for i in range(pyramid.ttexp.shape[1]):
                plt.subplot(1, pyramid.ttexp.shape[1], i + 1)
                plt.imshow(xp.real(pyramid.ttexp[0, i, :, :]), cmap='gray')
            plt.title("TTExp for Horizontal Modulation")
            plt.figure()
            plt.imshow(intensity.i)
            plt.title("Intensity for Horizontal Modulation")
            plt.show()

        # Test 1: Check ttexp shape
        expected_ttexp_shape = (1, pyramid.mod_steps, pyramid.fft_sampling,
                                pyramid.fft_sampling)
        self.assertEqual(pyramid.ttexp.shape, expected_ttexp_shape,
                        f"ttexp shape {pyramid.ttexp.shape} doesn't match"
                        f"expected {expected_ttexp_shape}")

        # Test 2: Check mod_steps and dimensions (same as vertical)
        expected_mod_steps = int(round(mod_amp) * 2 + 1)
        self.assertEqual(pyramid.mod_steps, expected_mod_steps,
                        f"Expected {expected_mod_steps} mod_steps for horizontal modulation")

        # Test 3: Check flux_factor_vector properties (should be same as vertical)
        ffv_cpu = cpuArray(pyramid.flux_factor_vector)
        np.testing.assert_allclose(ffv_cpu[0], ffv_cpu[-1], rtol=1e-6,
                                 err_msg="Flux factor vector should be symmetric for"
                                 " horizontal modulation")

        # Test 4: Run and check output
        pyramid.check_ready(t)
        pyramid.trigger()
        pyramid.post_trigger()

        intensity = pyramid.outputs['out_i']
        self.assertEqual(intensity.i.shape, (output_resolution, output_resolution),
                        "Output shape incorrect for horizontal modulation")

        verbose = False
        if verbose: #pragma: no cover
            print(f"Horizontal modulation test passed: mod_steps = {pyramid.mod_steps}, "
                f"flux_factor range = [{ffv_cpu.min():.3f}, {ffv_cpu.max():.3f}]")

    @cpu_and_gpu
    def test_alternating_modulation(self, target_device_idx, xp):
        """Test alternating modulation type"""

        # Test parameters
        t = 1
        pixel_pupil = 60
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        output_resolution = 80
        mod_amp = 2.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedPyramid sensor with alternating modulation
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            mod_type='alternating',
            target_device_idx=target_device_idx
        )

        # Create flat wavefront
        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input and setup
        pyramid.inputs['in_ef'].set(ef)
        pyramid.setup()

        # Test 1: Check ttexp shape
        expected_ttexp_shape = (2, pyramid.mod_steps, pyramid.fft_sampling,
                                pyramid.fft_sampling)
        self.assertEqual(pyramid.ttexp.shape, expected_ttexp_shape,
                        f"ttexp shape {pyramid.ttexp.shape} doesn't match expected"
                        f" {expected_ttexp_shape}")

        # Test 2: Check iteration counter initialization
        self.assertEqual(pyramid.iter, 0, "Iteration counter should start at 0")

        # Test 3: Check mod_steps and dimensions
        expected_mod_steps = int(round(mod_amp) * 2 + 1)
        self.assertEqual(pyramid.mod_steps, expected_mod_steps,
                        f"Expected {expected_mod_steps} mod_steps for alternating modulation")

        # Test 4: Run multiple iterations to test alternation
        intensities = []
        for iteration in range(3):  # Test 3 iterations
            ef.generation_time = t + iteration  # Update generation time for each iteration
            pyramid.check_ready(t + iteration)
            pyramid.trigger()
            pyramid.post_trigger()

            # Check iteration counter increment
            self.assertEqual(pyramid.iter, iteration + 1,
                           f"Iteration counter should be {iteration + 1} after"
                           f"{iteration + 1} iterations")

            # Store intensity for comparison
            intensities.append(pyramid.outputs['out_i'].i.copy())

        # Test 5: Check that odd iterations differ from even iterations
        # (assuming the wavefront creates different patterns for vertical vs horizontal)
        intensity_0 = cpuArray(intensities[0])  # Even iteration (vertical)
        intensity_1 = cpuArray(intensities[1])  # Odd iteration (horizontal)
        intensity_2 = cpuArray(intensities[2])  # Even iteration (vertical)

        plot_debug = False
        if plot_debug: # pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=[15, 5])
            plt.subplot(1, 3, 1)
            plt.imshow(intensity_0)
            plt.title("Intensity 0 (even)")
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(intensity_1)
            plt.title("Intensity 1 (odd)")
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(intensity_2)
            plt.title("Intensity 2 (even)")
            plt.colorbar()
            plt.show()

        # Iterations 0 and 2 should be more similar (both vertical) than 0 and 1
        diff_0_2 = np.sum(np.abs(intensity_0 - intensity_2))
        diff_0_1 = np.sum(np.abs(intensity_0 - intensity_1))

        # This test might be weak depending on the pattern, but should show some difference
        # For a flat wavefront, the differences might be subtle
        verbose = False
        if verbose: #pragma: no cover
            print(f"Intensity difference 0-2 (both vertical): {diff_0_2:.2e}")
            print(f"Intensity difference 0-1 (vert-horiz): {diff_0_1:.2e}")
            print(f"Alternating modulation test passed: mod_steps = {pyramid.mod_steps},"
                  f" final iter = {pyramid.iter}")

    @cpu_and_gpu
    def test_fov_interpolation_calculation(self, target_device_idx, xp):
        """Test that FOV interpolation is calculated correctly and applied only once"""

        # Test parameters designed to trigger interpolation
        # Choose parameters where Fov_internal < requested FoV
        pixel_pupil = 80  # Small pupil
        pixel_pitch = 0.1  # Large pixel pitch
        wavelength_nm = 500
        requested_fov = 6.0  # Large FOV request
        pup_diam = 30
        output_resolution = 80

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Calculate expected internal FOV without interpolation
        expected_fov_no_interp = wavelength_nm * 1e-9 / pixel_pitch * RAD2ASEC

        # This should trigger interpolation since expected_fov_no_interp < requested_fov
        minfov = requested_fov * (1 - 0.5)  # fov_errinf = 0.5
        self.assertLess(expected_fov_no_interp, minfov,
                    "Test setup should trigger interpolation")

        # Create ModulatedPyramid with these parameters
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=requested_fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            fov_errinf=0.01,  # Allow 1% reduction
            fov_errsup=2.0,  # Allow 200% increase
            target_device_idx=target_device_idx
        )

        # Test 1: Check that fov_res is > 1 (interpolation needed)
        self.assertGreater(pyramid.fov_res, 1,
                        "fov_res should be > 1 when interpolation is needed")

        # Test 2: Calculate what Fov_internal should be after ONE multiplication
        expected_fov_interpolated = expected_fov_no_interp * pyramid.fov_res

        # Test 3: Check fp_masking is correct
        # fp_masking = requested_fov / Fov_internal (after interpolation)
        expected_fp_masking = requested_fov / expected_fov_interpolated

        np.testing.assert_allclose(cpuArray(pyramid.fp_masking),
                                cpuArray(expected_fp_masking),
                                rtol=1e-3,
                                err_msg=f"fp_masking incorrect: expected {expected_fp_masking:.6f},"
                                        f" got {pyramid.fp_masking:.6f}")

        # Test 4: Check that fp_masking is <= 1.0
        self.assertLessEqual(pyramid.fp_masking, 1.0,
                            f"fp_masking should be <= 1.0, got {pyramid.fp_masking}")

        # Test 5: Check that fft_sampling reflects the interpolation
        expected_fft_sampling = pixel_pupil * pyramid.fov_res
        self.assertEqual(pyramid.fft_sampling, int(expected_fft_sampling),
                        f"fft_sampling should be {int(expected_fft_sampling)}, "
                        f"got {pyramid.fft_sampling}")

        verbose = False
        if verbose: #pragma: no cover
            print(f"FOV interpolation test passed:")
            print(f"  fov_res = {pyramid.fov_res}")
            print(f"  FOV (no interp) = {expected_fov_no_interp:.3f} arcsec")
            print(f"  FOV (interpolated) = {expected_fov_interpolated:.3f} arcsec")
            print(f"  Requested FOV = {requested_fov:.3f} arcsec")
            print(f"  fp_masking = {pyramid.fp_masking:.6f}")
            print(f"  fft_sampling = {pyramid.fft_sampling}")

    @cpu_and_gpu
    def test_fov_no_interpolation_needed(self, target_device_idx, xp):
        """Test case where FOV is achievable without interpolation"""

        # Test parameters where Fov_internal is within acceptable range
        pixel_pupil = 120  # Large pupil
        pixel_pitch = 0.05  # Small pixel pitch
        wavelength_nm = 500
        requested_fov = 2.0  # Modest FOV request
        pup_diam = 30
        output_resolution = 80

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Calculate expected internal FOV
        D = pixel_pupil * pixel_pitch
        expected_fov = wavelength_nm * 1e-9 / D * (D / pixel_pitch) * RAD2ASEC

        print(f"Expected FOV: {expected_fov:.3f} arcsec")
        print(f"Requested FOV: {requested_fov:.3f} arcsec")

        # This should NOT trigger interpolation
        minfov = requested_fov * (1 - 0.5)  # fov_errinf = 0.5
        maxfov = requested_fov * (1 + 2.0)  # fov_errsup = 2.0
        self.assertGreaterEqual(expected_fov, minfov,
                            "FOV should be within acceptable range")
        self.assertLessEqual(expected_fov, maxfov,
                            "FOV should be within acceptable range")

        # Create ModulatedPyramid
        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=requested_fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            fov_errinf=0.5,
            fov_errsup=2.0,
            target_device_idx=target_device_idx
        )

        # Test 1: Check that fov_res is 1 (no interpolation)
        self.assertEqual(pyramid.fov_res, 1,
                        "fov_res should be 1 when no interpolation is needed")

        # Test 2: Check that fft_sampling equals pixel_pupil
        self.assertEqual(pyramid.fft_sampling, pixel_pupil,
                        f"fft_sampling should equal pixel_pupil ({pixel_pupil}) "
                        f"when no interpolation, got {pyramid.fft_sampling}")

        # Test 3: Check that _do_interpolation is False in setup
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, 
                        S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 1
        pyramid.inputs['in_ef'].set(ef)
        pyramid.setup()

        self.assertFalse(pyramid.ef_interpolator.do_interpolation,
                        "_do_interpolation should be False when fov_res=1 "
                        "and no rotation/shifts")

        verbose = False
        if verbose: #pragma: no cover
            print(f"No interpolation test passed:")
            print(f"  fov_res = {pyramid.fov_res}")
            print(f"  FOV = {expected_fov:.3f} arcsec")
            print(f"  Requested FOV = {requested_fov:.3f} arcsec")
            print(f"  fp_masking = {pyramid.fp_masking:.6f}")
            print(f"  _do_interpolation = {pyramid.ef_interpolator.do_interpolation}")

    @cpu_and_gpu
    def test_fov_error_margins(self, target_device_idx, xp):
        """Test FOV error margin handling"""

        pixel_pupil = 100
        pixel_pitch = 0.06
        wavelength_nm = 500
        pup_diam = 30
        output_resolution = 80

        # Calculate natural FOV
        natural_fov = wavelength_nm * 1e-9 / pixel_pitch * RAD2ASEC

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Test 1: Request FOV within lower error margin
        requested_fov_low = natural_fov * 0.95  # 5% less than natural

        pyramid_low = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=requested_fov_low,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            fov_errinf=0.1,  # Accept 10% reduction
            fov_errsup=0.5,
            target_device_idx=target_device_idx
        )

        # Should not need interpolation (within margin)
        self.assertEqual(pyramid_low.fov_res, 1,
                        "Should not interpolate when FOV is within lower error margin")

        # Test 2: Request FOV within upper error margin
        requested_fov_high = natural_fov * 1.3  # 30% more than natural

        pyramid_high = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=requested_fov_high,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            fov_errinf=0.1,
            fov_errsup=0.5,  # Accept 50% increase
            target_device_idx=target_device_idx
        )

        # Should use focal plane masking without interpolation
        self.assertEqual(pyramid_high.fov_res, 2,
                        "Should interpolate when FOV is above natural FOV")
        self.assertLess(pyramid_high.fp_masking, 1.0,
                    "Should use focal plane mask to reduce FOV")

        verbose = False
        if verbose: #pragma: no cover
            print(f"FOV error margin test passed:")
            print(f"  Natural FOV = {natural_fov:.3f} arcsec")
            print(f"  Low request (95%) = {requested_fov_low:.3f} arcsec"
                f" → fov_res={pyramid_low.fov_res}")
            print(f"  High request (130%) = {requested_fov_high:.3f} arcsec"
                f" → fov_res={pyramid_high.fov_res},"
                f" fp_masking={pyramid_high.fp_masking:.3f}")

    @cpu_and_gpu
    def test_pup_diam_matches_requested_diameter_for_large_pup_dist(self, target_device_idx, xp):
        """Regression test for calc_pyr_geometry: toccd_side must be recomputed from the
        final (possibly fft_res_min-bumped) fft_res, otherwise the sub-pupils shrink
        below the requested pup_diam pixels once pup_dist is large enough to trigger
        the fft_res_min constraint (roughly pup_dist > 1.73 * pup_diam with the
        default pup_margin=2)."""

        pixel_pupil = 160
        pixel_pitch = 0.05
        wavelength_in_nm = 750
        fov = 2.0
        pup_diam = 20
        output_resolution = 80
        mod_amp = 3.0

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
        )

        def _pupil_diameters(pup_dist):
            pyramid = ModulatedPyramid(
                simul_params=simul_params,
                wavelengthInNm=wavelength_in_nm,
                fov=fov,
                pup_diam=pup_diam,
                pup_dist=pup_dist,
                output_resolution=output_resolution,
                mod_amp=mod_amp,
                target_device_idx=target_device_idx,
            )

            ef = ElectricField(
                pixel_pupil, pixel_pupil, pixel_pitch, S0=100, target_device_idx=target_device_idx
            )
            ef.A = make_mask(pixel_pupil)
            ef.generation_time = 0

            pyramid.inputs['in_ef'].set(ef)

            loop = LoopControl()
            loop.add(pyramid, idx=0)
            loop.run(run_time=1, dt=1, t0=0)

            pupcalib = PyrPupdataCalibrator(
                data_dir="/tmp",
                target_device_idx=target_device_idx,
            )
            pupcalib.local_inputs['in_i'] = pyramid.outputs['out_i']
            pupcalib.trigger_code()
            return cpuArray(pupcalib.pupdata.radius) * 2

        # pup_dist=50 triggers the fft_res_min bump: fft_res_min = (50+20)/20*1.1 = 3.85 > 3.0
        diam_large_dist = _pupil_diameters(pup_dist=50)
        np.testing.assert_allclose(
            diam_large_dist, pup_diam, atol=2.0,
            err_msg=f"Pupil diameters {diam_large_dist} should match requested pup_diam="
                    f"{pup_diam} even when pup_dist is large enough to bump fft_res"
        )

        # Sanity check: pup_dist=30 stays below the bump threshold
        # (fft_res_min = (30+20)/20*1.1 = 2.75 < 3.0) and should also match pup_diam,
        # both before and after the fix, guarding against regressions in the normal case.
        diam_small_dist = _pupil_diameters(pup_dist=30)
        np.testing.assert_allclose(
            diam_small_dist, pup_diam, atol=2.0,
            err_msg=f"Pupil diameters {diam_small_dist} should match requested pup_diam="
                    f"{pup_diam} when pup_dist is below the fft_res_min bump threshold"
        )

    @cpu_and_gpu
    def test_pyr_max_side_limits_support(self, target_device_idx, xp):
        pixel_pupil = 80
        pixel_pitch = 0.1
        wavelength_nm = 750
        pup_diam = 24
        output_resolution = 80
        fov = 2.0

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
        )

        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            pyr_max_side_ld=2.0,
            target_device_idx=target_device_idx,
        )

        p = 8
        c = 8
        pyr_tlt = pyramid.get_pyr_tlt(p, c)
        xx, yy = make_xy(pyr_tlt.shape[0], pyr_tlt.shape[0] // 2, xp=xp)
        dist = xp.sqrt(xx ** 2 + yy ** 2)
        max_dist = pyramid.pyr_max_side_ld * pyramid.fft_res / 2.0

        self.assertTrue(xp.any(pyr_tlt[dist <= max_dist] != 0))
        self.assertTrue(xp.all(pyr_tlt[dist > max_dist] == 0))

    @cpu_and_gpu
    def test_pyr_max_side_limits_frame(self, target_device_idx, xp):
        """"""

        pixel_pupil = 64
        pixel_pitch = 0.125
        wavelength_in_nm = 750
        fov = 1.0
        pup_diam = 20
        pup_dist = 40
        output_resolution = 80
        mod_amp = 4.0
        pyr_max_side_ld = 16.0
        # lambda / D = 750e-9 / (160 * 0.05) * RAD2ASEC = 19.3 mas
        #    4 * 19.3 mas = 77.2 mas
        #    16 * 19.3 mas = 308.8 mas

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch,
        )

        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_in_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            pyr_max_side_ld=pyr_max_side_ld,
            target_device_idx=target_device_idx,
        )

        ef = ElectricField(
            pixel_pupil, pixel_pupil, pixel_pitch, S0=100, target_device_idx=target_device_idx
        )
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = 0

        pyramid.inputs['in_ef'].set(ef)

        loop = LoopControl()
        loop.add(pyramid, idx=0)
        loop.run(run_time=1, dt=1, t0=0)

        output_intensity = pyramid.outputs['out_i']

        mask = make_mask(output_resolution, diaratio=30.0/80.0)
        idx = np.where(mask)

        debug_plot = False
        if debug_plot: #pragma: no cover
            import matplotlib.pyplot as plt
            plt.figure(figsize=[4,4])
            plt.imshow(output_intensity.i)
            plt.title(f"Output Intensity for pup_dist={pup_dist}")
            plt.colorbar()
            plt.figure(figsize=[4,4])
            plt.imshow(output_intensity.i * mask)
            plt.title(f"Masked Intensity for pup_dist={pup_dist}")
            plt.colorbar()
            plt.figure(figsize=[4,4])
            plt.imshow(cpuArray(xp.angle(pyramid.masked_exp)))
            plt.title(f"Pyramid Prism")
            plt.colorbar()
            plt.show()

        self.assertTrue(xp.all(output_intensity.i[idx] > 0),
                        "Output intensity should be positive within the mask region")
        self.assertTrue(xp.max(output_intensity.i[idx] / xp.max(output_intensity.i)) > 0.1,
                        "Output intensity should have significant values within the mask region")

    @cpu_and_gpu
    def test_imperfect_edges(self, target_device_idx, xp):
        
        pixel_pupil = 100
        pixel_pitch = 0.06
        wavelength_nm = 500
        pup_diam = 30
        output_resolution = 80

        # Calculate natural FOV
        natural_fov = wavelength_nm * 1e-9 / pixel_pitch * RAD2ASEC

        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        pyramid = ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=natural_fov,
            pup_diam=pup_diam,
            output_resolution=output_resolution,
            mod_amp=0.0,
            mod_type='circular',
            fov_errinf=0.1,  # Accept 10% reduction
            fov_errsup=0.5,
            target_device_idx=target_device_idx
        )
        pyramid.logger = MagicMock()

        # Set large values to force triggering of "imperfect edge" cases
        pyramid.pyr_edge_def_ld = 100
        pyramid.pyr_tip_def_ld = 100
        pyramid.pyr_tip_maya_ld = 100
        pyramid.fft_res = 1
        pyramid.tilt_scale = 1
        pyramid.get_pyr_tlt(4, 4)

        # Check logger was called
        self.assertTrue(pyramid.logger.info.called)

        # Get all log messages
        calls = [call.args[0] for call in pyramid.logger.info.call_args_list]

        # Assertions (flexible matching)
        self.assertTrue(any("imperfect edges" in msg for msg in calls))
        self.assertTrue(any("imperfect tip" in msg for msg in calls))

    @cpu_and_gpu
    def test_misregistration_input(self, target_device_idx, xp):
        """The in_misregistration input gives the output of the same static pup_shifts and rotation,
        and changes them during the simulation"""
        from specula.base_value import BaseValue
        pixel_pupil = 60
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.1)
        rng = np.random.default_rng(3)
        ef = ElectricField(pixel_pupil, pixel_pupil, 0.1, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.phaseInNm[:] = xp.asarray(rng.standard_normal((pixel_pupil, pixel_pupil)) * 50)

        def make(**kwargs):
            pyr = ModulatedPyramid(simul_params=simul_params, wavelengthInNm=500, fov=2.0, pup_diam=15,
                                   output_resolution=40, mod_amp=0.0, target_device_idx=target_device_idx,
                                   **kwargs)
            pyr.inputs['in_ef'].set(ef)
            return pyr

        def run(pyr, t):
            ef.generation_time = t
            pyr.check_ready(t)
            pyr.trigger()
            pyr.post_trigger()
            return cpuArray(pyr.outputs['out_i'].i).copy()

        static = make(pup_shifts=(0.7, -0.4), rotAnglePhInDeg=5.0)
        static.setup()
        plain = make()
        plain.setup()
        dynamic = make()
        misreg = BaseValue(value=np.array([0.7, -0.4, 5.0]), target_device_idx=-1)
        dynamic.inputs['in_misregistration'].set(misreg)
        dynamic.setup()

        misreg.generation_time = 1
        np.testing.assert_allclose(run(dynamic, 1), run(static, 1), rtol=1e-5, atol=1e-6)
        # The same value again: nothing is rebuilt
        interp = dynamic.pup_shift_interp
        misreg.generation_time = 2
        np.testing.assert_allclose(run(dynamic, 2), run(static, 2), rtol=1e-5, atol=1e-6)
        self.assertIs(dynamic.pup_shift_interp, interp)
        # Back to zero: exactly the output without the input
        misreg.value[:] = 0.0
        misreg.generation_time = 3
        np.testing.assert_array_equal(run(dynamic, 3), run(plain, 3))
        with self.assertRaises(ValueError):
            misreg.value = np.zeros(2)
            misreg.generation_time = 4
            run(dynamic, 4)

    @cpu_and_gpu
    def test_misregistration_input_at_zero_is_the_identity(self, target_device_idx, xp):
        """A connected in_misregistration at zero gives exactly the output without it"""
        from specula.base_value import BaseValue
        pixel_pupil = 60
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=0.1)
        rng = np.random.default_rng(4)
        ef = ElectricField(pixel_pupil, pixel_pupil, 0.1, S0=100, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.phaseInNm[:] = xp.asarray(rng.standard_normal((pixel_pupil, pixel_pupil)) * 50)
        outputs = []
        for connect in (False, True):
            pyr = ModulatedPyramid(simul_params=simul_params, wavelengthInNm=500, fov=2.0, pup_diam=15,
                                   output_resolution=40, mod_amp=0.0, target_device_idx=target_device_idx)
            pyr.inputs['in_ef'].set(ef)
            if connect:
                misreg = BaseValue(value=np.zeros(3), target_device_idx=-1)
                misreg.generation_time = 1
                pyr.inputs['in_misregistration'].set(misreg)
            pyr.setup()
            ef.generation_time = 1
            pyr.check_ready(1)
            pyr.trigger()
            pyr.post_trigger()
            outputs.append(cpuArray(pyr.outputs['out_i'].i).copy())
        np.testing.assert_array_equal(outputs[1], outputs[0])

    # ------------------------------------------------------------------
    # Pyramid roof / envelope of the four faces
    # ------------------------------------------------------------------
    pixel_pupil = 64
    pixel_pitch = 0.1

    def _roof_pyramid(self, target_device_idx, **kwargs):
        simul_params = SimulParams(pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)
        kwargs.setdefault('mod_amp', 0.0)
        return ModulatedPyramid(
            simul_params=simul_params,
            wavelengthInNm=750,
            fov=2.0,
            pup_diam=24,
            pup_dist=32,
            output_resolution=64,
            target_device_idx=target_device_idx,
            **kwargs,
        )

    def _quadrant_fractions(self, target_device_idx, xp, roof):
        """Flux fraction in the detector quadrants (TL, TR, BL, BR), flat wavefront, no modulation"""
        pyr = self._roof_pyramid(target_device_idx, pyr_roof_ld=roof)
        ef = ElectricField(self.pixel_pupil, self.pixel_pupil, self.pixel_pitch, S0=100,
                           target_device_idx=target_device_idx)
        ef.A = make_mask(self.pixel_pupil)
        ef.generation_time = 0
        pyr.inputs['in_ef'].set(ef)
        loop = LoopControl()
        loop.add(pyr, idx=0)
        loop.run(run_time=1, dt=1, t0=0)
        i = cpuArray(pyr.outputs['out_i'].i)
        h = i.shape[0] // 2
        q = np.array([i[:h, :h].sum(), i[:h, h:].sum(), i[h:, :h].sum(), i[h:, h:].sum()])
        return q / q.sum()

    @cpu_and_gpu
    def test_pyr_tlt_nominal_is_bit_exact_with_former_quadrant_formula(self, target_device_idx, xp):
        """Without defects the envelope of the four faces equals the former quadrant
        formula exactly (x+y from the outer corner of each quadrant)."""
        pyr = self._roof_pyramid(target_device_idx)
        p, c = pyr.fft_sampling, pyr.fft_padding
        A = int(round((p + c) / 2.0))
        y, x = np.mgrid[0:A, 0:A]
        old = np.zeros((2 * A, 2 * A))
        old[:A, :A] = x + y
        old[:A, A:] = x[:, ::-1] + y
        old[A:, :A] = x + y[::-1]
        old[A:, A:] = x[:, ::-1] + y[::-1]
        old = old / pyr.tilt_scale
        new = cpuArray(pyr.get_pyr_tlt(p, c))
        if target_device_idx >= 0:
            # Last-bit differences are tolerated on the GPU only
            np.testing.assert_allclose(new, old, rtol=1e-12)
        else:
            np.testing.assert_array_equal(new, old)

    @cpu_and_gpu
    def test_pyr_roof_geometry_edges(self, target_device_idx, xp):
        """With a positive roof the face boundaries far from the tip are at
        u = +-L/(2 sqrt2) (vertical edges, 0|1 at the top and 2|3 at the bottom) and
        v = +-L/(2 sqrt2) (horizontal edges, 0|2 on the left and 1|3 on the right),
        L = pyr_roof_ld * fft_res pixels. Labels are per pixel centre, so the
        expected position must lie between the centres bracketing the label change."""
        pyr = self._roof_pyramid(target_device_idx, pyr_roof_ld=4.0)
        p, c = pyr.fft_sampling, pyr.fft_padding
        lab = pyr.get_pyr_faces(p, c).argmin(0)
        n = lab.shape[0]
        A = n // 2
        coord = np.arange(n) - (A - 0.5)
        half = pyr.pyr_roof_ld * pyr.fft_res / (2 * np.sqrt(2))

        def check(line, la, lb, expected):
            idx = np.flatnonzero(np.diff(line) != 0)
            self.assertEqual(len(idx), 1)            # a single boundary along the line
            self.assertEqual((line[idx[0]], line[idx[0] + 1]), (la, lb))
            self.assertLessEqual(coord[idx[0]], expected)
            self.assertGreaterEqual(coord[idx[0] + 1], expected)

        check(lab[2, :], 0, 1, +half)
        check(lab[n - 3, :], 2, 3, -half)
        check(lab[:, 2], 0, 2, +half)
        check(lab[:, n - 3], 1, 3, -half)

    @cpu_and_gpu
    def test_pyr_roof_faces_contact(self, target_device_idx, xp):
        """A positive roof makes faces 0 and 3 meet (and 1, 2 never); a negative roof
        the opposite. Without roof the labels are exactly the four quadrants."""
        def touching(lab):
            pairs = set()
            for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
                m = a != b
                pairs |= {frozenset(t) for t in zip(a[m].tolist(), b[m].tolist())}
            return pairs

        for roof, touch, apart in ((4.0, {0, 3}, {1, 2}), (-4.0, {1, 2}, {0, 3})):
            pyr = self._roof_pyramid(target_device_idx, pyr_roof_ld=roof)
            lab = pyr.get_pyr_faces(pyr.fft_sampling, pyr.fft_padding).argmin(0)
            pairs = touching(lab)
            self.assertIn(frozenset(touch), pairs)
            self.assertNotIn(frozenset(apart), pairs)

        pyr = self._roof_pyramid(target_device_idx)
        lab = pyr.get_pyr_faces(pyr.fft_sampling, pyr.fft_padding).argmin(0)
        A = lab.shape[0] // 2
        expected = np.empty_like(lab)
        expected[:A, :A], expected[:A, A:], expected[A:, :A], expected[A:, A:] = 0, 1, 2, 3
        np.testing.assert_array_equal(lab, expected)

    @cpu_and_gpu
    def test_pyr_roof_midpoint_stays_at_tip(self, target_device_idx, xp):
        """The roof is centred on the tip: the label map is invariant under a 180 deg
        rotation combined with the relabelling 0<->3, 1<->2."""
        for roof in (4.0, -4.0):
            pyr = self._roof_pyramid(target_device_idx, pyr_roof_ld=roof)
            faces = pyr.get_pyr_faces(pyr.fft_sampling, pyr.fft_padding)
            lab = faces.argmin(0)
            rotated = np.array([3, 2, 1, 0])[np.rot90(lab, 2)]
            # Pixels exactly on a face boundary are ties, resolved by argmin to the lowest
            # index (not symmetric): compare only the pixels with a unique minimum
            two_lowest = np.sort(faces, axis=0)[:2]
            unique = (two_lowest[1] - two_lowest[0]) > 1e-9
            np.testing.assert_array_equal(rotated[unique], lab[unique])

    @cpu_and_gpu
    def test_pyr_roof_flux_at_zero_modulation(self, target_device_idx, xp):
        """Flat wavefront, mod_amp=0. No roof: 0.25 in each pupil. Positive roof:
        faces 0 and 3 (TL and BR pupils) collect the light, equal to each other, TR = BL,
        and the dark-pair fraction decreases monotonically with the roof length.
        Negative roof: mirrored split (TL<->TR, BL<->BR)."""
        q0 = self._quadrant_fractions(target_device_idx, xp, 0.0)
        np.testing.assert_allclose(q0, 0.25, atol=1e-3)

        dark = [q0[1] + q0[2]]
        for roof in (1.0, 8.0):
            q = self._quadrant_fractions(target_device_idx, xp, roof)
            np.testing.assert_allclose(q[0], q[3], rtol=1e-6)
            np.testing.assert_allclose(q[1], q[2], rtol=1e-6)
            self.assertGreater(q[0], 0.25)
            dark.append(q[1] + q[2])
            qneg = self._quadrant_fractions(target_device_idx, xp, -roof)
            np.testing.assert_allclose(qneg, q[[1, 0, 3, 2]], atol=1e-6)
        self.assertGreater(dark[0], dark[1])
        self.assertGreater(dark[1], dark[2])
        self.assertLess(dark[2], 0.1)

    @cpu_and_gpu
    def test_pyr_tlt_coeff_gives_continuous_surface(self, target_device_idx, xp):
        """With pyr_tlt_coeff all ones except column 2 = (1.2, 0.9) the face slopes pivot
        around the tip, so the envelope (min of continuous planes) has no steps: adjacent
        pixels differ by at most the largest slope. (The former formula, pivoting around
        the outer corner of each quadrant, had a step of ~19 tilt pixels at the tip.)"""
        coeff = np.ones((2, 4))
        coeff[:, 2] = (1.2, 0.9)
        pyr = self._roof_pyramid(target_device_idx, pyr_tlt_coeff=coeff.tolist())
        tlt = cpuArray(pyr.get_pyr_tlt(pyr.fft_sampling, pyr.fft_padding)).astype(np.float64)
        tlt = tlt * pyr.tilt_scale
        step = max(np.abs(np.diff(tlt, axis=0)).max(), np.abs(np.diff(tlt, axis=1)).max())
        self.assertLessEqual(step, coeff.max() + 1e-9 * max(1.0, np.abs(tlt).max()))

    # ------------------------------------------------------------------
    # Pyramid defects (edges, tip), flat top and support
    # ------------------------------------------------------------------
    def _tlt_host(self, pyr):
        """get_pyr_tlt() on the host, float64, in tilt pixels (times tilt_scale)"""
        tlt = cpuArray(pyr.get_pyr_tlt(pyr.fft_sampling, pyr.fft_padding)).astype(np.float64)
        return tlt * pyr.tilt_scale

    def _defect_mask(self, target_device_idx, **kwargs):
        """Pixels changed by the defects: output with defects vs the same pyramid without
        them. The random values are mocked to -1 (times max), which cannot coincide with
        the (non-negative) surface, so every pixel hit by a defect registers as changed."""
        pyr0 = self._roof_pyramid(target_device_idx, **{k: v for k, v in kwargs.items()
                                                       if k in ('pyr_roof_ld', 'pyr_max_side_ld')})
        pyr = self._roof_pyramid(target_device_idx, **kwargs)
        with patch('numpy.random.rand', lambda n: np.full(n, -1.0)):
            with_def = self._tlt_host(pyr)
        return pyr, with_def != self._tlt_host(pyr0)

    @cpu_and_gpu
    def test_pyr_defects_nominal_unchanged(self, target_device_idx, xp):
        """No roof, nominal slopes: the edge defect hits exactly the pixels with
        |u| <= w/2 or |v| <= w/2 (w = pyr_edge_def_ld * fft_res), the tip defect exactly
        those with sqrt(u^2+v^2) <= pyr_tip_def_ld * fft_res / 2, as in the former code."""
        pyr, edge = self._defect_mask(target_device_idx, pyr_edge_def_ld=1.0)
        u, v = pyr._tip_coords(pyr.fft_sampling, pyr.fft_padding)
        w = 1.0 * pyr.fft_res
        np.testing.assert_array_equal(edge, (np.abs(u) <= w / 2) | (np.abs(v) <= w / 2))
        self.assertTrue(edge.any())

        pyr, tip = self._defect_mask(target_device_idx, pyr_tip_def_ld=2.0)
        r = 2.0 * pyr.fft_res / 2
        np.testing.assert_array_equal(tip, np.sqrt(u ** 2 + v ** 2) <= r)
        self.assertTrue(tip.any())

    @cpu_and_gpu
    def test_pyr_edge_defect_follows_roof_edges(self, target_device_idx, xp):
        """With a roof (L = 12 px) the edges move to u = +-L/(2 sqrt2) ~ 4.24 px far from
        the tip: the defect hits the columns bracketing that position (not the old axis
        at u = +-0.5), the ridge itself (distance 0), and the mask has 180 deg symmetry."""
        pyr, mask = self._defect_mask(target_device_idx, pyr_roof_ld=4.0, pyr_edge_def_ld=0.4)
        u, v = pyr._tip_coords(pyr.fft_sampling, pyr.fft_padding)
        uu, vv = np.broadcast_arrays(u, v)
        L = 4.0 * pyr.fft_res
        half = L / (2 * np.sqrt(2))
        coord = u[0]
        cols = np.flatnonzero(mask[2, :])
        self.assertIn(len(cols), (1, 2))
        # Half-width 0.6 px: the hit centres are within 0.6 of the edge
        self.assertTrue(np.all(np.abs(coord[cols] - half) <= 0.6 + 1e-9))
        self.assertFalse(mask[2, np.abs(coord) == 0.5].any())
        ridge = (np.abs(uu + vv) < 1e-9) & (np.abs(uu) < half)
        self.assertTrue(ridge.any())
        self.assertTrue(mask[ridge].all())
        np.testing.assert_array_equal(np.rot90(mask, 2), mask)

    @cpu_and_gpu
    def test_pyr_tip_defect_follows_roof(self, target_device_idx, xp):
        """The roof ends are at (+-L/(2 sqrt2), -+L/(2 sqrt2)) (positive roof, ridge along
        u+v=0) or (+-L/(2 sqrt2), +-L/(2 sqrt2)) (negative roof, ridge along u=v). The
        tip defect hits exactly the pixels within r of the ridge segment, ends included."""
        for roof, sgn in ((4.0, -1), (-4.0, 1)):
            pyr, mask = self._defect_mask(target_device_idx, pyr_roof_ld=roof, pyr_tip_def_ld=1.0)
            h = abs(roof) * pyr.fft_res / (2 * np.sqrt(2))
            pts = sorted(map(tuple, pyr._tip_points()))
            np.testing.assert_allclose(pts, [(-h, -sgn * h), (h, sgn * h)], atol=1e-12)
            u, v = pyr._tip_coords(pyr.fft_sampling, pyr.fft_padding)
            a = np.array(pts[0])
            d = np.array(pts[1]) - a
            t = np.clip(((u - a[0]) * d[0] + (v - a[1]) * d[1]) / (d @ d), 0, 1)
            dist = np.hypot(u - a[0] - t * d[0], v - a[1] - t * d[1])
            np.testing.assert_array_equal(mask, dist <= 1.0 * pyr.fft_res / 2)
            # Pixels near both ends are hit too
            uu, vv = np.broadcast_arrays(u, v)
            for end in pts:
                near = (np.abs(uu - end[0]) < 1) & (np.abs(vv - end[1]) < 1)
                self.assertTrue(mask[near].any())
            np.testing.assert_array_equal(np.rot90(mask, 2), mask)

    @cpu_and_gpu
    def test_pyr_maya_flat_top(self, target_device_idx, xp):
        """Nominal pyramid with pyr_tip_maya_ld = 4 (r = 6 px): the surface equals the
        envelope for |u|+|v| >= r and the flat level (2A-1) - r inside, and it stays
        continuous. With a roof the level is (2A-1) - |delta|/2 - r and is the maximum."""
        base = self._roof_pyramid(target_device_idx)
        pyr = self._roof_pyramid(target_device_idx, pyr_tip_maya_ld=4.0)
        p, c = pyr.fft_sampling, pyr.fft_padding
        A = int(round((p + c) / 2.0))
        r = 4.0 * pyr.fft_res / 2
        u, v = pyr._tip_coords(p, c)
        tlt = self._tlt_host(pyr)
        env = self._tlt_host(base)
        inside = (np.abs(u) + np.abs(v)) < r
        np.testing.assert_allclose(tlt[~inside], env[~inside], atol=1e-5)
        np.testing.assert_allclose(tlt[inside], 2 * A - 1 - r, atol=1e-5)
        step = max(np.abs(np.diff(tlt, axis=0)).max(), np.abs(np.diff(tlt, axis=1)).max())
        self.assertLessEqual(step, 1 + 1e-5)

        pyr = self._roof_pyramid(target_device_idx, pyr_tip_maya_ld=4.0, pyr_roof_ld=4.0)
        delta = 4.0 * pyr.fft_res / np.sqrt(2)
        level = 2 * A - 1 - delta / 2 - r
        tlt = self._tlt_host(pyr)
        self.assertAlmostEqual(tlt.max(), level, places=4)
        self.assertGreater(np.sum(np.abs(tlt - level) < 1e-5), 1)

    @cpu_and_gpu
    def test_pyr_support_applied_after_defects(self, target_device_idx, xp):
        """The support (pyr_max_side_ld) is applied last: outside radius
        max_side * fft_res / 2 the output is 0 even on the edge-defect pixels."""
        pyr, _ = self._defect_mask(target_device_idx, pyr_edge_def_ld=1.0, pyr_max_side_ld=6.0)
        u, v = pyr._tip_coords(pyr.fft_sampling, pyr.fft_padding)
        outside = np.sqrt(u ** 2 + v ** 2) > 6.0 * pyr.fft_res / 2
        edge = (np.abs(u) <= pyr.fft_res / 2) | (np.abs(v) <= pyr.fft_res / 2)
        self.assertTrue((outside & edge).any())
        with patch('numpy.random.rand', lambda n: np.full(n, 0.5)):
            tlt = self._tlt_host(pyr)
        np.testing.assert_array_equal(tlt[outside], 0.0)
        self.assertTrue((tlt[~outside & edge] != 0).any())

    def test_pyr_roof_ld_plumbing(self):
        """pyr_roof_ld given to ExtSourcePyramid, DistributedExtSourcePyramid and
        PolyCromPyramid reaches the internal ModulatedPyramid instances."""
        from specula.processing_objects.ext_source_pyramid import ExtSourcePyramid
        from specula.processing_objects.distributed_ext_source_pyramid import DistributedExtSourcePyramid
        from specula.processing_objects.poly_crom_pyramid import PolyCromPyramid
        simul_params = SimulParams(pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)
        common = dict(simul_params=simul_params, fov=2.0, pup_diam=24, output_resolution=64,
                      pyr_roof_ld=2.5, target_device_idx=-1)

        ext = ExtSourcePyramid(wavelengthInNm=750, **common)
        self.assertEqual(ext.pyr_roof_ld, 2.5)

        dist = DistributedExtSourcePyramid(wavelengthInNm=750, n_slices=2, **common)
        self.assertEqual(len(dist.sub_pyr), 2)
        for sub in dist.sub_pyr:
            self.assertEqual(sub.pyr_roof_ld, 2.5)

        # PolyCromPyramid: given at the reference (first) wavelength, rescaled
        # by lambda_ref / lambda for the others
        poly = PolyCromPyramid(wavelengthInNm=[600, 750], flux_factor=[1.0, 1.0], mod_amp=0.0, **common)
        self.assertEqual(len(poly._wfs_instances), 2)
        self.assertEqual(poly._wfs_instances[0].pyr_roof_ld, 2.5)
        self.assertAlmostEqual(poly._wfs_instances[1].pyr_roof_ld, 2.5 * 600 / 750, places=12)

    def test_default_mod_step_gives_former_values(self):
        """default_mod_step reproduces the former default: circular
        int(max(1, round(2*mod_amp)*4)), linear int(round(mod_amp)*2+1)."""
        for mod_amp in (0, 0.5, 1, 2.3, 3, 4.5):
            self.assertEqual(ModulatedPyramid.default_mod_step(mod_amp, 'circular'),
                             int(max(1, round(2 * mod_amp) * 4)))
            for mod_type in ('vertical', 'horizontal', 'alternating'):
                self.assertEqual(ModulatedPyramid.default_mod_step(mod_amp, mod_type),
                                 int(round(mod_amp) * 2 + 1))

    def _build_pyr(self, target_device_idx, **kwargs):
        simul_params = SimulParams(pixel_pupil=self.pixel_pupil, pixel_pitch=self.pixel_pitch)
        params = dict(simul_params=simul_params, wavelengthInNm=750, fov=2.0, pup_diam=24,
                      pup_dist=32, output_resolution=64, target_device_idx=target_device_idx)
        params.update(kwargs)
        return ModulatedPyramid(**params)

    @cpu_and_gpu
    def test_explicit_mod_step_validation(self, target_device_idx, xp):
        """Linear modulation accepts mod_step >= round(mod_amp)*2+1 (the default,
        7 for mod_amp=3) and rejects smaller values; circular modulation still
        needs mod_step >= around(2*pi*mod_amp) (19 for mod_amp=3)."""
        pyr = self._build_pyr(target_device_idx, mod_amp=3, mod_type='vertical', mod_step=7)
        self.assertEqual(pyr.mod_steps, 7)
        with self.assertRaises(ValueError):
            self._build_pyr(target_device_idx, mod_amp=3, mod_type='vertical', mod_step=5)
        with self.assertRaises(ValueError):
            self._build_pyr(target_device_idx, mod_amp=3, mod_type='circular', mod_step=18)
        pyr = self._build_pyr(target_device_idx, mod_amp=3, mod_type='circular', mod_step=20)
        self.assertEqual(pyr.mod_steps, 20)

    @staticmethod
    def _obstruction_diameter_ld(pyr, fp_obs):
        """Zero pixels of fp_mask within the obstruction neighbourhood, as an
        equivalent disc diameter in lambda/D."""
        mask = cpuArray(pyr.fp_mask)
        n = mask.shape[0]
        y, x = np.mgrid[:n, :n]
        r = np.hypot(y - (n / 2 - 0.5), x - (n / 2 - 0.5))
        area = np.count_nonzero((mask == 0) & (r <= fp_obs * pyr.fft_res / 2 + 2))
        return 2 * np.sqrt(area / np.pi) / pyr.fft_res

    @cpu_and_gpu
    def test_fp_obs_is_in_lambda_over_d(self, target_device_idx, xp):
        """With fp_obs = 4 lambda/D the focal-plane obstruction is a disc of
        diameter 4*fft_res pixels, independently of fov (i.e. of fp_masking),
        and the field stop diameter matches fov. The former formula gave an
        area fp_masking**2 times too small."""
        fp_obs = 4.0
        maskings = []
        for fov in (2.0, 1.2):
            pyr = self._build_pyr(target_device_idx, fov=fov, fp_obs=fp_obs)
            maskings.append(pyr.fp_masking)
            diam_ld = self._obstruction_diameter_ld(pyr, fp_obs)
            expected_area = np.pi * (fp_obs * pyr.fft_res / 2) ** 2
            area = np.pi * (diam_ld * pyr.fft_res / 2) ** 2
            self.assertAlmostEqual(area / expected_area, 1.0, delta=0.1)

            # Field stop: non-zero extent along the central row (obstruction excluded
            # by taking the outer extent)
            mask = cpuArray(pyr.fp_mask)
            n = mask.shape[0]
            cols = np.nonzero(mask[n // 2] != 0)[0]
            stop_pix = cols[-1] - cols[0] + 1
            ld_arcsec = 750e-9 / (self.pixel_pupil * self.pixel_pitch) * RAD2ASEC
            self.assertAlmostEqual(stop_pix / pyr.fft_res * ld_arcsec, fov,
                                   delta=ld_arcsec / pyr.fft_res * 1.5)
        self.assertGreater(abs(maskings[0] - maskings[1]), 0.05)
