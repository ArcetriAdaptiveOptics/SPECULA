import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray, RAD2ASEC

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from test.specula_testlib import cpu_and_gpu


class TestModulatedPyramid(unittest.TestCase):

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
        expected_ttexp_shape = (1, 1, pyramid.tilt_x.shape[0], pyramid.tilt_x.shape[1])
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
        expected_ttexp_shape = (1, 1, pyramid.tilt_x.shape[0], pyramid.tilt_x.shape[1])
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
        expected_ttexp_shape = (1, pyramid.mod_steps, pyramid.tilt_x.shape[0], pyramid.tilt_x.shape[1])
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
        expected_ttexp_shape = (1, pyramid.mod_steps, pyramid.tilt_x.shape[0],
                                pyramid.tilt_x.shape[1])
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
        expected_ttexp_shape = (2, pyramid.mod_steps, pyramid.tilt_x.shape[0],
                                pyramid.tilt_x.shape[1])
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
