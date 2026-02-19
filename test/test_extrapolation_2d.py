import specula
specula.init(0)

import unittest

from specula import np, cp
from specula import cpuArray
from specula.lib.extrapolation_2d import _calculate_extrapolation_indices_coeffs, _apply_extrapolation
from specula.lib.zernike_generator import ZernikeGenerator
from specula.lib.mask import CircularMask
from specula.data_objects.electric_field import ElectricField
from specula.lib.extrapolation_2d import EFInterpolator


from test.specula_testlib import cpu_and_gpu

import warnings

class TestExtrapolation2D(unittest.TestCase):

    def _create_test_data(self, shape, outer_radius, inner_radius, zernike_mode, xp):
        """
        Create synthetic test data using Zernike polynomials

        Parameters:
        - shape: array dimensions
        - outer_radius: radius of the full circular mask (true data)
        - inner_radius: radius of the smaller mask (data to extrapolate from)
        - zernike_mode: which Zernike mode to use (1=piston, 2=tip, 3=tilt, etc.)

        Returns: true_data, input_data, full_mask, reduced_mask, annular_region
        """
        center = xp.array(shape) / 2.0
        full_mask = CircularMask(shape, outer_radius, center, xp=xp)
        reduced_mask = CircularMask(shape, inner_radius, center, xp=xp)

        zg = ZernikeGenerator(full_mask, xp=xp, dtype=xp.float64)
        true_data = zg.getZernike(zernike_mode)
        input_data = true_data.copy()

        full_mask = full_mask.mask()
        reduced_mask = reduced_mask.mask()

        input_data[reduced_mask] = 0  # Zero outside the reduced mask

        # Annular region: inside full_mask and outside reduced_mask
        annular_region = (~full_mask) & reduced_mask
        return true_data, input_data, full_mask, reduced_mask, annular_region

    @cpu_and_gpu
    def test_extrapolation_piston(self, target_device_idx, xp):
        """
        Test extrapolation with piston (constant) - should be perfect
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=1, xp=xp)

        # Calculate extrapolation coefficients using the reduced mask
        edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(
            ~reduced_mask)  # Invert mask: True=valid, False=invalid

        # Apply extrapolation
        result = _apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp.array(valid_indices),
            xp=xp
        )

        # Check that extrapolated values match the true data in the annular region
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=10,
            err_msg="Piston extrapolation should be perfect"
        )

    @cpu_and_gpu
    def test_extrapolation_tip(self, target_device_idx, xp):
        """
        Test extrapolation with tip (linear in X) - should be very good
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=2, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = _apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp.array(valid_indices),
            xp=xp
        )

        # Check extrapolated values (tip is linear, so extrapolation should be very accurate)
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=5,
            err_msg="Tip extrapolation should be very accurate for linear function"
        )

    @cpu_and_gpu
    def test_extrapolation_tilt(self, target_device_idx, xp):
        """
        Test extrapolation with tilt (linear in Y) - should be very good
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=3, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = _apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp.array(valid_indices),
            xp=xp
        )

        # Check extrapolated values
        np.testing.assert_array_almost_equal(
            cpuArray(result[annular_region]),
            cpuArray(true_data[annular_region]),
            decimal=5,
            err_msg="Tilt extrapolation should be very accurate for linear function"
        )

    @cpu_and_gpu
    def test_extrapolation_focus(self, target_device_idx, xp):
        """
        Test extrapolation with focus (quadratic) - should be reasonable
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=11, zernike_mode=4, xp=xp)

        # Calculate extrapolation coefficients
        edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(
            ~reduced_mask)

        # Apply extrapolation
        result = _apply_extrapolation(
            xp.array(input_data),
            xp.array(edge_pixels),
            xp.array(reference_indices),
            xp.array(coefficients),
            xp.array(valid_indices),
            xp=xp
        )

        # Check extrapolated values (focus is quadratic, so less accurate but should be reasonable)
        # Use RMS error instead of point-by-point comparison for quadratic
        rms_error = xp.sqrt(xp.mean((result[annular_region] - true_data[annular_region])**2))
        true_rms = xp.sqrt(xp.mean(true_data[annular_region]**2))
        relative_error = rms_error / true_rms

        self.assertLess(relative_error, 0.1,
                       f"Focus extrapolation relative RMS error"
                       f" {relative_error:.3f} should be < 10%")

    @cpu_and_gpu
    def test_extrapolation_preserves_input(self, target_device_idx, xp):
        """
        Test that _apply_extrapolation doesn't modify values inside the original mask
        """
        true_data, input_data, full_mask, reduced_mask, annular_region = self._create_test_data(
            shape=(32, 32), outer_radius=12, inner_radius=8, zernike_mode=2, xp=xp)

        # Store original values inside the reduced mask
        original_inside = input_data[~reduced_mask].copy()

        # Calculate and apply extrapolation
        edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(
            ~reduced_mask)
        result = _apply_extrapolation(input_data, edge_pixels, reference_indices,
                                      coefficients, valid_indices, xp=xp)

        # Check that values inside the original mask are unchanged
        np.testing.assert_array_equal(
            cpuArray(result[~reduced_mask]),
            cpuArray(original_inside),
            err_msg="Values inside the input mask should not be modified"
        )

    @cpu_and_gpu
    def test__calculate_extrapolation_indices_coeffs_output_shape(self, target_device_idx, xp):
        """
        Test that the calculation function returns arrays with expected shapes
        """
        # Create a simple test mask
        mask = xp.ones((20, 20), dtype=bool)
        mask[5:15, 5:15] = False  # 10x10 inner region is valid

        edge_pixels, reference_indices, coefficients, _ = _calculate_extrapolation_indices_coeffs(mask)

        # Check shapes are consistent
        n_max = len(edge_pixels)
        self.assertEqual(reference_indices.shape, (n_max, 8))
        self.assertEqual(coefficients.shape, (n_max, 8))

        # Check that we have some valid edge pixels
        valid_pixels = xp.sum(edge_pixels >= 0)
        self.assertGreater(valid_pixels, 0)

    def test_extrapolation_does_not_fail_with_many_edge_pixels(self):
        """
        Test that extrapolation does not fail when the mask has many edge pixels
        (multiple square perimeters)
        """

        n_pixels = 128
        shape = (n_pixels, n_pixels)
        mask = np.zeros(shape, dtype=bool)

        # Add several square perimeters (concentric)
        for r in range(2, n_pixels//2-1, 3):  # radii: 2, 5, 8, ...
            mask[r:-r, r] = True
            mask[r:-r, -r] = True
            mask[r, r:-r] = True
            mask[-r, r:-r] = True
            mask[-r, -r] = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Now mask has many edge pixels (all perimeters)
            edge_pixels, reference_indices, coefficients, valid_indices = _calculate_extrapolation_indices_coeffs(mask)
            self.assertTrue(any(issubclass(warning.category, RuntimeWarning) for warning in w),
                            "Should trigger a RuntimeWarning for too many edge pixels")

        # Even if valid pixels are more than 25% of total, extrapolation should not fail
        self.assertTrue(len(valid_indices) > 0.25 * (n_pixels * n_pixels),
                        "Extrapolation should not fail completely, should have some valid indices.")

    @cpu_and_gpu
    def test_cache_sharing_between_instances(self, target_device_idx, xp):
        """
        Test that multiple EFInterpolator instances share the same cached arrays
        when they have the same parameters.
        """
        # Create two identical ElectricFields
        ef_size = (32, 32)
        pixel_pitch = 0.01

        ef1 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=target_device_idx)
        ef2 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=target_device_idx)

        # Set some data
        ef1.A[:] = 1.0
        ef1.phaseInNm[:] = 0.0
        ef2.A[:] = 1.0
        ef2.phaseInNm[:] = 0.0

        # Clear cache before test
        EFInterpolator._EFInterpolator__zeros_cache.clear()
        EFInterpolator._EFInterpolator__ef_cache.clear()

        # Create first interpolator - should allocate cache
        interp1 = EFInterpolator(ef1, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_after_first = len(EFInterpolator._EFInterpolator__zeros_cache)
        self.assertGreater(cache_size_after_first, 0,
                          "Cache should have entries after first interpolator")

        # Get the cached array ID
        phase_extrap_id_1 = id(interp1.phase_extrapolated)

        # Create second interpolator with same parameters - should reuse cache
        interp2 = EFInterpolator(ef2, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_after_second = len(EFInterpolator._EFInterpolator__zeros_cache)
        self.assertEqual(cache_size_after_first, cache_size_after_second,
                        "Cache size should not increase for identical parameters")

        # Check that the same array is used
        phase_extrap_id_2 = id(interp2.phase_extrapolated)
        self.assertEqual(phase_extrap_id_1, phase_extrap_id_2,
                        "Both interpolators should share the same cached array")

    @cpu_and_gpu
    def test_cache_different_devices(self, target_device_idx, xp):
        """
        Test that cache creates separate entries for different devices.
        If only CPU is available, test that cache is reused for same device.
        """
        ef_size = (32, 32)
        pixel_pitch = 0.01

        ef1 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=target_device_idx)
        ef1.A[:] = 1.0
        ef1.phaseInNm[:] = 0.0

        # Clear cache
        EFInterpolator._EFInterpolator__zeros_cache.clear()
        EFInterpolator._EFInterpolator__ef_cache.clear()

        # Create interpolator on first device
        interp1 = EFInterpolator(ef1, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_first = len(EFInterpolator._EFInterpolator__zeros_cache)
        self.assertGreater(cache_size_first, 0, "Cache should have entries")

        # Determine other device
        other_device = -1 # it is CPU
        devices_are_different = (cp is not None) and (target_device_idx != other_device)

        ef2 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=other_device)
        ef2.A[:] = 1.0
        ef2.phaseInNm[:] = 0.0

        interp2 = EFInterpolator(ef2, (64, 64),
                                target_device_idx=other_device,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_second = len(EFInterpolator._EFInterpolator__zeros_cache)

        if devices_are_different:
            # Different devices should have separate cache entries
            self.assertGreater(cache_size_second, cache_size_first,
                              "Different devices should have separate cache entries")
            self.assertNotEqual(id(interp1.phase_extrapolated),
                              id(interp2.phase_extrapolated),
                              "Different devices should use different arrays")
        else:
            # Same device should reuse cache
            self.assertEqual(cache_size_first, cache_size_second,
                           "Same device should reuse cache entries")
            self.assertEqual(id(interp1.phase_extrapolated),
                           id(interp2.phase_extrapolated),
                           "Same device should share cached arrays")

    @cpu_and_gpu
    def test_cache_different_shapes(self, target_device_idx, xp):
        """
        Test that cache creates separate entries for different array shapes.
        """
        pixel_pitch = 0.01

        # Clear cache
        EFInterpolator._EFInterpolator__zeros_cache.clear()
        EFInterpolator._EFInterpolator__ef_cache.clear()

        # Create interpolator with first shape
        ef1 = ElectricField(32, 32, pixel_pitch, target_device_idx=target_device_idx)
        ef1.A[:] = 1.0
        ef1.phaseInNm[:] = 0.0

        interp1 = EFInterpolator(ef1, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_first = len(EFInterpolator._EFInterpolator__zeros_cache)

        # Create interpolator with different input shape
        ef2 = ElectricField(64, 64, pixel_pitch, target_device_idx=target_device_idx)
        ef2.A[:] = 1.0
        ef2.phaseInNm[:] = 0.0

        interp2 = EFInterpolator(ef2, (128, 128),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        cache_size_second = len(EFInterpolator._EFInterpolator__zeros_cache)

        # Cache should have more entries for different shapes
        self.assertGreater(cache_size_second, cache_size_first,
                          "Cache should create separate entries for different shapes")

        # Arrays should be different
        self.assertNotEqual(id(interp1.phase_extrapolated),
                          id(interp2.phase_extrapolated),
                          "Different shapes should have different cached arrays")

    @cpu_and_gpu
    def test_cache_reuse_after_interpolation(self, target_device_idx, xp):
        """
        Test that cached arrays can be safely reused after interpolation
        (they should work like scratch buffers).
        """
        ef_size = (32, 32)
        pixel_pitch = 0.01

        # Clear cache
        EFInterpolator._EFInterpolator__zeros_cache.clear()
        EFInterpolator._EFInterpolator__ef_cache.clear()

        # Create two interpolators
        ef1 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=target_device_idx)
        ef2 = ElectricField(ef_size[0], ef_size[1], pixel_pitch,
                           target_device_idx=target_device_idx)

        # Set different phase data
        center = xp.array(ef_size) / 2.0
        y, x = xp.mgrid[0:ef_size[0], 0:ef_size[1]]

        ef1.A[:] = 1.0
        ef1.phaseInNm[:] = ((x - center[1])**2 + (y - center[0])**2).astype(ef1.dtype)

        ef2.A[:] = 1.0
        ef2.phaseInNm[:] = (2 * ((x - center[1])**2 + (y - center[0])**2)).astype(ef2.dtype)

        interp1 = EFInterpolator(ef1, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)
        interp2 = EFInterpolator(ef2, (64, 64),
                                target_device_idx=target_device_idx,
                                force_extrapolation=True,
                                use_out_ef_cache=True)

        # Interpolate with first interpolator
        interp1.interpolate()
        result1 = cpuArray(interp1.interpolated_ef().phaseInNm.copy())

        # Interpolate with second interpolator (reuses cache)
        interp2.interpolate()
        result2 = cpuArray(interp2.interpolated_ef().phaseInNm.copy())

        # Results should be different (proving cache was properly overwritten)
        self.assertFalse(xp.allclose(result1, result2),
                        "Results should differ, proving cache reuse works correctly")

        # Interpolate again with first interpolator
        interp1.interpolate()
        result1_again = cpuArray(interp1.interpolated_ef().phaseInNm.copy())

        # Should get same result as first time
        np.testing.assert_array_almost_equal(result1, result1_again, decimal=5,
                                            err_msg="Re-interpolation should give same result")

    @cpu_and_gpu
    def test_efinterpolator_update_parameters(self, target_device_idx, xp):
        """
        Test that EFInterpolator correctly passes magnification to Interp2D
        and that update_parameters correctly re-initializes the interpolator.
        """
        ef_size = (32, 32)
        pixel_pitch = 0.01

        ef_in = ElectricField(ef_size[0], ef_size[1],
                              pixel_pitch, target_device_idx=target_device_idx)
        ef_in.A[:] = 1.0

        # 1. Test initial setup with magnification
        interp = EFInterpolator(ef_in, (64, 64), magnification=2.0,
                                target_device_idx=target_device_idx)

        # Verify initial values are passed down to Interp2D
        self.assertEqual(float(interp.interp.magnification), 2.0)
        self.assertEqual(float(interp.interp.shift_x), 0.0)

        # 2. Test dynamic update of parameters
        interp.update_parameters(xShiftPhInPixel=1.5, magnification=0.5)

        # Verify the internal Interp2D was recreated with new values
        self.assertEqual(float(interp.interp.magnification), 0.5)
        # Note: x and y are swapped in Interp2D because of the way shifts are defined
        # (colShift = xShift, rowShift = yShift)
        self.assertEqual(float(interp.interp.shift_y), 1.5)

        # Do a dummy interpolation to ensure it doesn't crash with the updated object
        interp.interpolate()
        self.assertEqual(interp.interpolated_ef().size, (64, 64))
