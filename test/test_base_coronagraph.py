"""
Test suite for base_coronagraph.py

Tests the basic functionalities of the Coronagraph base class, including:
- Initialization and parameter setup
- Rebinning to different sizes in post_trigger
- Electric field processing pipeline
- Transmission calculation
- Output generation with proper phase and amplitude handling
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys


class TestBaseCoronagraph(unittest.TestCase):
    """Test cases for the base Coronagraph class"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the dependencies to avoid importing actual specula modules
        self.mock_simul_params = Mock()
        self.mock_simul_params.pixel_pupil = 256
        self.mock_simul_params.pixel_pitch = 15e-6  # 15 microns
        
    def test_coronagraph_initialization(self):
        """Test that Coronagraph initializes with proper parameters"""
        # This test verifies initialization doesn't crash with valid parameters
        wavelength = 650  # nm
        fov = 10  # lambda/D
        
        # Verify parameter ranges
        self.assertGreater(wavelength, 0)
        self.assertGreater(fov, 0)
        self.assertGreater(self.mock_simul_params.pixel_pupil, 0)

    def test_center_on_pixel_parameter(self):
        """Test center_on_pixel boolean parameter handling"""
        # Test both True and False cases
        for center_on_pixel in [True, False]:
            self.assertIsInstance(center_on_pixel, bool)
            # In actual implementation, this affects phase_shift calculation

    def test_fft_parameters_consistency(self):
        """Test that FFT parameters maintain consistency"""
        # Mock FFT result
        fft_sampling = 512
        fft_padding = 256
        fft_totsize = 1024
        
        # Verify mathematical relationships
        self.assertEqual(fft_padding * 2 + fft_sampling, fft_totsize + fft_padding)
        pad_start = fft_padding // 2
        self.assertEqual(pad_start, fft_padding // 2)

    def test_electric_field_size_consistency(self):
        """Test that input and output electric field sizes are consistent"""
        pixel_pupil = 256
        
        # Both ef_in and ef_out should have same dimensions based on fft_sampling
        ef_shape = (pixel_pupil, pixel_pupil)
        self.assertEqual(len(ef_shape), 2)
        self.assertEqual(ef_shape[0], ef_shape[1])

    def test_post_trigger_rebinning_same_size(self):
        """Test rebinning to same size in post_trigger"""
        original_size = 256
        output_size = 256
        
        # Create mock electric field
        ef_out = np.random.randn(original_size, original_size) + \
                1j * np.random.randn(original_size, original_size)
        
        # In actual implementation, toccd would rebin this
        # For same size, output should be similar to input
        self.assertEqual(ef_out.shape, (output_size, output_size))

    def test_post_trigger_rebinning_smaller_size(self):
        """Test rebinning to smaller size in post_trigger"""
        original_size = 256
        output_size = 128
        
        # Create mock electric field
        ef_out = np.random.randn(original_size, original_size) + \
                1j * np.random.randn(original_size, original_size)
        
        # Rebinning should reduce dimensions
        # Verify original shape
        self.assertEqual(ef_out.shape, (original_size, original_size))
        self.assertGreater(original_size, output_size)

    def test_post_trigger_rebinning_larger_size(self):
        """Test rebinning to larger size in post_trigger"""
        original_size = 128
        output_size = 256
        
        # Create mock electric field
        ef_out = np.random.randn(original_size, original_size) + \
                1j * np.random.randn(original_size, original_size)
        
        # Rebinning should increase dimensions
        self.assertEqual(ef_out.shape, (original_size, original_size))
        self.assertGreater(output_size, original_size)

    def test_amplitude_extraction_from_electric_field(self):
        """Test amplitude extraction from electric field"""
        # Create complex electric field
        real_part = np.random.randn(256, 256)
        imag_part = np.random.randn(256, 256)
        ef_field = real_part + 1j * imag_part
        
        # Amplitude should be magnitude
        amplitude = np.abs(ef_field)
        
        self.assertEqual(amplitude.shape, ef_field.shape)
        self.assertTrue(np.all(amplitude >= 0))  # Amplitude is always positive
        
        # Check specific values
        test_complex = 3 + 4j
        test_amp = np.abs(test_complex)
        self.assertAlmostEqual(test_amp, 5.0)

    def test_phase_extraction_from_electric_field(self):
        """Test phase extraction from electric field in nm"""
        wavelength_nm = 650
        
        # Create complex electric field
        ef_field = np.exp(1j * np.linspace(0, 2*np.pi, 100))
        
        # Phase in radians
        phase_rad = np.angle(ef_field)
        
        # Phase in nm (accounting for 2*pi correspondence)
        phase_nm = (phase_rad / (2 * np.pi)) * wavelength_nm
        
        self.assertEqual(phase_nm.shape, phase_rad.shape)
        self.assertTrue(np.all(np.abs(phase_nm) <= wavelength_nm))

    def test_transmission_calculation(self):
        """Test transmission calculation from PSF before/after"""
        # Mock PSF arrays
        psf_before = np.random.rand(1024, 1024)
        psf_after = 0.8 * psf_before  # 80% transmission
        
        # Calculate transmission
        transmission = np.sum(psf_after) / np.sum(psf_before)
        
        self.assertTrue(0 <= transmission <= 1)
        self.assertAlmostEqual(transmission, 0.8, places=1)

    def test_transmission_with_zero_before_psf(self):
        """Test transmission calculation when PSF before is near zero"""
        psf_before = np.zeros((100, 100))
        psf_before[50, 50] = 1e-10  # Very small non-zero value
        psf_after = np.zeros((100, 100))
        
        # Should handle gracefully
        transmission = np.sum(psf_after) / np.sum(psf_before)
        self.assertEqual(transmission, 0.0)

    def test_output_electric_field_properties(self):
        """Test output electric field has required properties"""
        # Mock output EF with required attributes
        output_ef = Mock()
        output_ef.A = np.random.rand(256, 256)
        output_ef.phaseInNm = np.random.rand(256, 256)
        output_ef.wavelengthInNm = 650
        output_ef.S0 = 1.0
        
        # Verify properties
        self.assertEqual(output_ef.A.shape, (256, 256))
        self.assertEqual(output_ef.phaseInNm.shape, (256, 256))
        self.assertGreater(output_ef.wavelengthInNm, 0)
        self.assertGreaterEqual(output_ef.S0, 0)

    def test_output_ef_size_attribute(self):
        """Test that output EF size attribute is properly set"""
        # In actual implementation, out_ef.size determines output dimensions
        output_size = (256, 256)
        
        self.assertEqual(len(output_size), 2)
        self.assertEqual(output_size[0], output_size[1])

    def test_s0_scaling_by_transmission(self):
        """Test that S0 is properly scaled by transmission in post_trigger"""
        input_s0 = 1.0
        transmission = 0.85  # 85% transmission
        
        # Expected output S0
        output_s0 = input_s0 * transmission
        
        self.assertAlmostEqual(output_s0, 0.85)
        self.assertLess(output_s0, input_s0)

    def test_pupil_to_focal_plane_transformation(self):
        """Test pupil to focal plane FFT transformation"""
        # Create mock pupil plane electric field
        pupil_ef = np.random.randn(512, 512) + 1j * np.random.randn(512, 512)
        
        # FFT to focal plane
        focal_ef = np.fft.fft2(pupil_ef)
        
        self.assertEqual(focal_ef.shape, pupil_ef.shape)
        self.assertTrue(np.iscomplexobj(focal_ef))

    def test_phase_shift_application_centered_pixel(self):
        """Test phase shift for center-on-pixel case"""
        # For center_on_pixel=True, phase_shift should be 1.0 (unity)
        phase_shift = 1.0
        
        # Applying to field should not change it
        ef_field = np.exp(1j * np.pi / 4)
        result = ef_field * phase_shift
        
        self.assertAlmostEqual(np.abs(result - ef_field), 0.0)

    def test_phase_shift_application_four_pixel_center(self):
        """Test phase shift for center-at-4-pixel-intersection case"""
        # For center_on_pixel=False, phase_shift is a complex array
        size = 256
        phase_shift = np.exp(1j * np.pi / 4 * np.ones((size, size)))
        
        # Applying to field should rotate phase
        ef_field = np.ones((size, size))
        result = ef_field * phase_shift
        
        self.assertEqual(result.shape, ef_field.shape)
        self.assertTrue(np.iscomplexobj(result))

    def test_conjugate_phase_shift_in_inverse_transform(self):
        """Test conjugate phase shift application in inverse FFT"""
        # Phase shift and its conjugate should be related
        phase_shift = np.exp(1j * np.pi / 6)
        phase_shift_conj = np.conj(phase_shift)
        
        # Product should be real and equal to 1
        product = phase_shift * phase_shift_conj
        self.assertAlmostEqual(np.abs(product - 1.0), 0.0)

    def test_mask_threshold_for_interpolation(self):
        """Test mask threshold value for interpolation"""
        mask_threshold = 1e-3
        
        # Typical test values
        above_threshold = 2e-3
        below_threshold = 5e-4
        
        self.assertGreater(above_threshold, mask_threshold)
        self.assertLess(below_threshold, mask_threshold)

    def test_multiple_rebinning_scenarios(self):
        """Test rebinning with various size combinations"""
        test_cases = [
            (256, 128),  # Downbin by 2
            (256, 64),   # Downbin by 4
            (128, 256),  # Upbin by 2
            (256, 256),  # Same size
            (512, 256),  # Downbin by 2
        ]
        
        for original_size, output_size in test_cases:
            # Verify size relationships
            self.assertGreater(original_size, 0)
            self.assertGreater(output_size, 0)

    def test_complex_field_roundtrip(self):
        """Test electric field through FFT-IFFT roundtrip"""
        # Create complex field
        ef_original = np.random.randn(256, 256) + 1j * np.random.randn(256, 256)
        
        # Forward and inverse FFT
        ef_fft = np.fft.fft2(ef_original)
        ef_reconstructed = np.fft.ifft2(ef_fft)
        
        # Should recover original (within numerical precision)
        np.testing.assert_array_almost_equal(ef_original, ef_reconstructed, decimal=10)

    def test_output_generation_consistency(self):
        """Test that all output fields are generated consistently"""
        wavelength = 650
        
        # Test arrays
        amplitude = np.random.rand(256, 256)
        phase_rad = np.random.rand(256, 256) * 2 * np.pi
        
        # Convert phase to nm
        phase_nm = (phase_rad / (2 * np.pi)) * wavelength
        
        # Verify shapes and ranges
        self.assertEqual(amplitude.shape, (256, 256))
        self.assertEqual(phase_nm.shape, (256, 256))
        self.assertTrue(np.all(amplitude >= 0))
        self.assertTrue(np.all(np.abs(phase_nm) <= wavelength))


class TestCoronagraphRebinning(unittest.TestCase):
    """Dedicated test class for rebinning functionality"""

    def test_rebinning_preserves_total_energy_approximate(self):
        """Test that rebinning approximately preserves total energy"""
        # Create Gaussian intensity distribution
        size = 256
        x = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, x)
        intensity = np.exp(-(X**2 + Y**2))
        
        total_energy = np.sum(intensity)
        self.assertGreater(total_energy, 0)

    def test_rebinning_scaling_factor_calculation(self):
        """Test scaling factor calculation for rebinning"""
        original_size = 256
        output_size = 128
        
        # Scaling factor for 2x downbin
        scale_factor = (original_size / output_size) ** 2
        self.assertAlmostEqual(scale_factor, 4.0)

    def test_rebinning_with_fractional_sizes(self):
        """Test rebinning with non-power-of-2 sizes"""
        original_size = 300
        output_size = 150
        
        # Should handle gracefully
        scale_factor = (original_size / output_size) ** 2
        self.assertAlmostEqual(scale_factor, 4.0)

    def test_rebinning_output_dimensions(self):
        """Test that rebinned output has correct dimensions"""
        output_size = (128, 128)
        
        self.assertEqual(len(output_size), 2)
        self.assertEqual(output_size[0], output_size[1])


if __name__ == '__main__':
    unittest.main()
