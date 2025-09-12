
import specula
specula.init(0)  # Default target device

import unittest
import numpy as np
from specula import cpuArray
from specula.data_objects.iir_filter_data import IirFilterData
from test.specula_testlib import cpu_and_gpu

# Try to import control library for testing
try:
    import control
    CONTROL_AVAILABLE = True
except ImportError:
    CONTROL_AVAILABLE = False
    control = None

class TestIirFilterData(unittest.TestCase):

    @cpu_and_gpu
    def test_init_with_n_modes_expansion(self, target_device_idx, xp):
        """Test that n_modes expands filter blocks correctly in IirFilterData.__init__"""
        ordnum = [2, 2]
        ordden = [2, 2]
        num = xp.array([[0.0, 0.5], [0.0, 0.3]])
        den = xp.array([[-1.0, 1.0], [-0.9, 1.0]])
        n_modes = [3, 2]

        filt = IirFilterData(ordnum, ordden, num, den, n_modes=n_modes, target_device_idx=target_device_idx)

        # There should be 5 filters: 3 with the first coeffs, 2 with the second
        self.assertEqual(filt.num.shape, (5, 2))
        self.assertEqual(filt.den.shape, (5, 2))
        np.testing.assert_allclose(cpuArray(filt.num[:3]), [cpuArray(num[0])]*3)
        np.testing.assert_allclose(cpuArray(filt.num[3:]), [cpuArray(num[1])]*2)
        np.testing.assert_allclose(cpuArray(filt.den[:3]), [cpuArray(den[0])]*3)
        np.testing.assert_allclose(cpuArray(filt.den[3:]), [cpuArray(den[1])]*2)

    @cpu_and_gpu
    def test_numerator_from_gain_and_ff(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)

        # Original assertions
        assert all(cpuArray(f.num[:, 0]) == 0)
        assert all(cpuArray(f.num[:, 1]) == 0.2)

        # Debug plot
        debug_plot = False
        if debug_plot:
            self._debug_plot_filter(f, "numerator_gain_ff", target_device_idx)
            # Debug print
            print(f"\nDEBUG - test_numerator_from_gain_and_ff (device {target_device_idx}):")
            print(f"Numerator shape: {f.num.shape}")
            print(f"Numerator coefficients:\n{cpuArray(f.num)}")
            print(f"Expected: all num[:, 0] = 0, all num[:, 1] = {gain}")

    @cpu_and_gpu
    def test_denominator_from_gain_and_ff_num(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)

        assert all(cpuArray(f.den[:, 0]) == -1)
        assert all(cpuArray(f.den[:, 1]) == 1)

    @cpu_and_gpu
    def test_num_and_den_shape_from_gain_and_ff_num(self, target_device_idx, xp):
        gain = 0.2
        nmodes = 10
        f = IirFilterData.from_gain_and_ff([gain] * nmodes, target_device_idx=target_device_idx)

        assert f.num.shape == (nmodes, 2)
        assert f.den.shape == (nmodes, 2)

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    @cpu_and_gpu
    def test_control_conversion_roundtrip_single_filter(self, target_device_idx, xp):
        """Test roundtrip conversion: IirFilterData -> control.TransferFunction -> IirFilterData"""

        # Create a simple low-pass filter
        fc = 50  # Hz
        fs = 1000  # Hz
        original_filter = IirFilterData.lpf_from_fc(fc, fs, n_ord=2, target_device_idx=target_device_idx)

        # Convert to control transfer function
        dt = 1.0 / fs
        tf = original_filter.to_control_tf(mode=0, dt=dt)

        # Convert back to IirFilterData
        reconstructed_filter = IirFilterData.from_control_tf(tf, target_device_idx=target_device_idx)

        # Compare coefficients (allowing for small numerical differences)
        original_num = cpuArray(original_filter.num[0, :])
        original_den = cpuArray(original_filter.den[0, :])
        reconstructed_num = cpuArray(reconstructed_filter.num[0, :])
        reconstructed_den = cpuArray(reconstructed_filter.den[0, :])

        # Remove trailing zeros for comparison
        original_num = original_num[original_num != 0] if np.any(original_num != 0) else np.array([0])
        original_den = original_den[original_den != 0] if np.any(original_den != 0) else np.array([1])
        reconstructed_num = reconstructed_num[reconstructed_num != 0] if np.any(reconstructed_num != 0) else np.array([0])
        reconstructed_den = reconstructed_den[reconstructed_den != 0] if np.any(reconstructed_den != 0) else np.array([1])

        np.testing.assert_allclose(original_num, reconstructed_num, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(original_den, reconstructed_den, rtol=1e-10, atol=1e-12)

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    @cpu_and_gpu
    def test_control_conversion_roundtrip_multiple_filters(self, target_device_idx, xp):
        """Test roundtrip conversion with multiple filters"""

        # Create multiple filters with different cutoff frequencies
        fc_list = [10, 50, 100]  # Hz
        fs = 1000  # Hz
        original_filter = IirFilterData.lpf_from_fc(fc_list, fs, n_ord=2, target_device_idx=target_device_idx)

        # Convert to list of control transfer functions
        dt = 1.0 / fs
        tf_list = original_filter.to_control_tf_list(dt=dt)

        # Verify we have the right number of transfer functions
        self.assertEqual(len(tf_list), len(fc_list))

        # Convert back to IirFilterData
        reconstructed_filter = IirFilterData.from_control_tf(tf_list, target_device_idx=target_device_idx)

        # Compare each filter
        for mode in range(len(fc_list)):
            original_num = cpuArray(original_filter.num[mode, :])
            original_den = cpuArray(original_filter.den[mode, :])
            reconstructed_num = cpuArray(reconstructed_filter.num[mode, :])
            reconstructed_den = cpuArray(reconstructed_filter.den[mode, :])

            # Remove trailing zeros for comparison
            original_num = original_num[original_num != 0] if np.any(original_num != 0) else np.array([0])
            original_den = original_den[original_den != 0] if np.any(original_den != 0) else np.array([1])
            reconstructed_num = reconstructed_num[reconstructed_num != 0] if np.any(reconstructed_num != 0) else np.array([0])
            reconstructed_den = reconstructed_den[reconstructed_den != 0] if np.any(reconstructed_den != 0) else np.array([1])

            np.testing.assert_allclose(original_num, reconstructed_num, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Numerator mismatch for filter {mode}")
            np.testing.assert_allclose(original_den, reconstructed_den, rtol=1e-10, atol=1e-12,
                                     err_msg=f"Denominator mismatch for filter {mode}")

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    @cpu_and_gpu
    def test_control_conversion_with_gain_and_ff(self, target_device_idx, xp):
        """Test conversion with gain and forgetting factor filters"""

        gains = [0.1, 0.5, 1.0]
        ff = [0.9, 0.95, 0.99]
        original_filter = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # Convert to control and back
        tf_list = original_filter.to_control_tf_list(dt=1.0)  # Assume unit sampling time
        reconstructed_filter = IirFilterData.from_control_tf(tf_list, target_device_idx=target_device_idx)

        # Compare all filters using frequency response instead of direct coefficient comparison
        freq = np.logspace(-2, 0, 100)  # Test frequencies

        for mode in range(len(gains)):
            # Get frequency responses
            original_response = original_filter.frequency_response(
                original_filter.num[mode, :], original_filter.den[mode, :],
                fs=1.0, freq=freq
            )

            reconstructed_response = reconstructed_filter.frequency_response(
                reconstructed_filter.num[mode, :], reconstructed_filter.den[mode, :],
                fs=1.0, freq=freq
            )

            # Compare frequency responses (which should be identical)
            np.testing.assert_allclose(
                np.abs(original_response), np.abs(reconstructed_response), 
                rtol=1e-12, atol=1e-14,
                err_msg=f"Magnitude response mismatch for filter {mode}"
            )
            np.testing.assert_allclose(
                np.angle(original_response), np.angle(reconstructed_response), 
                rtol=1e-12, atol=1e-14,
                err_msg=f"Phase response mismatch for filter {mode}"
            )

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    def test_control_conversion_error_handling(self):
        """Test error handling when control library methods are called without control"""

        # Use unittest.mock to temporarily disable control
        import unittest.mock
        import specula.data_objects.iir_filter_data as iir_module

        with unittest.mock.patch.object(iir_module, 'CONTROL_AVAILABLE', False):
            # Create a filter
            filter_data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=None)

            # Test that control-dependent methods raise ImportError
            with self.assertRaises(ImportError):
                filter_data.to_control_tf()

            with self.assertRaises(ImportError):
                filter_data.bode_plot()

            with self.assertRaises(ImportError):
                filter_data.nyquist_plot()

            with self.assertRaises(ImportError):
                IirFilterData.from_control_tf([])

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    @cpu_and_gpu
    def test_frequency_response_consistency(self, target_device_idx, xp):
        """Test that frequency response is consistent between control and manual computation"""

        # Create a simple filter
        fc = 100  # Hz
        fs = 1000  # Hz
        filter_data = IirFilterData.lpf_from_fc(fc, fs, n_ord=2, target_device_idx=target_device_idx)
        # Frequency vector for testing - limit to avoid Nyquist frequency warnings
        freq = np.logspace(0, np.log10(fs/2 - 10), 100)  # 1 to 490 Hz (avoid Nyquist at 500 Hz)

        # Get transfer function using control library
        tf_control = filter_data.to_control_tf(mode=0, dt=1.0/fs)

        # Calculate frequency response using control library
        omega = 2 * np.pi * freq
        response_control = tf_control.frequency_response(omega)

        # Get transfer function using manual computation
        response_manual = filter_data.frequency_response(
            filter_data.num[0, :], filter_data.den[0, :], fs,
            freq=freq
        )

        plot_debug = False  # Set to True to enable plotting for debugging
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(freq, np.abs(response_control.magnitude), label='Control Response')
            plt.plot(freq, np.abs(response_manual), label='Manual Response', linestyle='--')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Magnitude')
            plt.title('Frequency Response Comparison')
            plt.legend()
            plt.grid()
            plt.figure()
            plt.plot(freq, response_control.phase, label='Control Response')
            plt.plot(freq, np.angle(response_manual), label='Manual Response', linestyle='--')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase [radians]')
            plt.title('Phase Response Comparison')
            plt.legend()
            plt.grid()
            plt.show()

        # Compare responses (allowing for small numerical differences)
        np.testing.assert_allclose(
            np.abs(np.squeeze(response_control.magnitude)), np.abs(response_manual),
            rtol=1e-10, atol=1e-12,
            err_msg="Magnitude response mismatch between control and manual computation"
        )
        np.testing.assert_allclose(
            np.squeeze(response_control.phase), np.angle(response_manual),
            rtol=1e-10, atol=1e-12,
            err_msg="Phase response mismatch between control and manual computation"
        )

    def test_has_control_support_property(self):
        """Test the has_control_support property"""

        filter_data = IirFilterData.from_gain_and_ff([0.5])

        # The property should return the same as the global flag
        self.assertEqual(filter_data.has_control_support, CONTROL_AVAILABLE)

    @cpu_and_gpu
    def test_discrete_delay_tf(self, target_device_idx, xp):
        """Test discrete delay transfer function implementation"""

        filter_data = IirFilterData.from_gain_and_ff([0.5], target_device_idx=target_device_idx)

        # Test integer delay
        delay = 3
        num, den = filter_data.discrete_delay_tf(delay)

        expected_den = np.zeros(4)
        expected_den[3] = 1.0
        expected_num = np.zeros(4)
        expected_num[0] = 1.0

        np.testing.assert_array_equal(num, expected_num)
        np.testing.assert_array_equal(den, expected_den)

        # Test fractional delay
        delay = 2.3
        num, den = filter_data.discrete_delay_tf(delay)

        # For delay = 2.3: d_m = ceil(2.3) = 3, so arrays have length 4
        expected_den = np.zeros(4)
        expected_den[3] = 1.0  # den[d_m] = 1
        expected_num = np.zeros(4)
        expected_num[0] = 0.3  # delay - fix(delay) = 2.3 - 2.0 = 0.3
        expected_num[1] = 0.7  # 1 - num[0] = 1 - 0.3 = 0.7

        np.testing.assert_allclose(num, expected_num, rtol=1e-12)
        np.testing.assert_array_equal(den, expected_den)

    @cpu_and_gpu
    def test_set_gain(self, target_device_idx, xp):
        """Test set_gain method functionality"""

        # Test 1: Set gain on gain+ff filters (simple case)
        gains = [0.1, 0.5, 1.0]
        ff = [0.9, 0.95, 0.99]
        filter_data = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # Original gains should match input
        original_gains = cpuArray(filter_data.gain)
        np.testing.assert_allclose(original_gains, gains, rtol=1e-12)

        # Set new gains
        new_gains = [0.2, 1.0, 0.5]
        filter_data.set_gain(new_gains)

        # Check that gains were updated
        updated_gains = cpuArray(filter_data.gain)
        np.testing.assert_allclose(updated_gains, new_gains, rtol=1e-12)

        # Check that numerator coefficients were scaled correctly
        expected_num = np.zeros((3, 2))
        for i in range(3):
            expected_num[i, 0] = 0
            expected_num[i, 1] = new_gains[i]

        actual_num = cpuArray(filter_data.num)
        np.testing.assert_allclose(actual_num, expected_num, rtol=1e-7)

        # Denominators should remain unchanged
        expected_den = np.zeros((3, 2))
        for i in range(3):
            expected_den[i, 0] = -ff[i]
            expected_den[i, 1] = 1.0

        actual_den = cpuArray(filter_data.den)
        np.testing.assert_allclose(actual_den, expected_den, rtol=1e-7)

    @cpu_and_gpu
    def test_poles_zeros_gains_roundtrip_simple(self, target_device_idx, xp):
        """Test roundtrip conversion with a simpler approach"""

        # Create a simple known filter
        gains = [0.1, 0.5]
        ff = [0.9, 0.95]
        original_filter = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # For gain+ff filters, we know the exact structure:
        # Poles should be at ff[i]
        # Zeros should be at 0
        # Gains should match input gains

        # Test poles extraction
        extracted_poles = cpuArray(original_filter.get_poles())
        expected_poles = np.array([[1/ff[0]], [1/ff[1]]])
        np.testing.assert_allclose(extracted_poles, expected_poles, rtol=1e-7, atol=1e-7,
                                 err_msg="Poles extraction failed")

        # Test zeros extraction (should be zeros at origin for gain+ff filters)
        extracted_zeros = cpuArray(original_filter.get_zeros())
        expected_zeros = np.array([[0.00], [0.0]])
        np.testing.assert_allclose(extracted_zeros, expected_zeros, rtol=1e-12, atol=1e-14,
                                 err_msg="Zeros extraction failed")

        # Test gains
        extracted_gains = cpuArray(original_filter.gain)
        np.testing.assert_allclose(extracted_gains, gains, rtol=1e-12, atol=1e-14,
                                 err_msg="Gains extraction failed")

    @cpu_and_gpu
    def test_rtf_ntf_output_shape(self, target_device_idx, xp):
        """Test RTF and NTF output shape and basic values"""
        gains = [0.5]
        ff = [1.0]
        filter_data = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)
        fs = 1000
        delay = 2
        freq = np.linspace(1/10, fs/2, 20*fs)
        dm = xp.array([0.0, 1.0], dtype=xp.float32)
        nw, dw = filter_data.discrete_delay_tf(delay - 1)
        rtf = filter_data.RTF(0, fs, freq=freq, nw=nw, dw=dw, dm=dm, plot=False)
        ntf = filter_data.NTF(0, fs, freq=freq, nw=nw, dw=dw, dm=dm, plot=False)
        self.assertEqual(rtf.shape, freq.shape)
        self.assertEqual(ntf.shape, freq.shape)
        self.assertTrue(np.all(np.isfinite(rtf)))
        self.assertTrue(np.all(np.isfinite(ntf)))

    @cpu_and_gpu
    def test_is_stable(self, target_device_idx, xp):
        """Test is_stable for a stable and unstable filter"""
        # Stable: pole inside unit circle
        gains = [0.3]
        ff = [1.0]
        stable_filter = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)
        # Plant TF
        delay = 3
        dm = xp.array([0.0, 1.0], dtype=xp.float32)
        nw, dw = stable_filter.discrete_delay_tf(delay - 1)
        self.assertTrue(stable_filter.is_stable(0, nw=nw, dw=dw, dm=dm))
        # Unstable: pole outside unit circle
        gains = [0.7]
        ff = [1.0]
        unstable_filter = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)
        self.assertFalse(unstable_filter.is_stable(0, nw=nw, dw=dw, dm=dm))

    @unittest.skipIf(not CONTROL_AVAILABLE, "Control library not available")
    @cpu_and_gpu
    def test_stability_margins(self, target_device_idx, xp):
        """Test stability margins for a simple filter"""
        # Use a clearly stable filter
        gains = [0.1]
        ff = [0.5]  # Well inside unit circle
        filter_data = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)
        gm, pm, wg, wp = filter_data.stability_margins(0, dt=1.0)  # Add dt parameter

        # For a stable system, margins should be finite and positive
        self.assertTrue(gm > 0 or np.isinf(gm))  # Gain margin can be infinite
        self.assertTrue(pm > 0)  # Phase margin should be positive
        
    @cpu_and_gpu
    def test_max_stable_gain_integrator_known_values(self, target_device_idx, xp):
        """Test maximum stable gain for integrator with known delay values"""

        # Create integrator filter (gain=1, ff=1 gives perfect integrator)
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)

        # Test delay = 2 frames -> max gain should be ~1.0
        max_gain_delay2 = integrator.max_stable_gain(mode=0, delay=2, n_gain=50000)
        self.assertAlmostEqual(max_gain_delay2, 1.0, places=2)

        # Test delay = 3 frames -> max gain should be ~0.618
        max_gain_delay3 = integrator.max_stable_gain(mode=0, delay=3, n_gain=50000)
        self.assertAlmostEqual(max_gain_delay3, 0.618, places=2)

        # Verify that max gain decreases with increasing delay
        max_gain_delay4 = integrator.max_stable_gain(mode=0, delay=4, n_gain=50000)
        self.assertLess(max_gain_delay4, max_gain_delay3)

    @cpu_and_gpu
    def test_max_stable_gain_all_modes(self, target_device_idx, xp):
        """Test maximum stable gain calculation for all modes"""

        # Create multiple identical integrators
        gains = [1.0, 1.0, 1.0]
        ff = [1.0, 1.0, 1.0]
        filters = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # Test with delay=3 for all modes
        max_gains = filters.max_stable_gain(delay=3, n_gain=20000)

        # All should be approximately 0.618
        for i, max_gain in enumerate(max_gains):
            self.assertAlmostEqual(max_gain, 0.618, places=1,
                                msg=f"Mode {i} max gain incorrect")

        # All should be identical (same filter)
        np.testing.assert_allclose(max_gains, max_gains[0], rtol=1e-10)

    @cpu_and_gpu
    def test_max_stable_gain_caching(self, target_device_idx, xp):
        """Test that caching works correctly for identical filters"""

        # Create filters with some identical and some different
        gains = [0.5, 1.0, 0.5, 1.0]  # Two pairs of identical filters
        ff = [0.9, 1.0, 0.9, 1.0]
        filters = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # Calculate twice
        max_gains_cached1 = filters.max_stable_gain(delay=3, n_gain=10000)
        max_gains_cached2 = filters.max_stable_gain(delay=3, n_gain=10000)

        # Results should be identical
        np.testing.assert_allclose(max_gains_cached1, max_gains_cached2, rtol=1e-12)

        # Identical filters should have identical results
        self.assertAlmostEqual(max_gains_cached1[0], max_gains_cached1[2], places=10)
        self.assertAlmostEqual(max_gains_cached2[1], max_gains_cached2[3], places=10)

    @cpu_and_gpu
    def test_resonance_frequency_delay_relationship(self, target_device_idx, xp):
        """Test that resonance frequency decreases with increasing delay"""

        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        fs = 1000.0  # Hz

        delays = [2, 3, 4, 5]
        resonance_freqs = []

        for delay in delays:
            # Use 80% of max stable gain
            max_gain = integrator.max_stable_gain(mode=0, delay=delay, n_gain=20000)
            gain_factor = max_gain * 0.8

            if gain_factor > 0:
                res_freq, res_amp = integrator.resonance_frequency(
                    mode=0, gain_factor=gain_factor, delay=delay, fs=fs
                )
                resonance_freqs.append(res_freq)
            else:
                resonance_freqs.append(0)

        # Verify decreasing trend
        for i in range(1, len(resonance_freqs)):
            if resonance_freqs[i-1] > 0 and resonance_freqs[i] > 0:
                self.assertLess(resonance_freqs[i], resonance_freqs[i-1], 
                            f"Resonance frequency should decrease with delay: "
                            f"delay={delays[i-1]} -> {resonance_freqs[i-1]:.1f} Hz, "
                            f"delay={delays[i]} -> {resonance_freqs[i]:.1f} Hz")

    @cpu_and_gpu
    def test_stability_analysis_comprehensive(self, target_device_idx, xp):
        """Test comprehensive stability analysis"""

        # Create integrator
        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)

        # Single mode analysis
        result = integrator.stability_analysis(mode=0, delay=3, fs=1000.0, n_gain=20000)

        # Check result structure
        self.assertIn('mode', result)
        self.assertIn('max_stable_gain', result)
        self.assertIn('resonance_frequency', result)
        self.assertIn('resonance_amplitude', result)
        self.assertIn('is_stable_at_max', result)

        # Check values
        self.assertEqual(result['mode'], 0)
        self.assertAlmostEqual(result['max_stable_gain'], 0.618, places=1)
        self.assertGreater(result['resonance_frequency'], 0)
        self.assertTrue(result['is_stable_at_max'])

    @cpu_and_gpu
    def test_stability_analysis_all_modes(self, target_device_idx, xp):
        """Test stability analysis for all modes"""

        gains = [0.5, 1.0]
        ff = [0.9, 1.0]
        filters = IirFilterData.from_gain_and_ff(gains, ff, target_device_idx=target_device_idx)

        # All modes analysis
        results = filters.stability_analysis(delay=3, fs=1000.0, n_gain=10000)

        # Should have results for both modes
        self.assertEqual(len(results), 2)

        # Check each result
        for i, result in enumerate(results):
            self.assertEqual(result['mode'], i)
            self.assertGreater(result['max_stable_gain'], 0)
            self.assertGreater(result['resonance_frequency'], 0)
            self.assertTrue(result['is_stable_at_max'])

    @cpu_and_gpu
    def test_max_stable_gain_with_plant_dynamics(self, target_device_idx, xp):
        """Test max stable gain with explicit plant dynamics"""

        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)

        # Create plant dynamics using discrete_delay_tf
        delay = 3
        dm = xp.array([0.0, 1.0])  # Pure delay in dm
        nw, dw = integrator.discrete_delay_tf(delay - 1)  # Additional delay

        # Test with explicit plant dynamics
        max_gain_plant = integrator.max_stable_gain(
            mode=0, dm=dm, nw=nw, dw=dw, n_gain=20000
        )

        # Test with equivalent delay
        max_gain_delay = integrator.max_stable_gain(mode=0, delay=delay, n_gain=20000)

        # Should be approximately equal
        self.assertAlmostEqual(max_gain_plant, max_gain_delay, places=1)

    @cpu_and_gpu
    def test_resonance_frequency_gain_scaling(self, target_device_idx, xp):
        """Test that resonance frequency changes with gain factor"""

        integrator = IirFilterData.from_gain_and_ff([1.0], [1.0], target_device_idx=target_device_idx)
        delay = 3
        fs = 1000.0

        # Test different gain factors
        gain_factors = [0.1, 0.3, 0.5]
        resonance_freqs = []

        for gain_factor in gain_factors:
            res_freq, res_amp = integrator.resonance_frequency(
                mode=0, gain_factor=gain_factor, delay=delay, fs=fs
            )
            resonance_freqs.append(res_freq)

        # Higher gain should generally give higher resonance frequency
        # (though this depends on the specific system)
        for i in range(len(resonance_freqs)):
            self.assertGreater(resonance_freqs[i], 0, f"Invalid resonance frequency for gain {gain_factors[i]}")

    @cpu_and_gpu
    def test_discrete_delay_tf_implementation(self, target_device_idx, xp):
        """Test discrete delay transfer function implementation in detail"""

        filter_data = IirFilterData.from_gain_and_ff([1.0], target_device_idx=target_device_idx)

        # Test integer delays
        for delay in [1, 2, 3, 4]:
            num, den = filter_data.discrete_delay_tf(delay)

            # For integer delay: num = [1, 0, 0, ..., 0], den = [0, 0, ..., 0, 1]
            expected_num = np.zeros(delay + 1)
            expected_num[0] = 1.0
            expected_den = np.zeros(delay + 1)
            expected_den[-1] = 1.0

            np.testing.assert_array_almost_equal(num, expected_num)
            np.testing.assert_array_almost_equal(den, expected_den)

        # Test fractional delays
        delay = 2.3
        num, den = filter_data.discrete_delay_tf(delay)

        # For delay = 2.3: ceil(2.3) = 3, so length = 4
        # num[0] = 2.3 - 2.0 = 0.3, num[1] = 1 - 0.3 = 0.7
        expected_num = np.array([0.3, 0.7, 0.0, 0.0])
        expected_den = np.array([0.0, 0.0, 0.0, 1.0])

        np.testing.assert_array_almost_equal(num, expected_num)
        np.testing.assert_array_almost_equal(den, expected_den)

    def test_stability_methods_consistency(self):
        """Test consistency between different stability analysis methods"""

        # Create test filter
        filter_data = IirFilterData.from_gain_and_ff([0.1], [1.0])
        delay = 3

        # Test is_stable at different gain levels
        dm = np.array([1.0])
        nw, dw = filter_data.discrete_delay_tf(delay)

        # Should be stable at low gain
        is_stable_low = filter_data.is_stable(0, dm=dm, nw=nw, dw=dw)
        self.assertTrue(is_stable_low)

        # Find max stable gain
        max_gain = filter_data.max_stable_gain(mode=0, delay=delay, n_gain=10000)

        # Should be stable at 95% of max gain
        filter_data.set_gain([max_gain * 0.95])
        is_stable_near_max = filter_data.is_stable(0, dm=dm, nw=nw, dw=dw)
        self.assertTrue(is_stable_near_max)

        # Should be unstable at 110% of max gain (if max_gain < max possible gain)
        if max_gain < 10.0:  # Only test if we found a reasonable max gain
            filter_data.set_gain([max_gain * 1.1])
            is_stable_over_max = filter_data.is_stable(0, dm=dm, nw=nw, dw=dw)
            self.assertFalse(is_stable_over_max)

# --- debugging utility for filter tests ---

    def _debug_plot_filter(self, filter_data, test_name, target_device_idx, max_modes=3):
        """Helper function to create debug plots for filter tests"""

        import matplotlib.pyplot as plt

        fs = 1000
        freq = np.logspace(-1, 2, 1000)
        nmodes = min(max_modes, filter_data.nfilter)

        plt.figure(figsize=(15, 10))

        # Magnitude response
        plt.subplot(2, 3, 1)
        for mode in range(nmodes):
            response = filter_data.frequency_response(
                filter_data.num[mode, :], filter_data.den[mode, :], fs,
                freq=freq
            )
            plt.loglog(freq, np.abs(response), label=f'Mode {mode}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('Magnitude Response')
        plt.grid(True)
        plt.legend()

        # Phase response
        plt.subplot(2, 3, 2)
        for mode in range(nmodes):
            response = filter_data.plot_iirfilter_tf(
                filter_data.num[mode, :], filter_data.den[mode, :], fs,
                freq=freq
            )
            plt.semilogx(freq, np.angle(response) * 180/np.pi, label=f'Mode {mode}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (degrees)')
        plt.title('Phase Response')
        plt.grid(True)
        plt.legend()

        # Pole-zero plot
        plt.subplot(2, 3, 3)
        for mode in range(nmodes):
            poles = cpuArray(filter_data.get_poles())[mode, :]
            zeros = cpuArray(filter_data.get_zeros())[mode, :]
            plt.plot(np.real(poles), np.imag(poles), 'x', markersize=8)
            plt.plot(np.real(zeros), np.imag(zeros), 'o', markersize=6)

        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Pole-Zero Plot')
        plt.grid(True)
        plt.axis('equal')

        # Numerator coefficients
        plt.subplot(2, 3, 4)
        num_coeffs = cpuArray(filter_data.num)[:nmodes, :]
        for i in range(num_coeffs.shape[1]):
            plt.bar(np.arange(nmodes) + i*0.3, num_coeffs[:, i],
                    width=0.25, label=f'num[{i}]', alpha=0.7)
        plt.xlabel('Mode')
        plt.ylabel('Coefficient')
        plt.title('Numerator Coefficients')
        plt.legend()
        plt.grid(True)

        # Denominator coefficients
        plt.subplot(2, 3, 5)
        den_coeffs = cpuArray(filter_data.den)[:nmodes, :]
        for i in range(den_coeffs.shape[1]):
            plt.bar(np.arange(nmodes) + i*0.3, den_coeffs[:, i],
                    width=0.25, label=f'den[{i}]', alpha=0.7)
        plt.xlabel('Mode')
        plt.ylabel('Coefficient')
        plt.title('Denominator Coefficients')
        plt.legend()
        plt.grid(True)

        # Impulse response
        plt.subplot(2, 3, 6)
        n_samples = 50
        for mode in range(nmodes):
            # Simple impulse response calculation
            impulse = np.zeros(n_samples)
            impulse[0] = 1

            # Apply filter (simplified)
            num = cpuArray(filter_data.num[mode, :])
            den = cpuArray(filter_data.den[mode, :])

            # Remove leading zeros
            num = num[num != 0] if np.any(num != 0) else np.array([0])
            den = den[den != 0] if np.any(den != 0) else np.array([1])

            # Use scipy.signal if available
            try:
                from scipy import signal
                _, impulse_resp = signal.dimpulse((num, den, 1), n=n_samples)
                plt.plot(impulse_resp[0].flatten(), label=f'Mode {mode}')
            except ImportError:
                # Fallback: just show the gain
                plt.axhline(y=cpuArray(filter_data.gain)[mode], 
                           label=f'Mode {mode} (gain only)')

        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('Impulse Response')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.show()
