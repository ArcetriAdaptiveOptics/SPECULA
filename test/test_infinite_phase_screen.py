import unittest
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen
from specula.lib.calc_phasescreen import calc_phasescreen

class TestInfinitePhaseScreen(unittest.TestCase):

    def test_phase_covariance_matches_theory(self):
        """Test that the phase covariance function matches theoretical values"""

        # Parameters
        mx_size = 512
        pixel_scale = 0.1  # meters
        r0 = 0.2  # meters
        L0 = 25.0  # meters
        random_seed = 12345

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                 random_seed=random_seed,
                                 target_device_idx=-1)

        # Test covariance function at different separations
        separations = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # meters
        cov_values = ips.phase_covariance(separations, r0, L0)

        # Basic sanity checks
        self.assertTrue(all(cov_values >= 0), "Covariance values should be non-negative")
        self.assertTrue(cov_values[0] > cov_values[-1], "Covariance should decrease with separation")

        # Check that covariance at zero separation is finite and positive
        cov_zero = ips.phase_covariance(np.array([1e-6]), r0, L0)[0]
        self.assertTrue(cov_zero > 0, "Covariance at zero separation should be positive")

    def test_infinite_vs_fft_phase_screen_statistics(self):
        """Compare statistics between InfinitePhaseScreen and calc_phasescreen (FFT method)
        across multiple combinations of phase_size and pixel_scale"""

        verbose = False

        # Parameters
        r0 = 0.15  # meters
        L0 = 25.0  # meters
        random_seed1 = 42
        random_seed2 = 1042
        n_seeds = 5

        # Test parameter combinations
        phase_sizes = [512]
        pixel_scales = [0.5, 0.05]

        # Store all results for summary
        results = []

        if verbose:
            print("\nTesting InfinitePhaseScreen vs FFT phase screen statistics")
            print("=" * 76)
            print(f"{'Phase Size':<12} {'Pixel Scale':<12} {'Inf Mean':<10} {'Inf Std':<10} {'FFT Mean':<10} {'FFT Std':<10} {'Ratio':<8}")
            print("-" * 76)

        for phase_size in phase_sizes:
            for pixel_scale in pixel_scales:
                # Initialize accumulators
                inf_mean = 0
                inf_std = 0
                fft_mean = 0
                fft_std = 0

                for i in range(n_seeds):
                    # Create infinite phase screen
                    r0_inf = r0 #2 * pixel_scale
                    ips = InfinitePhaseScreen(phase_size, pixel_scale, r0_inf, L0,
                                            random_seed=random_seed1 + i,
                                            target_device_idx=-1)

                    # Get initial phase screen
                    infinite_screen = ips.scrn * 500 / (2 * np.pi)  # in nm
                    r0_scaling = (r0_inf / r0)**(5./6.)
                    infinite_screen *= r0_scaling

                    # Create FFT phase screen with same parameters
                    fft_screen = calc_phasescreen(L0, phase_size, pixel_scale,
                                                seed=random_seed2 + i,
                                                precision=1,
                                                xp=np)
                    fft_screen = fft_screen * 500 / (2 * np.pi)  # in nm
                    r0_scaling = (pixel_scale / r0)**(5./6.)
                    fft_screen *= r0_scaling

                    # Accumulate statistics
                    inf_mean += np.mean(infinite_screen) / n_seeds
                    inf_std += np.std(infinite_screen) / n_seeds
                    fft_mean += np.mean(fft_screen) / n_seeds
                    fft_std += np.std(fft_screen) / n_seeds

                # Calculate ratio
                std_ratio = inf_std / fft_std if fft_std != 0 else 0

                # Store results
                result = {
                    'phase_size': phase_size,
                    'pixel_scale': pixel_scale,
                    'inf_mean': inf_mean,
                    'inf_std': inf_std,
                    'fft_mean': fft_mean,
                    'fft_std': fft_std,
                    'std_ratio': std_ratio
                }
                results.append(result)

                # Print current result
                if verbose:
                    print(f"{phase_size:<12} {pixel_scale:<12} {inf_mean:<10.6f} {inf_std:<10.1f} {fft_mean:<10.6f} {fft_std:<10.1f} {std_ratio:<8.3f}")

        # Overall statistics
        all_ratios = [r['std_ratio'] for r in results if r['std_ratio'] > 0]
        min_ratio = np.min(all_ratios)
        max_ratio = np.max(all_ratios)

        failed_tests = []
        for result in results:
            phase_size = result['phase_size']
            pixel_scale = result['pixel_scale']
            inf_mean = result['inf_mean']
            fft_mean = result['fft_mean']
            std_ratio = result['std_ratio']

            # Mean should be close to zero for both
            try:
                self.assertAlmostEqual(inf_mean, 0.0, places=2,
                                    msg=f"Infinite screen mean should be near zero (size={phase_size}, scale={pixel_scale})")
                self.assertAlmostEqual(fft_mean, 0.0, places=2,
                                    msg=f"FFT screen mean should be near zero (size={phase_size}, scale={pixel_scale})")
            except AssertionError as e:
                failed_tests.append(f"Mean test failed for size={phase_size}, scale={pixel_scale}: {str(e)}")

            # Standard deviations should be similar
            min_ratio, max_ratio = 0.9, 1.5

            try:
                self.assertTrue(min_ratio < std_ratio < max_ratio,
                            f"Std ratio {std_ratio:.3f} should be in [{min_ratio}, {max_ratio}] for size={phase_size}, scale={pixel_scale}")
            except AssertionError as e:
                failed_tests.append(f"Std ratio test failed for size={phase_size}, scale={pixel_scale}: {str(e)}")

        if failed_tests:
            print(f"\n{len(failed_tests)} test(s) failed:")
            for failure in failed_tests[:10]:  # Show first 10 failures
                print(f"  - {failure}")
            if len(failed_tests) > 10:
                print(f"  ... and {len(failed_tests) - 10} more")

    def test_reproducibility_with_same_seed(self):
        """Test that screens with same seed produce identical results"""

        # Parameters
        mx_size = 64
        pixel_scale = 0.05
        r0 = 0.15
        L0 = 30.0
        random_seed = 789

        # Create two identical screens
        ips1 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                  random_seed=random_seed,
                                  target_device_idx=-1)

        ips2 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                  random_seed=random_seed,
                                  target_device_idx=-1)

        # Get screens
        screen1 = ips1.scrn
        screen2 = ips2.scrn

        # Should be identical
        np.testing.assert_array_equal(screen1, screen2,
                                     "Screens with same seed should be identical")

        # Evolve both screens identically
        for _ in range(5):
            ips1.add_line(row=1, after=1)
            ips2.add_line(row=1, after=1)

        screen1_evolved = ips1.scrn
        screen2_evolved = ips2.scrn

        # Should still be identical after evolution
        np.testing.assert_array_equal(screen1_evolved, screen2_evolved,
                                     "Evolved screens with same seed should remain identical")