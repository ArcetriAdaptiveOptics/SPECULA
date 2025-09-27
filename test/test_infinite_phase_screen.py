import unittest
import os
import specula
specula.init(0)  # Default target device

from specula import np, cpuArray
from specula.data_objects.infinite_phase_screen import InfinitePhaseScreen
from specula.lib.calc_phasescreen import calc_phasescreen
from test.specula_testlib import cpu_and_gpu

import matplotlib.pyplot as plt


class TestInfinitePhaseScreen(unittest.TestCase):

    @cpu_and_gpu
    def test_phase_covariance_matches_theory(self, target_device_idx, xp):
        """Test that the phase covariance function matches theoretical values"""

        # Parameters
        mx_size = 512
        pixel_scale = 0.1  # meters
        r0 = 0.2  # meters
        L0 = 25.0  # meters
        l0 = 0.005  # meters
        random_seed = 12345

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                 random_seed=random_seed,
                                 target_device_idx=target_device_idx)

        # Test covariance function at different separations
        separations = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])  # meters
        cov_values = ips.phase_covariance(separations, r0, L0)

        # Basic sanity checks
        self.assertTrue(all(cov_values >= 0), "Covariance values should be non-negative")
        self.assertTrue(cov_values[0] > cov_values[-1], "Covariance should decrease with separation")

        # Check that covariance at zero separation is finite and positive
        cov_zero = ips.phase_covariance(np.array([1e-6]), r0, L0)[0]
        self.assertTrue(cov_zero > 0, "Covariance at zero separation should be positive")

    @cpu_and_gpu
    def test_infinite_vs_fft_phase_screen_statistics(self, target_device_idx, xp):
        """Compare statistics between InfinitePhaseScreen and calc_phasescreen (FFT method)"""

        # Parameters
        mx_size = 150
        pixel_scale = 0.05 # meters
        r0 = 0.15  # meters
        L0 = 25.0  # meters
        random_seed1 = 42
        random_seed2 = 1042
        
        n_seeds = 200
        
        inf_mean = 0
        inf_std = 0
        fft_mean = 0
        fft_std = 0

        for i in range(n_seeds):
            random_seed1 += i
            random_seed2 += i
            # Create infinite phase screen
            ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0,
                                    random_seed=random_seed1,
                                    target_device_idx=target_device_idx)

            # Get initial phase screen
            infinite_screen = cpuArray(ips.scrn) * 500 / (2 * np.pi)

            # Create FFT phase screen with same parameters
            fft_screen = calc_phasescreen(L0, mx_size, pixel_scale,
                                        seed=random_seed2,
                                        precision=1,  # single precision
                                        xp=xp)
            fft_screen = cpuArray(fft_screen)
            r0_scaling = (pixel_scale / r0)**(5./6.)
            fft_screen *= r0_scaling

            # Compare basic statistics
            inf_mean_i = np.mean(infinite_screen)
            inf_std_i = np.std(infinite_screen)
            fft_mean_i = np.mean(fft_screen)
            fft_std_i = np.std(fft_screen)
            
            inf_mean += inf_mean_i/n_seeds
            inf_std += inf_std_i/n_seeds
            fft_mean += fft_mean_i/n_seeds
            fft_std += fft_std_i/n_seeds

        print(f"Infinite screen - Mean: {inf_mean:.6f}, Std: {inf_std:.6f}")
        print(f"FFT screen - Mean: {fft_mean:.6f}, Std: {fft_std:.6f}")

        # Mean should be close to zero for both
        self.assertAlmostEqual(inf_mean, 0.0, places=2,
                              msg="Infinite screen mean should be near zero")
        self.assertAlmostEqual(fft_mean, 0.0, places=2,
                              msg="FFT screen mean should be near zero")

        # Standard deviations should be similar (within 20%)
        std_ratio = inf_std / fft_std
        self.assertTrue(0.8 < std_ratio < 1.2,
                       f"Standard deviation ratio {std_ratio:.3f} should be near 1.0")

    @cpu_and_gpu
    def test_screen_evolution_with_add_line(self, target_device_idx, xp):
        """Test that adding lines to the screen works correctly"""

        # Parameters
        mx_size = 64
        pixel_scale = 0.1
        r0 = 0.2
        L0 = 25.0
        l0 = 0.005
        random_seed = 456

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                 random_seed=random_seed,
                                 target_device_idx=target_device_idx)

        # Get initial screen
        initial_screen = cpuArray(ips.scrn.copy())

        # Add a line (simulate wind evolution)
        ips.add_line(row=1, after=0)  # Add row at the end
        evolved_screen = cpuArray(ips.scrn.copy())

        # Check that screen dimensions are maintained
        self.assertEqual(initial_screen.shape, evolved_screen.shape,
                        "Screen dimensions should remain constant after adding line")

        # Check that the screen has actually changed
        diff = np.mean(np.abs(initial_screen - evolved_screen))
        self.assertTrue(diff > 0, "Screen should change after adding a line")

        # Add multiple lines and check statistics remain reasonable
        for _ in range(5):
            ips.add_line(row=1, after=1)
            ips.add_line(row=0, after=1)

        final_screen = cpuArray(ips.scrn)
        final_std = np.std(final_screen)
        initial_std = np.std(initial_screen)

        # Standard deviation should remain in reasonable range
        std_ratio = final_std / initial_std
        self.assertTrue(0.5 < std_ratio < 2.0,
                       f"Standard deviation ratio {std_ratio:.3f} after evolution should be reasonable")

    @cpu_and_gpu
    def test_reproducibility_with_same_seed(self, target_device_idx, xp):
        """Test that screens with same seed produce identical results"""

        # Parameters
        mx_size = 64
        pixel_scale = 0.05
        r0 = 0.15
        L0 = 30.0
        l0 = 0.005
        random_seed = 789

        # Create two identical screens
        ips1 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                  random_seed=random_seed,
                                  target_device_idx=target_device_idx)

        ips2 = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                  random_seed=random_seed,
                                  target_device_idx=target_device_idx)

        # Get screens
        screen1 = cpuArray(ips1.scrn)
        screen2 = cpuArray(ips2.scrn)

        # Should be identical
        np.testing.assert_array_equal(screen1, screen2,
                                     "Screens with same seed should be identical")

        # Evolve both screens identically
        for _ in range(5):
            ips1.add_line(row=1, after=1)
            ips2.add_line(row=1, after=1)

        screen1_evolved = cpuArray(ips1.scrn)
        screen2_evolved = cpuArray(ips2.scrn)

        # Should still be identical after evolution
        np.testing.assert_array_equal(screen1_evolved, screen2_evolved,
                                     "Evolved screens with same seed should remain identical")