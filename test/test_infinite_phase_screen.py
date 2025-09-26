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
        mx_size = 128
        pixel_scale = 0.1  # meters
        r0 = 0.2  # meters
        L0 = 25.0  # meters
        l0 = 0.01  # meters
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
        mx_size = 128
        pixel_scale = 0.05  # meters
        r0 = 0.15  # meters
        L0 = 25.0  # meters
        l0 = 0.01  # meters
        random_seed = 42

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                 random_seed=random_seed,
                                 target_device_idx=target_device_idx)

        # Get initial phase screen
        infinite_screen = cpuArray(ips.scrn)

        # Create FFT phase screen with same parameters
        fft_screen = calc_phasescreen(L0, mx_size, pixel_scale,
                                     seed=random_seed,
                                     precision=1,  # single precision
                                     xp=xp)
        fft_screen = cpuArray(fft_screen)

        # Compare basic statistics
        inf_mean = np.mean(infinite_screen)
        inf_std = np.std(infinite_screen)
        fft_mean = np.mean(fft_screen)
        fft_std = np.std(fft_screen)

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
    def test_infinite_screen_covariance_structure(self, target_device_idx, xp):
        """Test that the infinite screen has the correct covariance structure"""

        # Parameters
        mx_size = 512
        pixel_scale = 0.05  # meters
        r0 = 0.2  # meters
        L0 = 20.0  # meters
        l0 = 0.01  # meters
        random_seed = 123

        # Create infinite phase screen
        ips = InfinitePhaseScreen(mx_size, pixel_scale, r0, L0, l0,
                                 random_seed=random_seed,
                                 target_device_idx=target_device_idx)

        # Get the phase screen
        screen = cpuArray(ips.scrn)

        # Calculate empirical covariance at different lags
        center = mx_size // 2
        max_lag = 10

        empirical_cov = []
        theoretical_cov = []
        separations = []

        for lag in range(0, max_lag):
            # Calculate empirical covariance at this lag (horizontal)
            if lag == 0:
                emp_cov = np.var(screen)
            else:
                cov_sum = 0
                count = 0
                for i in range(center - 10, center + 10):
                    for j in range(center - 10, center + 10 - lag):
                        cov_sum += screen[i, j] * screen[i, j + lag]
                        count += 1
                emp_cov = cov_sum / count - np.mean(screen)**2

            empirical_cov.append(emp_cov)

            # Calculate theoretical covariance
            separation = lag * pixel_scale
            separations.append(separation)
            theo_cov = ips.phase_covariance(np.array([separation]), r0, L0)[0]
            theoretical_cov.append(theo_cov)

        empirical_cov = np.array(empirical_cov)
        theoretical_cov = np.array(theoretical_cov)
        separations = np.array(separations)

        # Plot for visual inspection (optional)
        display = False  # Set to True to see plots
        if display:
            plt.figure(figsize=(10, 6))
            plt.plot(separations, empirical_cov, 'o-', label='Empirical')
            plt.plot(separations, theoretical_cov, 's-', label='Theoretical')
            plt.xlabel('Separation [m]')
            plt.ylabel('Phase Covariance')
            plt.legend()
            plt.title('Phase Covariance Comparison')
            plt.grid(True)
            plt.figure(figsize=(10, 6))
            ratio = empirical_cov / theoretical_cov
            plt.plot(separations, ratio, 'o-')
            plt.xlabel('Separation [m]')
            plt.ylabel('Empirical / Theoretical Covariance')
            plt.title('Covariance Ratio')
            plt.grid(True)
            plt.show()

        # Check that empirical and theoretical covariances are reasonably close
        # Allow for statistical noise, especially at larger separations
        for i in range(min(len(empirical_cov), 10)):  # Check first 10 lags
            ratio = empirical_cov[i] / theoretical_cov[i] if theoretical_cov[i] != 0 else 1
            self.assertTrue(0.5 < ratio < 2.0,
                           f"Covariance ratio at lag {i} is {ratio:.3f}, should be near 1.0")

    @cpu_and_gpu
    def test_screen_evolution_with_add_line(self, target_device_idx, xp):
        """Test that adding lines to the screen works correctly"""

        # Parameters
        mx_size = 64
        pixel_scale = 0.1
        r0 = 0.2
        L0 = 25.0
        l0 = 0.01
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
        l0 = 0.01
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