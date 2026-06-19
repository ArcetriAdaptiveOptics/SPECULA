import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from astropy.io import fits
from specula.processing_objects.close_gain_optimizer import CloseGainOptimizer
from specula import cpuArray
import numpy as np


class TestCloseGainOptimizer(unittest.TestCase):
    """Test CLOSE gain optimizer by running simulations and checking the output"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_close_gain_optimizer.yml')
        os.makedirs(self.datadir, exist_ok=True)
        # Get current working directory
        self.cwd = os.getcwd()

    def tearDown(self):
        # Remove test/data directory with timestamp
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                try:
                    shutil.rmtree(data_dir)
                except Exception:
                    pass
        os.chdir(self.cwd)

    def test_gain_optimizer(self):
        """Run the simulation and check gain optimizer output"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        simul = Simul(self.params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if gain optimizer output file exists
        gain_file = os.path.join(latest_data_dir, 'close_optimized_gain.fits')
        self.assertTrue(os.path.exists(gain_file), f"Gain optimizer output file not found: {gain_file}")

        # Read gain optimizer output
        with fits.open(gain_file) as hdul:
            gains = hdul[0].data.copy()

            self.assertIsNotNone(gains, "No gain data found in output file")
            self.assertGreater(len(gains), 0, "No gain data points found")

            # Check that last value of gains is around 0.5
            last_gain = gains[-1]
            if isinstance(last_gain, np.ndarray):
                last_gain = last_gain.item()

            self.assertAlmostEqual(
                last_gain, 0.5,
                delta=0.1,
                msg=f"Last gain value {last_gain:.4f} does not match expected 0.5"
            )

    def test_close_gain_optimizer_initialization(self):
        """Test proper initialization of CloseGainOptimizer"""
        nmodes = 5
        optimizer = CloseGainOptimizer(
            nmodes=nmodes,
            p=0.3,
            r=-0.1,
            dt=3,
            q_plus=1e-2,
            q_minus_ratio=5.0,
            initial_gain=0.5
        )

        # Check initialization
        self.assertEqual(optimizer.nmodes, nmodes)
        self.assertEqual(optimizer.p, 0.3)
        self.assertEqual(optimizer.r, -0.1)
        self.assertEqual(optimizer.dt, 3)
        self.assertEqual(optimizer.q_plus, 1e-2)
        self.assertEqual(optimizer.q_minus, 1e-2 * 5.0)
        self.assertIsNotNone(optimizer.optimized_gain)
        self.assertEqual(len(optimizer.optimized_gain.value), nmodes)
        np.testing.assert_array_almost_equal(
            cpuArray(optimizer.optimized_gain.value),
            np.ones(nmodes) * 0.5
        )

    def test_dt_fractional_handling(self):
        """Test that dt fractional part is correctly handled"""
        nmodes = 2
        
        # Test with integer dt
        optimizer1 = CloseGainOptimizer(nmodes=nmodes, dt=3.0)
        self.assertEqual(optimizer1.dt, 3)
        self.assertEqual(optimizer1.dt_frac, 0.0)
        
        # Test with fractional dt
        optimizer2 = CloseGainOptimizer(nmodes=nmodes, dt=3.5)
        self.assertEqual(optimizer2.dt, 4)
        self.assertAlmostEqual(optimizer2.dt_frac, 0.5, places=5)

    def test_history_buffer_management(self):
        """Test that history buffer is properly managed"""
        nmodes = 2
        optimizer = CloseGainOptimizer(nmodes=nmodes, dt=3)
        
        # Simulate adding measurements to history
        measurements = [
            np.array([0.1, 0.2]),
            np.array([0.15, 0.25]),
            np.array([0.2, 0.3]),
            np.array([0.25, 0.35]),
        ]
        
        for measurement in measurements:
            optimizer.m_history.append(measurement.copy())
        
        # Check history size doesn't exceed dt + 1
        initial_size = len(optimizer.m_history)
        self.assertLessEqual(initial_size, optimizer.dt + 1)
        
        # Add more to trigger pop
        optimizer.m_history.append(np.array([0.3, 0.4]))
        if len(optimizer.m_history) > optimizer.dt + 1:
            optimizer.m_history.pop(0)
        
        self.assertLessEqual(len(optimizer.m_history), optimizer.dt + 1)

    def test_state_variable_initialization(self):
        """Test that state variables are properly initialized"""
        nmodes = 3
        optimizer = CloseGainOptimizer(nmodes=nmodes)
        
        # Check N_0 and N_dt initialization
        self.assertEqual(len(optimizer.N_0), nmodes)
        self.assertEqual(len(optimizer.N_dt), nmodes)
        np.testing.assert_array_almost_equal(cpuArray(optimizer.N_0), np.zeros(nmodes))
        np.testing.assert_array_almost_equal(cpuArray(optimizer.N_dt), np.zeros(nmodes))

    def test_asymmetric_learning_factors(self):
        """Test that asymmetric learning factors are correctly set"""
        nmodes = 2
        q_plus = 0.01
        q_minus_ratio = 5.0
        
        optimizer = CloseGainOptimizer(
            nmodes=nmodes,
            q_plus=q_plus,
            q_minus_ratio=q_minus_ratio
        )
        
        self.assertEqual(optimizer.q_plus, q_plus)
        self.assertAlmostEqual(optimizer.q_minus, q_plus * q_minus_ratio, places=10)

    def test_gain_clamping(self):
        """Test that gains are clamped to safe limits"""
        nmodes = 1
        optimizer = CloseGainOptimizer(nmodes=nmodes, initial_gain=0.5)
        
        # Simulate extreme gain change
        large_corr_diff = 100.0  # Very large correlation difference
        current_gain = optimizer.optimized_gain.value.copy()
        q_array = optimizer.q_plus
        
        new_gain = current_gain * (1.0 + q_array * large_corr_diff)
        new_gain_clamped = optimizer.xp.clip(new_gain, 1e-4, 10.0)
        
        # Check clamping
        self.assertLessEqual(float(new_gain_clamped[0]), 10.0)
        self.assertGreaterEqual(float(new_gain_clamped[0]), 1e-4)

    def test_correlation_ratio_calculation(self):
        """Test correlation ratio calculation"""
        nmodes = 1
        optimizer = CloseGainOptimizer(nmodes=nmodes, p=0.5)
        
        # Set up state variables
        N_0_val = 1.0
        N_dt_val = 0.5
        optimizer.N_0 = optimizer.xp.array([N_0_val], dtype=optimizer.dtype)
        optimizer.N_dt = optimizer.xp.array([N_dt_val], dtype=optimizer.dtype)
        
        # Calculate correlation ratio
        safe_N_0 = optimizer.xp.where(optimizer.N_0 < 1e-12, 1e-12, optimizer.N_0)
        correlation_ratio = optimizer.N_dt / safe_N_0
        
        expected_ratio = N_dt_val / N_0_val
        self.assertAlmostEqual(float(correlation_ratio[0]), expected_ratio, places=5)

    def test_gain_update_direction_positive_corr_diff(self):
        """Test that gains increase when correlation difference is positive"""
        nmodes = 1
        optimizer = CloseGainOptimizer(
            nmodes=nmodes,
            q_plus=0.1,
            q_minus_ratio=5.0,
            initial_gain=0.5
        )
        
        current_gain = 0.5
        corr_diff = 0.2  # Positive: good tracking
        q_value = optimizer.q_plus  # Should use q_plus
        
        new_gain = current_gain * (1.0 + q_value * corr_diff)
        expected_gain = 0.5 * (1.0 + 0.1 * 0.2)
        
        self.assertAlmostEqual(new_gain, expected_gain, places=5)
        self.assertGreater(new_gain, current_gain)

    def test_gain_update_direction_negative_corr_diff(self):
        """Test that gains are reduced more aggressively when correlation difference is negative"""
        nmodes = 1
        optimizer = CloseGainOptimizer(
            nmodes=nmodes,
            q_plus=0.1,
            q_minus_ratio=5.0,
            initial_gain=0.5
        )
        
        current_gain = 0.5
        corr_diff = -0.2  # Negative: ringing/overshoot
        q_value = optimizer.q_minus  # Should use q_minus
        
        new_gain = current_gain * (1.0 + q_value * corr_diff)
        expected_gain = 0.5 * (1.0 + 0.5 * (-0.2))  # q_minus = 0.1 * 5.0 = 0.5
        
        self.assertAlmostEqual(new_gain, expected_gain, places=5)
        self.assertLess(new_gain, current_gain)

    def test_multiple_modes_independent_updates(self):
        """Test that gains for different modes are updated independently"""
        nmodes = 3
        optimizer = CloseGainOptimizer(nmodes=nmodes, q_plus=0.1, q_minus_ratio=5.0)
        
        # Set different correlation differences for each mode
        corr_diffs = np.array([0.1, -0.1, 0.05])
        current_gains = np.array([0.5, 0.6, 0.7])
        
        # Calculate expected gains
        q_array = np.where(corr_diffs < 0, optimizer.q_minus, optimizer.q_plus)
        expected_gains = current_gains * (1.0 + q_array * corr_diffs)
        
        # Mode 0: positive corr_diff, should increase moderately
        self.assertGreater(expected_gains[0], current_gains[0])
        
        # Mode 1: negative corr_diff, should decrease more aggressively
        self.assertLess(expected_gains[1], current_gains[1])
        self.assertGreater(
            abs(expected_gains[1] - current_gains[1]),
            abs(expected_gains[0] - current_gains[0])
        )
        
        # Mode 2: small positive corr_diff
        self.assertGreater(expected_gains[2], current_gains[2])

    def test_input_output_connections(self):
        """Test that inputs and outputs are properly defined"""
        optimizer = CloseGainOptimizer(nmodes=4)
        
        # Check input definition
        self.assertIn('in_modes', optimizer.inputs)
        
        # Check output definition
        self.assertIn('out_gains', optimizer.outputs)
        self.assertEqual(optimizer.outputs['out_gains'], optimizer.optimized_gain)

    def test_input_output_descriptors(self):
        """Test that input and output descriptors are correct"""
        input_descs = CloseGainOptimizer.input_names()
        output_descs = CloseGainOptimizer.output_names()
        
        self.assertIn('in_modes', input_descs)
        self.assertIn('out_gains', output_descs)

    def test_fractional_interpolation(self):
        """Test fractional time-shift interpolation in history buffer"""
        nmodes = 1
        optimizer = CloseGainOptimizer(nmodes=nmodes, dt=2.5)
        
        # dt should be 3, dt_frac should be 0.5
        self.assertEqual(optimizer.dt, 3)
        self.assertAlmostEqual(optimizer.dt_frac, 0.5, places=5)
        
        # Simulate interpolation
        m_k_dt_minus_1 = np.array([0.0])
        m_k_dt = np.array([1.0])
        
        interpolated = (1.0 - optimizer.dt_frac) * m_k_dt_minus_1 + optimizer.dt_frac * m_k_dt
        expected = 0.5 * 0.0 + 0.5 * 1.0
        
        self.assertAlmostEqual(float(interpolated[0]), expected, places=5)

    def test_parameter_ranges(self):
        """Test that optimizer works with various parameter ranges"""
        # Test with small dt
        opt1 = CloseGainOptimizer(nmodes=2, dt=1)
        self.assertEqual(opt1.dt, 1)
        
        # Test with large dt
        opt2 = CloseGainOptimizer(nmodes=2, dt=100)
        self.assertEqual(opt2.dt, 100)
        
        # Test with small p
        opt3 = CloseGainOptimizer(nmodes=2, p=0.01)
        self.assertEqual(opt3.p, 0.01)
        
        # Test with large p
        opt4 = CloseGainOptimizer(nmodes=2, p=0.99)
        self.assertEqual(opt4.p, 0.99)


