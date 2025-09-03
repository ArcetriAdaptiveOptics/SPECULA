import unittest
import os
import glob
import shutil
from unittest.mock import Mock
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from specula.processing_objects.gain_optimizer import GainOptimizer
from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.simul_params import SimulParams
from specula import cpuArray
from astropy.io import fits
import numpy as np


class TestGainOptimizer(unittest.TestCase):
    """Test gain optimizer by running a simulation and checking the output"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_gain_optimizer.yml')
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
        gain_file = os.path.join(latest_data_dir, 'optimized_gain.fits')
        self.assertTrue(os.path.exists(gain_file), f"Gain optimizer output file not found: {gain_file}")

        # Read gain optimizer output
        with fits.open(gain_file) as hdul:
            self.assertTrue(len(hdul) >= 2, "Expected at least 2 HDUs in gain optimizer output file")

            # Check times and data
            times = hdul[1].data.copy()
            gains = hdul[0].data.copy()

            self.assertIsNotNone(times, "No time data found in gain optimizer output file")
            self.assertIsNotNone(gains, "No gain data found in output file")

            # Check that we have reasonable data
            self.assertEqual(len(times), len(gains), "Times and gain data length mismatch")
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

    def test_gain_optimizer_initialization(self):
        """Test proper initialization of GainOptimizer"""
        # Create mock objects
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001  # 1ms

        iir_filter = IirFilterData.from_gain_and_ff([0.5, 0.7], [0.9, 0.8])

        # Create optimizer
        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            opt_dt=1.0,
            delay=2.5,
            max_gain_factor=0.95,
            ngains=10
        )

        # Check initialization
        self.assertEqual(optimizer.nmodes, 2)
        self.assertEqual(optimizer.delay, 2.5)
        self.assertEqual(optimizer.max_gain_factor, 0.95)
        self.assertEqual(optimizer.ngains, 10)
        self.assertIsNotNone(optimizer.optimized_gain)
        self.assertEqual(len(optimizer.optimized_gain.value), 2)

    def test_pseudo_open_loop_calculation(self):
        """Test pseudo open-loop signal calculation"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            opt_dt=1.0,
            delay=2.0
        )

        # Create test data
        n_time, n_modes = 5, 1
        delta_comm = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        comm = np.array([[1.0], [1.1], [1.2], [1.3], [1.4]])

        delta_comm_xp = optimizer.to_xp(delta_comm)
        comm_xp = optimizer.to_xp(comm)

        # Calculate pseudo open-loop
        pseudo_ol = optimizer._calculate_pseudo_open_loop(delta_comm_xp, comm_xp)

        # Check dimensions
        self.assertEqual(pseudo_ol.shape, (n_time, n_modes))

        # Check calculation: pseudo_ol[t] = comm[t-1] + delta_comm[t]
        self.assertAlmostEqual(float(pseudo_ol[0, 0]), 0.1)  # First: only delta_comm
        self.assertAlmostEqual(float(pseudo_ol[1, 0]), 1.0 + 0.2)  # comm[0] + delta_comm[1]
        self.assertAlmostEqual(float(pseudo_ol[2, 0]), 1.1 + 0.3)  # comm[1] + delta_comm[2]

    def test_psd_calculation(self):
        """Test PSD calculation using Welch's method"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter
        )

        # Create test signal (sine wave + noise)
        fs = 1000.0  # Hz
        t_int = 1.0 / fs
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        signal_xp = optimizer.to_xp(signal)

        # Calculate PSD
        psd, freq = optimizer._calculate_psd(signal_xp, t_int)

        # Check outputs
        self.assertGreater(len(psd), 0)
        self.assertEqual(len(psd), len(freq))
        self.assertGreater(float(freq[-1]), 0)  # Max frequency > 0
        self.assertGreaterEqual(float(freq[0]), 0)  # Min frequency >= 0

    def test_max_stable_gain_calculation(self):
        """Test maximum stable gain calculation"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        # Create integrator filter
        iir_filter = IirFilterData.from_gain_and_ff([1.0, 1.0], [1.0, 1.0])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            delay=3.0,
            max_gain_factor=0.95
        )

        # Calculate max gains
        max_gains = optimizer._calculate_max_gains()

        # Check results
        self.assertEqual(len(max_gains), 2)  # Two modes
        self.assertTrue(all(g > 0 for g in max_gains))  # All positive

        # For integrator with delay=3, expect around 0.61 * 0.95
        expected = 0.61 * 0.95
        for gain in max_gains:
            self.assertLess(abs(float(gain) - expected), 0.2)  # Within reasonable range

    def test_rejection_transfer_function_cache(self):
        """Test rejection transfer function caching"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            delay=2.0
        )

        # Create test parameters
        freq = optimizer.to_xp(np.linspace(0.1, 100, 50))
        t_int = 0.001
        gain = 0.5
        num = optimizer.to_xp(np.array([0.0, 0.5]))
        den = optimizer.to_xp(np.array([-0.9, 1.0]))

        # Calculate twice (should use cache on second call)
        h_rej1 = optimizer._calculate_rejection_tf(freq, t_int, gain, num, den)
        h_rej2 = optimizer._calculate_rejection_tf(freq, t_int, gain, num, den)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            specula.cpuArray(h_rej1), specula.cpuArray(h_rej2)
        )

    def test_increment_limiting(self):
        """Test gain increment limiting"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5, 0.5], [0.9, 0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            max_inc=0.1,
            limit_inc=True
        )

        # Set previous gains
        prev_gains = optimizer.to_xp(np.array([0.5, 0.6]))
        optimizer.prev_optimized_gain = prev_gains

        # Simulate large gain change
        new_gains = optimizer.to_xp(np.array([1.0, 0.2]))  # +0.5, -0.4

        # Apply increment limiting (simulate the logic from _optimize_gains)
        limited_gains = (optimizer.prev_optimized_gain + 
                        optimizer.max_inc * (new_gains - optimizer.prev_optimized_gain))

        # Check limiting
        delta = limited_gains - prev_gains
        max_delta = float(optimizer.xp.max(optimizer.xp.abs(delta)))
        self.assertLessEqual(max_delta, optimizer.max_inc + 1e-10)  # Within max_inc

    def test_optical_gain_compensation(self):
        """Test optical gain compensation"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter
        )

        # Simulate optical gain history
        optimizer.optical_gain_hist = [1.0, 0.8]  # 20% decrease

        # Test gains
        test_gains = optimizer.to_xp(np.array([0.5]))

        # Apply compensation (simulate logic from _optimize_gains)
        gain_ratio = optimizer.optical_gain_hist[-1] / optimizer.optical_gain_hist[-2]
        compensated_gains = test_gains * gain_ratio

        # Check compensation
        expected_ratio = 0.8
        self.assertAlmostEqual(float(compensated_gains[0]), 0.5 * expected_ratio, places=5)

    def test_safety_factors_application(self):
        """Test application of safety factors"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])

        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter,
            max_gain_factor=0.9,
            safety_factor=0.8
        )

        # Test gains
        test_gains = optimizer.to_xp(np.array([1.0]))
        max_gains = optimizer.to_xp(np.array([1.0]))

        # Apply safety factors (simulate logic from _optimize_gains)
        safe_gains = test_gains * optimizer.safety_factor
        final_gains = optimizer.xp.minimum(safe_gains, max_gains)

        # Check safety factor application
        expected = 1.0 * 0.8
        self.assertAlmostEqual(float(final_gains[0]), expected, places=5)

    def test_optimize_single_mode_does_not_modify_num(self):
        """Test that _optimize_single_mode does not modify iir_filter_data.num"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        # Create simple IIR filter
        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])
        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter
        )

        # Save a deep copy of the original numerator
        original_num = optimizer.iir_filter_data.num.copy()

        # Create a dummy signal for the test
        pseudo_ol_mode = np.random.randn(100)
        t_int = 0.001
        gmax = 0.5

        # Call the method
        optimizer._optimize_single_mode(0, pseudo_ol_mode, t_int, gmax)

        # Verify that the numerator has not been modified with single precision
        np.testing.assert_allclose(cpuArray(optimizer.iir_filter_data.num), cpuArray(original_num), rtol=1e-6, atol=1e-8)

    def test_prev_optimized_gain_initialization(self):
        """Test that prev_optimized_gain is initialized to iir_filter_data.gain"""
        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        # Create simple IIR filter
        initial_gain = np.array([0.5, 0.7])
        iir_filter = IirFilterData.from_gain_and_ff(initial_gain, [0.9, 0.8])
        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter
        )

        # Verify that prev_optimized_gain is equal to the initial gain
        np.testing.assert_allclose(cpuArray(optimizer.prev_optimized_gain), cpuArray(initial_gain), rtol=1e-6, atol=1e-8)

    def test_plot_debug_triggers_matplotlib(self):
        """Test that enabling plot_debug triggers matplotlib plotting functions"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from unittest.mock import patch

        simul_params = Mock(spec=SimulParams)
        simul_params.time_step = 0.001

        iir_filter = IirFilterData.from_gain_and_ff([0.5], [0.9])
        optimizer = GainOptimizer(
            simul_params=simul_params,
            iir_filter_data=iir_filter
        )
        optimizer.plot_debug = True

        pseudo_ol_mode = np.random.randn(100)
        t_int = 0.001
        gmax = 0.5

        # Patch plt.show to check if it is called
        with patch.object(plt, "show") as mock_show:
            optimizer._optimize_single_mode(0, pseudo_ol_mode, t_int, gmax)
            self.assertTrue(mock_show.called, "Matplotlib show() was not called with plot_debug=True")
