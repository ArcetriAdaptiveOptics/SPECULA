import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
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
                places=1,
                msg=f"Last gain value {last_gain:.4f} does not match expected 0.5"
            )