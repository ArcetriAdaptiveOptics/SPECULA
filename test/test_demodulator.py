import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from astropy.io import fits
import numpy as np

class TestDemodulator(unittest.TestCase):
    """Test demodulator by running a simulation and checking the output amplitude"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_demodulator_test.yml')
        os.makedirs(self.datadir, exist_ok=True)
        self.expected_amplitude = 5.0  # From params_demodulator_test.yml

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

        # Change back to original directory
        os.chdir(self.cwd)

    def test_demodulator_amplitude(self):
        """Run the simulation and check demodulator output amplitude"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        simul = Simul(self.params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if demodulator output file exists
        demod_file = os.path.join(latest_data_dir, 'dem.fits')
        self.assertTrue(os.path.exists(demod_file), f"Demodulator output file not found: {demod_file}")

        # Read demodulator output
        with fits.open(demod_file) as hdul:
            self.assertTrue(len(hdul) >= 1, "No data found in demodulator output file")
            demod_values = hdul[0].data.copy()
            self.assertIsNotNone(demod_values, "No data found in first HDU of demodulator output file")

            # Check that the output matches the input amplitude (within tolerance)
            mean_demod = np.mean(demod_values)
            tolerance = 0.05 * self.expected_amplitude  # 5% tolerance
            self.assertTrue(
                abs(mean_demod - self.expected_amplitude) < tolerance,
                f"Demodulator output {mean_demod:.3f} does not match expected amplitude {self.expected_amplitude:.3f} (tol={tolerance:.3f})"
            )
            print(f"Demodulator output OK: mean={mean_demod:.3f}, expected={self.expected_amplitude:.3f}")

