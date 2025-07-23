import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from astropy.io import fits
import numpy as np
import yaml

class TestOpticalGainEstimator(unittest.TestCase):
    """Test optical gain estimator by running a simulation and checking the output"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_optical_gain_estimator_test.yml')
        os.makedirs(self.datadir, exist_ok=True)
        # read yaml file to get expected initial gain
        with open(self.params_file, 'r') as f:
            params = yaml.safe_load(f)
            self.initial_gain = float(params['optical_gain_estimator']['initial_optical_gain'])
            self.int_gain = float(params['optical_gain_estimator']['gain'])
            self.optg = float(params['disturbance1']['amp'][0]) / float(params['disturbance2']['amp'][0])

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

    def test_optical_gain_estimator(self):
        """Run the simulation and check optical gain estimator output"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        simul = Simul(self.params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if optical gain estimator output file exists
        optg_file = os.path.join(latest_data_dir, 'optg.fits')
        self.assertTrue(os.path.exists(optg_file), f"Optical gain estimator output file not found: {optg_file}")

        # Read optical gain estimator output
        with fits.open(optg_file) as hdul:
            self.assertTrue(len(hdul) >= 2, "Expected at least 2 HDUs in optical gain output file")

            # Check times and data
            times = hdul[1].data.copy()
            optg_values = hdul[0].data.copy()
   
            self.assertIsNotNone(times, "No time data found in optical gain output file")
            self.assertIsNotNone(optg_values, "No optical gain data found in output file")

            # Check that we have reasonable data
            self.assertEqual(len(times), len(optg_values), "Times and optical gain data length mismatch")
            self.assertGreater(len(optg_values), 0, "No optical gain data points found")

            # the estimation is:
            #      opticalGain = opticalGain - (1 - demod_delta_cmd/demod_cmd) * gain * opticalGain
            # where:
            # - initial opticalGain           is    self.initial_gain
            # - gain                          is    self.int_gain
            # - demod_delta_cmd/demod_cmd     is    self.optg
            opticalGain = [0] * len(optg_values)
            for i in range(len(optg_values)):
                if i == 0:
                    opticalGain[i] = self.initial_gain - (1 - self.optg) * self.int_gain * self.initial_gain
                else:
                    opticalGain[i] = opticalGain[i-1] - (1 - self.optg) * self.int_gain * opticalGain[i-1]

            # Check that the output matches opticalGain
            for i in range(len(optg_values)):
                val = optg_values[i]
                expected = opticalGain[i]
                # If val and expected are numpy arrays, extract the scalar
                if isinstance(val, np.ndarray):
                    val = val.item()
                if isinstance(expected, np.ndarray):
                    expected = expected.item()
                self.assertAlmostEqual(
                    val, expected,
                    places=3,
                    msg=f"Optical gain value {val:.6f} at index {i} does not match expected {expected:.6f}"
                )