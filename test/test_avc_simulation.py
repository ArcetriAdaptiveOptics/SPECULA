import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from astropy.io import fits


class TestAVCSimulation(unittest.TestCase):
    """Test AVC by running a single-mode closed loop with a vibration"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.params_file = os.path.join(os.path.dirname(__file__), 'params_avc_test.yml')
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

    def test_avc_simulation(self):
        """Run the simulation and check that the vibration is cancelled"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        simul = Simul(self.params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        with fits.open(os.path.join(latest_data_dir, 'res.fits')) as hdul:
            res = hdul[0].data.ravel()
        with fits.open(os.path.join(latest_data_dir, 'avc_freq.fits')) as hdul:
            freq = hdul[0].data.ravel()

        self.assertEqual(len(res), 2000)

        # Residual: the vibration is cancelled down to the noise
        first = res[:200].std()
        last = res[-200:].std()
        self.assertLess(last, 0.2 * first,
                        msg=f"Residual rms {last:.2f} not reduced enough from {first:.2f}")

        # Frequency estimate pulled from the 46.5 Hz guess towards 47 Hz
        self.assertAlmostEqual(float(freq[-1]), 47.0, delta=0.2)
