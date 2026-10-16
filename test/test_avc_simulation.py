import unittest
import os
import glob
import shutil
import numpy as np
import specula
specula.init(-1, precision=1)  # CPU, single precision

from specula.simul import Simul
from astropy.io import fits


class TestAVCSimulation(unittest.TestCase):
    """Test AVC by running single-mode closed loops with vibrations"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
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

    def _run_simulation(self, params_file):
        """Run the simulation and return the data stored by the DataStore"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        simul = Simul(params_file)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        data = {}
        for name in glob.glob(os.path.join(latest_data_dir, '*.fits')):
            with fits.open(name) as hdul:
                data[os.path.basename(name)[:-len('.fits')]] = hdul[0].data.copy()
        return data

    def test_avc_simulation(self):
        """Run the simulation and check that the vibration is cancelled"""
        data = self._run_simulation('params_avc_test.yml')
        res = data['res'].ravel()
        freq = data['avc_freq'].ravel()

        self.assertEqual(len(res), 2000)

        # Residual: the vibration is cancelled down to the noise
        first = res[:200].std()
        last = res[-200:].std()
        self.assertLess(last, 0.2 * first,
                        msg=f"Residual rms {last:.2f} not reduced enough from {first:.2f}")

        # Frequency estimate pulled from the 46.5 Hz guess towards 47 Hz
        self.assertAlmostEqual(float(freq[-1]), 47.0, delta=0.2)

    def test_avc_two_vibrations_on_one_mode(self):
        """Two vibrations on the same mode: one AVC channel with two
        instances, whose corrections are summed into one command. Both
        vibrations are cancelled and both frequencies tracked."""
        data = self._run_simulation('params_avc_two_vibrations_test.yml')
        res = data['res'].ravel()
        freq = data['avc_freq']

        self.assertEqual(len(res), 2000)
        # one command for the channel, one frequency per vibration
        self.assertEqual(data['avc_comm'].shape, (2000, 1))
        self.assertEqual(freq.shape, (2000, 2))

        # Residual: both vibrations are cancelled down to the noise
        first = res[:200].std()
        last = res[-200:].std()
        self.assertLess(last, 0.2 * first,
                        msg=f"Residual rms {last:.2f} not reduced enough from {first:.2f}")

        # Each frequency estimate pulled from its guess towards its vibration
        np.testing.assert_allclose(freq[-1], [47.0, 113.0], atol=0.2)
