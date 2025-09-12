import unittest
import os
import shutil
import glob
import numpy as np
import specula
specula.init(-1, precision=1)

from specula.simul import Simul
from astropy.io import fits

class TestLowPassFilterSimulation(unittest.TestCase):
    """Test LowPassFilter simulation and compare diff.fits with reference"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'output')
        self.refdir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.refdir, exist_ok=True)
        self.diff_ref_path = os.path.join(self.refdir, 'diff_ref.fits')
        self.cwd = os.getcwd()

    def tearDown(self):
        # Clean up output directories created by the simulation
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
        os.chdir(self.cwd)

    def test_low_pass_filter_simulation(self):
        """Run the simulation and compare diff.fits with reference"""
        os.chdir(os.path.dirname(__file__))

        yml_files = ['params_low_pass_filter_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent output directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No output directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if diff.fits exists
        diff_path = os.path.join(latest_data_dir, 'diff.fits')
        self.assertTrue(os.path.exists(diff_path), f"{diff_path} not found")

        # Compare diff.fits with diff_ref.fits
        with fits.open(diff_path) as hdul, fits.open(self.diff_ref_path) as ref_hdul:
            data = hdul[0].data
            ref_data = ref_hdul[0].data
            np.testing.assert_allclose(
                data, ref_data, rtol=1e-6, atol=1e-6,
                err_msg="diff.fits does not match diff_ref.fits"
            )

    @unittest.skipIf(int(os.getenv('CREATE_REF', 0)) < 1, "This test is only used to create reference files")
    def test_create_reference_diff(self):
        """
        This test is used to create diff_ref.fits for the first time.
        It should be run once, and then the generated file should be renamed
        and committed to the repository.
        """
        os.chdir(os.path.dirname(__file__))
        yml_files = ['params_low_pass_filter_test.yml']
        simul = Simul(*yml_files)
        simul.run()
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No output directory found after simulation")
        latest_data_dir = data_dirs[-1]
        diff_path = os.path.join(latest_data_dir, 'diff.fits')
        self.assertTrue(os.path.exists(diff_path), f"{diff_path} not found")
        shutil.copy(diff_path, self.diff_ref_path)
        print(f"Reference diff_ref.fits created at {self.diff_ref_path}")
