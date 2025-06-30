import unittest
import os
import glob
import shutil
import specula
specula.init(-1, precision=1)

from specula.simul import Simul
from astropy.io import fits

class TestModalAnalysisSimulation(unittest.TestCase):
    """Test ELT PFS modal analysis simulation with short total_time"""

    def setUp(self):
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.outputdir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(self.datadir, exist_ok=True)
        os.makedirs(self.outputdir, exist_ok=True)
        self.phasescreen_path = os.path.join(self.calibdir, 'phasescreens',
                                   'ps_seed1_dim2048_pixpit0.301_L025.0000_single.fits')
        self.cwd = os.getcwd()

    def tearDown(self):
        # Clean up output directories created by the simulation
        data_dirs = glob.glob(os.path.join(self.outputdir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir):
                shutil.rmtree(data_dir)
        if os.path.exists(self.phasescreen_path):
            os.remove(self.phasescreen_path)
        os.chdir(self.cwd)

    def test_modal_analysis_simulation(self):
        """Run the simulation and check that output files are created"""
        os.chdir(os.path.dirname(__file__))

        yml_files = ['params_elt_pfs_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent output directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.outputdir, '2*')))
        self.assertTrue(data_dirs, "No output directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check that at least one result file exists
        result_files = glob.glob(os.path.join(latest_data_dir, '*.fits'))
        self.assertTrue(result_files, f"No FITS result files found in {latest_data_dir}")

        # Optionally, check the shape or content of a specific output
        for fits_file in result_files:
            with fits.open(fits_file) as hdul:
                self.assertTrue(len(hdul) >= 1, f"No data in {fits_file}")
                self.assertTrue(hasattr(hdul[0], 'data'), f"No data in first HDU of {fits_file}")

if __name__ == "__main__":
    unittest.main()