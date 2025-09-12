import unittest
import os
import shutil
import specula
specula.init(-1,precision=1)  # Default target device

from specula.simul import Simul
from test.specula_testlib import assert_HDU_contents_match


class TestShCalibration(unittest.TestCase):
    """Test SH calibration by comparing generated calibration files with reference ones"""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')

        """Set up test by ensuring calibration directory exists"""
        # Make sure the calib directory exists
        os.makedirs(os.path.join(self.calibdir, 'subapdata'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'slopenulls'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'rec'), exist_ok=True)

        self.subap_ref_path = os.path.join(self.datadir, 'scao_subaps_n8_th0.5_ref.fits')
        self.sn_ref_path = os.path.join(self.datadir, 'scao_sn_n8_th0.5_ref.fits')
        self.im_ref_path = os.path.join(self.datadir, 'scao_im_n8_th0.5_ref.fits')
        self.rec_ref_path = os.path.join(self.datadir, 'scao_rec_n8_th0.5_ref.fits')

        self.subap_path = os.path.join(self.calibdir, 'subapdata', 'scao_subaps_n8_th0.5.fits')
        self.sn_path = os.path.join(self.calibdir, 'slopenulls', 'scao_sn_n8_th0.5.fits')
        self.im_path = os.path.join(self.calibdir, 'im', 'scao_im_n8_th0.5.fits')
        self.rec_path = os.path.join(self.calibdir, 'rec', 'scao_rec_n8_th0.5.fits')

        self._cleanFiles()

        # Get current working directory
        self.cwd = os.getcwd()

    def _cleanFiles(self):
        """Clean up before or after test by removing generated files"""
        if os.path.exists(self.subap_path):
            os.remove(self.subap_path)
        if os.path.exists(self.sn_path):
            os.remove(self.sn_path)
        if os.path.exists(self.im_path):
            os.remove(self.im_path)
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)

    def tearDown(self):
        self._cleanFiles()
        # Change back to original directory
        os.chdir(self.cwd)

    def test_sh_calibration(self):
        """Test SH calibration by comparing generated calibration files with reference ones"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Path to reference files

        # Check if reference files exist
        self.assertTrue(os.path.exists(self.subap_ref_path), f"Reference file {self.subap_path} does not exist")
        self.assertTrue(os.path.exists(self.sn_ref_path), f"Reference file {self.sn_path} does not exist")
        self.assertTrue(os.path.exists(self.im_ref_path), f"Reference file {self.im_path} does not exist")
        self.assertTrue(os.path.exists(self.rec_ref_path), f"Reference file {self.rec_path} does not exist")

        # Run the simulations using subprocess
        # First, generate the subapdata calibration
        print("Running subap calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_subap.yml']
        simul_sa = Simul(*yml_files)
        simul_sa.run()

        # First make sure we have the necessary subap calibration file
        # (slope nulls calibration depends on subap calibration)
        # TODO why this copy? We compare these two files later on...

        # Run the slope nulls calibration
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_sn.yml']
        simul_sn = Simul(*yml_files)
        simul_sn.run()

        # Then, generate the reconstruction matrix calibration
        print("Running reconstruction calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_rec.yml']
        simul_rec = Simul(*yml_files)
        simul_rec.run()

        # Check if the files were generated
        self.assertTrue(os.path.exists(self.subap_path),
                       "Subaperture data file was not generated")
        self.assertTrue(os.path.exists(self.sn_path),
                    "Slope nulls file was not generated")
        self.assertTrue(os.path.exists(self.im_path),
                       "Interaction matrix file was not generated")
        self.assertTrue(os.path.exists(self.rec_path),
                       "Reconstruction matrix file was not generated")

        assert_HDU_contents_match(self.subap_path, self.subap_ref_path)
        assert_HDU_contents_match(self.sn_path, self.sn_ref_path)
        assert_HDU_contents_match(self.im_path, self.im_ref_path)
        assert_HDU_contents_match(self.rec_path, self.rec_ref_path)

        print("All calibration files match reference files!")

        # Clean up the copied subap file
        if os.path.exists(self.subap_path):
            os.remove(self.subap_path)

    @unittest.skipIf(int(os.getenv('CREATE_REF', 0)) < 1, "This test is only used to create reference files")
    def test_create_reference_files(self):
        """
        This test is used to create reference files for the first time.
        It should be run once, and then the generated files should be renamed
        and committed to the repository.
        """

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulations
        print("Running subap calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_subap.yml']
        simul_sa = Simul(*yml_files)
        simul_sa.run()

        print("Running slope nulls calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_sn.yml']
        simul_sn = Simul(*yml_files)
        simul_sn.run()       

        print("Running reconstruction calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_rec.yml']
        simul_rec = Simul(*yml_files)
        simul_rec.run()

        # Check if the files were generated
        self.assertTrue(os.path.exists(self.subap_path),
                       "Subaperture data file was not generated")
        self.assertTrue(os.path.exists(self.sn_path),
                    "Slope nulls file was not generated")
        self.assertTrue(os.path.exists(self.im_path),
                       "Interaction matrix file was not generated")
        self.assertTrue(os.path.exists(self.rec_path),
                       "Reconstruction matrix file was not generated")

        # Create ref_data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Copy files to reference directory
        shutil.copy(self.subap_path, self.subap_ref_path)
        shutil.copy(self.sn_path, self.sn_ref_path)
        shutil.copy(self.im_path, self.im_ref_path)
        shutil.copy(self.rec_path, self.rec_ref_path)

        print("Reference files created and saved to test/data/")
        print("Please commit these files to the repository for future tests")
