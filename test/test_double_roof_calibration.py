import unittest
import os
import shutil
import specula
specula.init(0)

from specula.simul import Simul
from test.specula_testlib import assert_HDU_contents_match

class TestDoubleRoofCalibration(unittest.TestCase):
    """Test Double Roof calibration by comparing generated calibration files with reference ones"""

    def setUp(self):
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        
        """Set up test by ensuring calibration directory exists"""
        # Make sure the calib directory exists
        os.makedirs(os.path.join(self.calibdir, 'pupils'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'im'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'rec'), exist_ok=True)

        self.pupdata_ref_path = os.path.join(self.datadir, 'scao_dr_pupdata_ref.fits')
        self.im_ref_path = os.path.join(self.datadir, 'scao_dr_im_20modes_ref.fits')
        self.rec_ref_path = os.path.join(self.datadir, 'scao_dr_rec_20modes_ref.fits')

        self.pupdata_path = os.path.join(self.calibdir, 'pupils', 'scao_dr_pupdata.fits')
        self.im_path = os.path.join(self.calibdir, 'im', 'scao_dr_im_20modes.fits')
        self.rec_path = os.path.join(self.calibdir, 'rec', 'scao_dr_rec_20modes.fits')

        self._cleanFiles()
        self.cwd = os.getcwd()

    def _cleanFiles(self):
        if os.path.exists(self.pupdata_path):
            os.remove(self.pupdata_path)
        if os.path.exists(self.im_path):
            os.remove(self.im_path)
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)

    def tearDown(self):
        self._cleanFiles()
        os.chdir(self.cwd)

    def test_dr_pupdata_calibration(self):
        """Test Double Roof PupData calibration by comparing generated calibration file with reference"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Check if reference file exists
        self.assertTrue(os.path.exists(self.pupdata_ref_path), f"Reference file {self.pupdata_ref_path} does not exist")

        # Run the simulation for calibration
        yml_files = ['params_scao_dr_test.yml','params_scao_dr_test_pupdata.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Check if the calibration file was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Double Roof PupData calibration file was not generated")

        # Compare the generated file with the reference file
        assert_HDU_contents_match(self.pupdata_path, self.pupdata_ref_path)

        print("Double Roof PupData calibration matches reference!")

    def test_dr_rec_calibration(self):
        """Test Double Roof reconstruction matrix calibration by comparing generated calibration files with reference ones"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Check if reference files exist
        self.assertTrue(os.path.exists(self.pupdata_ref_path), f"Reference file {self.pupdata_ref_path} does not exist")
        self.assertTrue(os.path.exists(self.im_ref_path), f"Reference file {self.im_ref_path} does not exist")
        self.assertTrue(os.path.exists(self.rec_ref_path), f"Reference file {self.rec_ref_path} does not exist")

        # First, generate the pupdata calibration (needed for reconstruction)
        print("Running Double Roof pupdata calibration...")
        yml_files = ['params_scao_dr_test.yml', 'params_scao_dr_test_pupdata.yml']
        simul_pup = Simul(*yml_files)
        simul_pup.run()

        # Check that pupdata was generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Double Roof PupData calibration file was not generated")

        # Then, generate the reconstruction matrix calibration
        print("Running Double Roof reconstruction calibration...")
        yml_files = ['params_scao_dr_test.yml', 'params_scao_dr_test_rec.yml']
        simul_rec = Simul(*yml_files)
        simul_rec.run()

        # Check if the files were generated
        self.assertTrue(os.path.exists(self.im_path), "Interaction matrix file was not generated")
        self.assertTrue(os.path.exists(self.rec_path), "Reconstruction matrix file was not generated")

        assert_HDU_contents_match(self.im_path, self.im_ref_path)
        assert_HDU_contents_match(self.rec_path, self.rec_ref_path)

        print("All Double Roof calibration files match reference files!")

    @unittest.skipIf(int(os.getenv('CREATE_REF', 0)) < 1, "This test is only used to create reference files")
    def test_create_reference_files(self):
        """Create reference files for Double Roof calibration"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the pupdata calibration
        print("Running Double Roof pupdata calibration...")
        yml_files = ['params_scao_dr_test.yml','params_scao_dr_test_pupdata.yml']
        simul_pup = Simul(*yml_files)
        simul_pup.run()

        # Run the reconstruction calibration
        print("Running Double Roof reconstruction calibration...")
        yml_files = ['params_scao_dr_test.yml', 'params_scao_dr_test_rec.yml']
        simul_rec = Simul(*yml_files)
        simul_rec.run()

        # Check if the calibration files were generated
        self.assertTrue(os.path.exists(self.pupdata_path), "Double Roof PupData calibration file was not generated")
        self.assertTrue(os.path.exists(self.im_path), "Interaction matrix file was not generated")
        self.assertTrue(os.path.exists(self.rec_path), "Reconstruction matrix file was not generated")

        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)

        # Copy files to reference directory
        shutil.copy(self.pupdata_path, self.pupdata_ref_path)
        shutil.copy(self.im_path, self.im_ref_path)
        shutil.copy(self.rec_path, self.rec_ref_path)

        print("Reference files created and saved to test/data/")
        print("Please commit these files to the repository for future tests")
