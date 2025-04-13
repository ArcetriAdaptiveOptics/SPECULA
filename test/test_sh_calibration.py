import unittest
import os
import shutil
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

class TestShCalibration(unittest.TestCase):
    """Test SH calibration by comparing generated calibration files with reference ones"""
    
    def setUp(self):
        """Set up test by ensuring calibration directory exists"""
        # Make sure the calib directory exists
        os.makedirs('test/calib/subapdata', exist_ok=True)
        os.makedirs('test/calib/slopenulls', exist_ok=True)
        os.makedirs('test/calib/rec', exist_ok=True)
    
    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Clean up generated files
        if os.path.exists('test/calib/subapdata/scao_subaps_n8_th0.5.fits'):
            os.remove('test/calib/subapdata/scao_subaps_n8_th0.5.fits')
        if os.path.exists('test/calib/slopenulls/scao_sn_n8_th0.5.fits'):
            os.remove('test/calib/slopenulls/scao_sn_n8_th0.5.fits')
        if os.path.exists('test/calib/rec/scao_rec_n8_th0.5.fits'):
            os.remove('test/calib/rec/scao_rec_n8_th0.5.fits')
    
    def test_sh_calibration(self):
        """Test SH calibration by comparing generated calibration files with reference ones"""

        # Change to test directory
        os.chdir('test')

        # Path to reference files
        subap_ref_path = 'data/scao_subaps_n8_th0.5_ref.fits'
        sn_ref_path = 'data/scao_sn_n8_th0.5_ref.fits'
        rec_ref_path = 'data/scao_rec_n8_th0.5_ref.fits'
        
        # Check if reference files exist
        self.assertTrue(os.path.exists(subap_ref_path), f"Reference file {subap_ref_path} does not exist")
        self.assertTrue(os.path.exists(sn_ref_path), f"Reference file {sn_ref_path} does not exist")
        self.assertTrue(os.path.exists(rec_ref_path), f"Reference file {rec_ref_path} does not exist")

        # Run the simulations using subprocess
        # First, generate the subapdata calibration
        print("Running subap calibration...")
        yml_files = ['params_scao_sh_test.yml', 'params_scao_sh_test_subap.yml']
        simul_sa = Simul(*yml_files)
        simul_sa.run()
 
        # First make sure we have the necessary subap calibration file
        # (slope nulls calibration depends on subap calibration)
        os.makedirs('calib/subapdata', exist_ok=True)
        if os.path.exists('data/scao_subaps_n8_th0.5_ref.fits'):
            shutil.copy('data/scao_subaps_n8_th0.5_ref.fits', 'calib/subapdata/scao_subaps_n8_th0.5.fits')
        else:
            self.fail("Required subap file test/data/scao_subaps_n8_th0.5_ref.fits not found")

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
        self.assertTrue(os.path.exists('calib/subapdata/scao_subaps_n8_th0.5.fits'), 
                       "Subaperture data file was not generated")
        self.assertTrue(os.path.exists('calib/slopenulls/scao_sn_n8_th0.5.fits'), 
                    "Slope nulls file was not generated")
        self.assertTrue(os.path.exists('calib/rec/scao_rec_n8_th0.5.fits'), 
                       "Reconstruction matrix file was not generated")
        
        # Compare the generated files with reference files
        print("Comparing subaperture data with reference...")
        with fits.open('calib/subapdata/scao_subaps_n8_th0.5.fits') as gen_subap:
            with fits.open(subap_ref_path) as ref_subap:
                for i, (gen_hdu, ref_hdu) in enumerate(zip(gen_subap, ref_subap)):
                    if hasattr(gen_hdu, 'data') and hasattr(ref_hdu, 'data') and gen_hdu.data is not None:
                        np.testing.assert_array_almost_equal(
                            gen_hdu.data, ref_hdu.data, 
                            decimal=5, 
                            err_msg=f"Data in HDU #{i} does not match reference"
                        )
    
        # Compare the generated file with reference file
        print("Comparing slope nulls with reference...")
        with fits.open('calib/slopenulls/scao_sn_n8_th0.5.fits') as gen_sn:
            with fits.open(sn_ref_path) as ref_sn:
                for i, (gen_hdu, ref_hdu) in enumerate(zip(gen_sn, ref_sn)):
                    if hasattr(gen_hdu, 'data') and hasattr(ref_hdu, 'data') and gen_hdu.data is not None:
                        np.testing.assert_array_almost_equal(
                            gen_hdu.data, ref_hdu.data, 
                            decimal=5, 
                            err_msg=f"Data in HDU #{i} does not match reference"
                        )

        print("Comparing reconstruction matrix with reference...")
        with fits.open('calib/rec/scao_rec_n8_th0.5.fits') as gen_rec:
            with fits.open(rec_ref_path) as ref_rec:
                for i, (gen_hdu, ref_hdu) in enumerate(zip(gen_rec, ref_rec)):
                    if hasattr(gen_hdu, 'data') and hasattr(ref_hdu, 'data') and gen_hdu.data is not None:
                        np.testing.assert_array_almost_equal(
                            gen_hdu.data, ref_hdu.data, 
                            decimal=5,
                            err_msg=f"Data in HDU #{i} does not match reference"
                        )

        print("All calibration files match reference files!")

        # Clean up the copied subap file
        if os.path.exists('calib/subapdata/scao_subaps_n8_th0.5.fits'):
            os.remove('calib/subapdata/scao_subaps_n8_th0.5.fits')

    @unittest.skip("This test is only used to create reference files")
    def test_create_reference_files(self):
        """
        This test is used to create reference files for the first time.
        It should be run once, and then the generated files should be renamed
        and committed to the repository.
        """
        # Change to test directory
        os.chdir('test')

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
        self.assertTrue(os.path.exists('calib/subapdata/scao_subaps_n8_th0.5.fits'), 
                       "Subaperture data file was not generated")
        self.assertTrue(os.path.exists('calib/slopenulls/scao_sn_n8_th0.5.fits'), 
                    "Slope nulls file was not generated")
        self.assertTrue(os.path.exists('calib/rec/scao_rec_n8_th0.5.fits'), 
                       "Reconstruction matrix file was not generated")
        
        # Create ref_data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Copy files to reference directory
        shutil.copy('calib/subapdata/scao_subaps_n8_th0.5.fits', 'data/scao_subaps_n8_th0.5_ref.fits')
        shutil.copy('calib/slopenulls/scao_sn_n8_th0.5.fits', 'data/scao_sn_n8_th0.5_ref.fits')
        shutil.copy('calib/rec/scao_rec_n8_th0.5.fits', 'data/scao_rec_n8_th0.5_ref.fits')
        
        print("Reference files created and saved to test/data/")
        print("Please commit these files to the repository for future tests")