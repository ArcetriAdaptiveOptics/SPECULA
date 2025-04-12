import unittest
import os
import shutil
import subprocess
import sys
import glob
import time
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

class TestShSimulation(unittest.TestCase):
    """Test SH SCAO simulation by running a full simulation and checking the results"""
    
    def setUp(self):
        """Set up directories and copy reference calibration files to the right locations"""
        # Create required directories
        os.makedirs('test/calib/subapdata', exist_ok=True)
        os.makedirs('test/calib/rec', exist_ok=True)
        
        # Copy reference calibration files
        if os.path.exists('test/data/scao_subaps_n8_th0.5_ref.fits'):
            shutil.copy('test/data/scao_subaps_n8_th0.5_ref.fits', 
                       'test/calib/subapdata/scao_subaps_n8_th0.5.fits')
        else:
            self.fail("Reference file test/data/scao_subaps_n8_th0.5_ref.fits not found")
            
        if os.path.exists('test/data/scao_rec_n8_th0.5_ref.fits'):
            shutil.copy('test/data/scao_rec_n8_th0.5_ref.fits', 
                       'test/calib/rec/scao_rec_n8_th0.5.fits')
        else:
            self.fail("Reference file test/data/scao_rec_n8_th0.5_ref.fits not found")
    
    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        data_dirs = glob.glob('test/data/2*')
        for data_dir in data_dirs:
            if os.path.isdir(data_dir) and os.path.exists(f"{data_dir}/res_sr.fits"):
                shutil.rmtree(data_dir)
        
        # Clean up copied calibration files
        if os.path.exists('test/calib/subapdata/scao_subaps_n8_th0.5.fits'):
            os.remove('test/calib/subapdata/scao_subaps_n8_th0.5.fits')
        if os.path.exists('test/calib/rec/scao_rec_n8_th0.5.fits'):
            os.remove('test/calib/rec/scao_rec_n8_th0.5.fits')
    
    def test_sh_simulation(self):
        """Run the simulation and check the results"""
        # Get current working directory
        cwd = os.getcwd()
        
        # Change to test directory
        os.chdir('test')
        
        try:
            # Run the simulation
            print("Running SH SCAO simulation...")
            yml_files = ['params_scao_sh_test.yml']
            simul = Simul(*yml_files)
            simul.run()
            
            # Find the most recent data directory (with timestamp)
            data_dirs = sorted(glob.glob('data/2*'))
            self.assertTrue(data_dirs, "No data directory found after simulation")
            latest_data_dir = data_dirs[-1]
            
            # Check if res_sr.fits exists
            res_sr_path = os.path.join(latest_data_dir, 'res_sr.fits')
            self.assertTrue(os.path.exists(res_sr_path), 
                           f"res_sr.fits not found in {latest_data_dir}")
            
            # Verify SR values are within expected range
            with fits.open(res_sr_path) as hdul:
                # Check if there's data
                self.assertTrue(len(hdul) >= 1, "No data found in res_sr.fits")
                self.assertTrue(hasattr(hdul[0], 'data') and hdul[0].data is not None, 
                               "No data found in first HDU of res_sr.fits")
                
                # For this test, we'll check that the SR values are reasonable 
                # (typically between 0.0 and 1.0, with higher values indicating better correction)
                sr_values = hdul[0].data
                self.assertTrue(np.all(sr_values >= 0.0) and np.all(sr_values <= 1.0),
                               f"SR values outside expected range [0,1]: min={np.min(sr_values)}, max={np.max(sr_values)}")
                
                # Check that median SR is above a minimum threshold
                # This value might need adjustment based on your expected performance
                median_sr = np.median(sr_values)
                min_expected_sr = 0.3  # Adjust this based on your expected performance
                self.assertGreaterEqual(median_sr, min_expected_sr,
                                      f"Median SR {median_sr} is below expected minimum {min_expected_sr}")
                
                print(f"Simulation successful. Median SR: {median_sr}")
                
                # Optional: Compare with a reference SR file
                if os.path.exists('data/res_sr_ref.fits'):
                    with fits.open('data/res_sr_ref.fits') as ref_hdul:
                        if hasattr(ref_hdul[0], 'data') and ref_hdul[0].data is not None:
                            np.testing.assert_allclose(
                                sr_values, ref_hdul[0].data, 
                                rtol=1e-3, atol=1e-3,
                                err_msg="SR values do not match reference values"
                            )
                            print("SR values match reference values")
                
        finally:
            # Change back to original directory
            os.chdir(cwd)

    @unittest.skip("This test is only used to create reference files")
    def test_create_reference_sr(self):
        """
        This test is used to create reference SR file for the first time.
        It should be run once, and then the generated file should be renamed
        and committed to the repository.
        """
        # Get current working directory
        cwd = os.getcwd()
        
        # Change to test directory
        os.chdir('test')
        
        try:
            # Run the simulation
            print("Running SH SCAO simulation to create reference SR file...")
            result = subprocess.run(
                [sys.executable, os.path.join('..', 'main', 'scao', 'main_simul.py'), 
                 'params_scao_sh_test.yml'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                universal_newlines=True
            )
            self.assertEqual(result.returncode, 0, f"Simulation failed: {result.stderr}")
            
            # Find the most recent data directory (with timestamp)
            data_dirs = sorted(glob.glob('data/2*'))
            self.assertTrue(data_dirs, "No data directory found after simulation")
            latest_data_dir = data_dirs[-1]
            
            # Check if res_sr.fits exists
            res_sr_path = os.path.join(latest_data_dir, 'res_sr.fits')
            self.assertTrue(os.path.exists(res_sr_path), 
                           f"res_sr.fits not found in {latest_data_dir}")
            
            # Copy to reference file
            shutil.copy(res_sr_path, 'data/res_sr_ref.fits')
            print("Reference SR file created at test/data/res_sr_ref.fits")
        
        finally:
            # Change back to original directory
            os.chdir(cwd)