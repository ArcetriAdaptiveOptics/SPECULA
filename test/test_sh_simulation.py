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
        """Set up test by ensuring calibration directory exists"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')

        # Make sure the calib directory exists
        os.makedirs(os.path.join(self.calibdir, 'subapdata'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'slopenulls'), exist_ok=True)
        os.makedirs(os.path.join(self.calibdir, 'rec'), exist_ok=True)

        self.subap_ref_path = os.path.join(self.datadir, 'scao_subaps_n8_th0.5_ref.fits')
        self.sn_ref_path = os.path.join(self.datadir, 'scao_sn_n8_th0.5_ref.fits')
        self.rec_ref_path = os.path.join(self.datadir, 'scao_rec_n8_th0.5_ref.fits')
        self.res_sr_ref_path = os.path.join(self.datadir, 'res_sr_ref.fits')
    
        self.subap_path = os.path.join(self.calibdir, 'subapdata', 'scao_subaps_n8_th0.5.fits')
        self.sn_path = os.path.join(self.calibdir, 'slopenulls', 'scao_sn_n8_th0.5.fits')
        self.rec_path = os.path.join(self.calibdir, 'rec', 'scao_rec_n8_th0.5.fits')
        self.phasescreen_path = os.path.join(self.calibdir, 'phasescreens',
                                   'ps_seed1_dim1024_pixpit0.016_L025.0000_single.fits')

        # Copy reference calibration files
        if os.path.exists(self.subap_ref_path):
            shutil.copy(self.subap_ref_path, self.subap_path)
        else:
            self.fail(f"Reference file {self.subap_ref_path} not found")
            
        if os.path.exists(self.rec_ref_path):
            shutil.copy(self.rec_ref_path, self.rec_path)
        else:
            self.fail("Reference file {self.rec_path} not found")

        # Get current working directory
        self.cwd = os.getcwd()
        
    
    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir) and os.path.exists(f"{data_dir}/res_sr.fits"):
                shutil.rmtree(data_dir)
        
        # Clean up copied calibration files
        if os.path.exists(self.subap_path):
            os.remove(self.subap_path)
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)
        if os.path.exists(self.phasescreen_path):
            os.remove(self.phasescreen_path)

        # Change back to original directory
        os.chdir(self.cwd)
    
    def test_sh_simulation(self):
        """Run the simulation and check the results"""
        
        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        print(os.path.exists(self.rec_path))
        print(os.getcwd())
        
        # Run the simulation
        print("Running SH SCAO simulation...")
        yml_files = ['params_scao_sh_test.yml']
        simul = Simul(*yml_files)
        simul.run()
            
        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
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
            if os.path.exists(self.res_sr_ref_path):
                with fits.open(self.res_sr_ref_path) as ref_hdul:
                    if hasattr(ref_hdul[0], 'data') and ref_hdul[0].data is not None:
                        np.testing.assert_allclose(
                            sr_values, ref_hdul[0].data, 
                            rtol=1e-3, atol=1e-3,
                            err_msg="SR values do not match reference values"
                        )
                        print("SR values match reference values")
            
    @unittest.skip("This test is only used to create reference files")
    def test_create_reference_sr(self):
        """
        This test is used to create reference SR file for the first time.
        It should be run once, and then the generated file should be renamed
        and committed to the repository.
        """
        # Change to test directory
        os.chdir(os.path.dirname(__file__))
        
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
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]
        
        # Check if res_sr.fits exists
        res_sr_path = os.path.join(latest_data_dir, 'res_sr.fits')
        self.assertTrue(os.path.exists(res_sr_path), 
                       f"res_sr.fits not found in {latest_data_dir}")
        
        # Copy to reference file
        shutil.copy(res_sr_path, self.res_sr_ref_path)
        print(f"Reference SR file created at {self.res_sr_ref_path}")
    
