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
        self.outputdir = os.path.join(os.path.dirname(__file__), 'output')
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')

        self.phasescreen_path = os.path.join(self.calibdir, 'phasescreens',
                                   'ps_seed1_dim16384_pixpit0.05_L010.0000_single.fits')

        self.turb_rms_path = os.path.join(self.datadir, 'atmo_s0.8asec_L010m_D8m_100modes_rms.fits')
        
        if not os.path.exists(self.turb_rms_path):
            self.fail("Reference file {self.turb_rms_path} not found")
            
        # Get current working directory
        self.cwd = os.getcwd()

    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        output_dirs = glob.glob(os.path.join(self.outputdir, '2*'))
        for output_dir in output_dirs:
            if os.path.isdir(output_dir) and os.path.exists(f"{output_dir}/modes1.fits"):
                shutil.rmtree(output_dir)
        
        # Clean up copied calibration files
        if os.path.exists(self.phasescreen_path):
            os.remove(self.phasescreen_path)

        # Change back to original directory
        os.chdir(self.cwd)
    
    def test_sh_simulation(self):
        """Run the simulation and check the results"""
        
        # Change to test directory
        os.chdir(os.path.dirname(__file__))
        print(os.getcwd())
        
        # Run the simulation
        print("Running ATMO simulation...")
        yml_files = ['params_atmo_test.yml']
        simul = Simul(*yml_files)
        simul.run()
            
        # Find the most recent output directory (with timestamp)
        output_dirs = sorted(glob.glob(os.path.join(self.outputdir, '2*')))
        self.assertTrue(output_dirs, "No output directory found after simulation")
        latest_output_dir = output_dirs[-1]
            
        # Check if modes1.fits and modes2.fits exists
        modes1_path = os.path.join(latest_output_dir, 'modes1.fits')
        modes2_path = os.path.join(latest_output_dir, 'modes2.fits')
        self.assertTrue(os.path.exists(modes1_path), 
                       f"modes1.fits not found in {latest_output_dir}")
        self.assertTrue(os.path.exists(modes2_path),
                       f"modes2.fits not found in {latest_output_dir}")
            
        # read modal coefficients from modes1.fits and modes2.fits
        with fits.open(modes1_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) >= 1, "No data found in modes1.fits")
            self.assertTrue(hasattr(hdul[0], 'data') and hdul[0].data is not None, 
                           "No data found in first HDU of modes1.fits")
            modes1 = hdul[0].data
        
        with fits.open(modes2_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) >= 1, "No data found in modes2.fits")
            self.assertTrue(hasattr(hdul[0], 'data') and hdul[0].data is not None,
                            "No data found in first HDU of modes2.fits")
            modes2 = hdul[0].data
            
        # Compute RMS of modes1 and modes2
        rms_modes1 = np.sqrt(np.mean(modes1**2, axis=0))
        rms_modes2 = np.sqrt(np.mean(modes2**2, axis=0))

        # restore fits file with turbulence RMS
        with fits.open(self.turb_rms_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) >= 1, "No data found in turbulence RMS fits file")
            self.assertTrue(hasattr(hdul[0], 'data') and hdul[0].data is not None, 
                           "No data found in first HDU of turbulence RMS fits file")
            turb_rms = hdul[0].data
        
            # Compare the sqrt of the covariance, check if the diagonal elements are similar       
            verbose = False
            if verbose:
                print(f"Empirical RMS from modes1: {rms_modes1[0:9]}")
                print(f"Empirical RMS from modes2: {rms_modes2[0:9]}")
                print(f"Theoretical RMS: {turb_rms[0:9]}")
            display = False
            if display:
                import matplotlib.pyplot as plt
                plt.figure()
                # build a vector of indices for the x-axis
                x = np.arange(len(rms_modes1))+2
                plt.plot(x, rms_modes1, label='Empirical RMS from modes1')
                plt.plot(x, rms_modes2, label='Empirical RMS from modes2')
                plt.plot(x, turb_rms, label='Theoretical RMS')
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('Zernike mode index')
                plt.ylabel('RMS')
                plt.title('RMS Comparison')
                plt.legend()
                plt.show()
            
            # Check if the ratio is within a reasonable range (may need adjustment)
            tolerance = 0.25
            diff1 = abs(rms_modes1 - turb_rms)
            diff2 = abs(rms_modes2 - turb_rms)
            
            self.assertTrue(np.all((diff1 / turb_rms) < tolerance),
                            "Empirical RMS from modes1 does not match theoretical RMS")
            self.assertTrue(np.all((diff2 / turb_rms) < tolerance),
                            "Empirical RMS from modes2 does not match theoretical RMS")
            print("Turbulence RMS match within tolerance.")
