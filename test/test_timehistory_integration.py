import unittest
import os
import shutil
import glob
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

class TestTimeHistoryIntegration(unittest.TestCase):
    """Test the time history generation by running small simulation checking the results"""
    
    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.outputdir = os.path.join(os.path.dirname(__file__), 'output_timehistory')

        # Get current working directory
        self.cwd = os.getcwd()
    
    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        if os.path.exists(self.outputdir):
            shutil.rmtree(self.outputdir)
        
        # Change back to original directory
        os.chdir(self.cwd)
    
    def test_timehistory_integration(self):
        """Run the simulation and check the results"""
        
        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        yml_files = ['params_timehistory_test.yml']
        simul = Simul(*yml_files)
        simul.run()
            
        # Find the most recent data directory (with timestamp)
        output_dirs = sorted(glob.glob(os.path.join(self.outputdir, '*')))
        self.assertTrue(output_dirs, "No data directory found after simulation")
        latest_output_dir = output_dirs[-1]
            
        # Check if output time history file exists
        timehist_path = os.path.join(latest_output_dir, 'timehist.fits')
        self.assertTrue(os.path.exists(timehist_path), 
                       f"timehist.fits not found in {latest_output_dir}")
            
        # Read output time history file
        with fits.open(timehist_path) as hdul:
            self.assertTrue(len(hdul) >= 1, "No data found in timehist.fits")
            self.assertTrue(hasattr(hdul[0], 'data') and hdul[0].data is not None, 
                           "No data found in first HDU of timehist.fits")
                
            saved_data = hdul[0].data

        # Check that the saved data is identical to the input data
        # of our function generator

        reference_data = fits.getdata(os.path.join(self.datadir, 'timehistory_test.fits'))
        np.testing.assert_array_equal(saved_data, reference_data)
            
    
