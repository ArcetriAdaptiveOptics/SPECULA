import unittest
import os
import shutil
import glob
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from astropy.io import fits

import tempfile
import yaml
    
class TestAtmoSimulation(unittest.TestCase):
    """Test AtmoEvolution and AtmoInfiniteEvolution by running a full simulation and checking the results"""

    def setUp(self):
        """Set up test by ensuring calibration directory exists"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.turb_rms_path = os.path.join(self.datadir, 'atmo_s0.8asec_L010m_D8m_100modes_rms.fits')

        if not os.path.exists(self.turb_rms_path):
            self.fail("Reference file {self.turb_rms_path} not found")

        # Get current working directory
        self.cwd = os.getcwd()

    @classmethod
    def tearDownClass(cls):
        """Clean up after test by removing generated files"""
        cls.datadir = os.path.join(os.path.dirname(__file__), 'data')
        cls.calibdir = os.path.join(os.path.dirname(__file__), 'calib')

        # Clean up copied calibration files
        # this can be "single" or "double"
        files = ['ps_seed1_dim8192_pixpit0.100_L010.0000_single.fits',
                 'ps_seed1_dim8192_pixpit0.100_L010.0000_double.fits']
        if os.path.exists(os.path.join(cls.calibdir, 'phasescreens', files[0])):
            os.remove(os.path.join(cls.calibdir, 'phasescreens', files[0]))
        elif os.path.exists(os.path.join(cls.calibdir, 'phasescreens', files[1])):
            os.remove(os.path.join(cls.calibdir, 'phasescreens', files[1]))
        
        # Remove test/data directory with timestamp
        output_dirs = glob.glob(os.path.join(cls.datadir, '2*'))
        for output_dir in output_dirs:
            if os.path.isdir(output_dir) and os.path.exists(f"{output_dir}/modes2.fits"):
                shutil.rmtree(output_dir)

    def test_atmo_simulation(self):
        """Run the simulation and check the results"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        print("Running ATMO simulation...")
        yml_files = ['params_atmo_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent output directory (with timestamp)
        output_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
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
            # Average the Zernike modes of the same radial order (tip and tilt, focus and astigmatisms, ...)
            tolerance = 0.1

            rel_diff1 = []
            rel_diff2 = []
            for n in range(2, len(rms_modes1)+1):
                mean1 = np.mean(rms_modes1[:n])
                mean2 = np.mean(rms_modes2[:n])
                meant = np.mean(turb_rms[:n])
                rel_diff1.append((mean1 - meant) / meant)
                rel_diff2.append((mean2 - meant) / meant)
            rel_diff1 = np.array(rel_diff1)
            rel_diff2 = np.array(rel_diff2)

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

            self.assertTrue(np.all(rel_diff1 < tolerance),
                            "Turbulence RMS from AtmoEvolution does not match theoretical RMS")
            self.assertTrue(np.all(rel_diff2 < tolerance),
                            "Turbulence RMS from AtmoInfiniteEvolution does not match theoretical RMS")
            print("Turbulence RMS match within tolerance.")


    def test_modes2_repeatability(self):
        """Test that modes2.fits is identical when running the simulation multiple times."""

        import time

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Create a minimal override YAML file with the two lines you want
        override_dict = {
            'main_override': {
                'total_time': 10.0  # seconds
            },
            'data_store_override': {
                'inputs': {
                    'input_list': ['modes2-modal_analysis2.out_modes']
                }
            },
            'remove': ['atmo1', 'prop1', 'modal_analysis1']
        }
        with tempfile.NamedTemporaryFile('w', suffix='.yml', delete=False) as tmp:
            yaml.dump(override_dict, tmp)
            override_yml_path = tmp.name

        try:
            # Run the simulation for the first time
            simul1 = Simul('params_atmo_test.yml', override_yml_path)
            simul1.run()

            # Find the latest output directory and modes2.fits after first run
            output_dirs1 = sorted(glob.glob(os.path.join(self.datadir, '2*')))
            self.assertTrue(output_dirs1, "No output directory found after first simulation")
            latest_output_dir1 = output_dirs1[-1]
            modes2_path1 = os.path.join(latest_output_dir1, 'modes2.fits')
            self.assertTrue(os.path.exists(modes2_path1), "modes2.fits not found after first run")

            # Wait a second to ensure a new output directory is created
            time.sleep(1)

            # Run the simulation for the second time
            simul2 = Simul('params_atmo_test.yml', override_yml_path)
            simul2.run()

            # Find the latest output directory and modes2.fits after second run
            output_dirs2 = sorted(glob.glob(os.path.join(self.datadir, '2*')))
            self.assertTrue(output_dirs2, "No output directory found after second simulation")
            latest_output_dir2 = output_dirs2[-1]
            modes2_path2 = os.path.join(latest_output_dir2, 'modes2.fits')
            self.assertTrue(os.path.exists(modes2_path2), "modes2.fits not found after second run")

            # Compare the data in modes2.fits from both runs
            from astropy.io import fits
            with fits.open(modes2_path1) as hdul1, fits.open(modes2_path2) as hdul2:
                data1 = hdul1[0].data
                data2 = hdul2[0].data
                np.testing.assert_array_equal(data1, data2, err_msg="modes2.fits is not identical between runs")

            print("modes2.fits is identical between two consecutive runs.")

        finally:
            # Clean up the temporary file
            os.remove(override_yml_path)