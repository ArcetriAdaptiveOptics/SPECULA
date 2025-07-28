import unittest
import os
import shutil
import glob
import specula
specula.init(0,precision=1)

from specula import np
from specula.simul import Simul
from astropy.io import fits

class TestPyramidSimulation(unittest.TestCase):
    """Test Pyramid simulation by running a full simulation and checking the results"""

    def setUp(self):
        """Set up test by ensuring calibration directory exists"""
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.calibdir = os.path.join(os.path.dirname(__file__), 'calib')
        # Make sure the calib directory exists
        os.makedirs(os.path.join(self.calibdir, 'phasescreens'), exist_ok=True)
        self.phasescreen_path = os.path.join(self.calibdir, 'phasescreens',
                                   'ps_seed1_dim1024_pixpit0.050_L025.0000_single.fits')
        
        # Get current working directory
        self.cwd = os.getcwd()

    def tearDown(self):
        """Clean up after test by removing generated files"""
        # Remove test/data directory with timestamp
        data_dirs = glob.glob(os.path.join(self.datadir, '2*'))
        for data_dir in data_dirs:
            if os.path.isdir(data_dir) and os.path.exists(f"{data_dir}/intensity1.fits"):
                shutil.rmtree(data_dir)
        if os.path.exists(self.phasescreen_path):
            os.remove(self.phasescreen_path)

        # Change back to original directory
        os.chdir(self.cwd)

    def test_pyramid_simulation(self):
        """Run the simulation and check the results"""

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation
        print("Running Pyramid simulation...")
        yml_files = ['params_pyr_ol_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        print(f"Data directories found: {data_dirs}")
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if intensity.fits exists
        intensity1_path = os.path.join(latest_data_dir, 'intensity1.fits')
        intensity2_path = os.path.join(latest_data_dir, 'intensity2.fits')
        intensity3_path = os.path.join(latest_data_dir, 'intensity3.fits')
        intensity4_path = os.path.join(latest_data_dir, 'intensity4.fits')
        self.assertTrue(os.path.exists(intensity1_path),
                       f"intensity1.fits not found in {latest_data_dir}")
        self.assertTrue(os.path.exists(intensity2_path),
                       f"intensity2.fits not found in {latest_data_dir}")
        self.assertTrue(os.path.exists(intensity3_path),
                       f"intensity3.fits not found in {latest_data_dir}")
        self.assertTrue(os.path.exists(intensity4_path),
                       f"intensity4.fits not found in {latest_data_dir}")
        ccd_path = os.path.join(latest_data_dir, 'ccd1.fits')
        self.assertTrue(os.path.exists(ccd_path),
                       f"ccd1.fits not found in {latest_data_dir}")

        # Verify intensity values are within expected range
        # and the size is correct [10,60,60]
        with fits.open(intensity1_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) > 0, "No data found in intensity1.fits")
            data1 = hdul[0].data.copy()  # Copy data to avoid issues with lazy loading
            self.assertIsNotNone(data1, "Data in intensity1.fits is None")
            self.assertTrue(np.all(data1 >= 0), "Intensity values are negative")
            self.assertEqual(data1.shape, (10, 60, 60), "Intensity data shape is incorrect")

        with fits.open(intensity2_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) > 0, "No data found in intensity2.fits")
            data2 = hdul[0].data.copy()  # Copy data to avoid issues with lazy loading

        with fits.open(intensity3_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) > 0, "No data found in intensity3.fits")
            data3 = hdul[0].data.copy()

        with fits.open(intensity4_path) as hdul:
            # Check if there's data
            self.assertTrue(len(hdul) > 0, "No data found in intensity4.fits")
            data4 = hdul[0].data.copy()

        plot_debug = False # Set to True to enable plotting for debugging
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(data1[0], origin='lower')
            plt.colorbar()
            plt.title('Intensity1')
            plt.subplot(1, 3, 2)
            plt.imshow(data2[0], origin='lower')
            plt.colorbar()
            plt.title('Intensity2')
            plt.subplot(1, 3, 3)
            plt.imshow(data3[0], origin='lower')
            plt.colorbar()
            plt.title('Intensity3')
            plt.tight_layout()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(data2[0]-data1[0], origin='lower')
            plt.colorbar()
            plt.title('difference Intensity2 - Intensity1')
            plt.subplot(1, 2, 2)
            plt.imshow(data2[0]-np.rot90(data1[0]), origin='lower')
            plt.colorbar()
            plt.title('difference Intensity2 - rotated Intensity1')
            plt.tight_layout()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(data3[0]-data1[0], origin='lower')
            plt.colorbar()
            plt.title('difference Intensity3 - Intensity1')
            plt.subplot(1, 2, 2)
            plt.imshow(data3[0]-np.roll(data1[0], shift=-1, axis=1), origin='lower')
            plt.colorbar()
            plt.title('difference Intensity3 - shifted Intensity1')
            plt.tight_layout()

            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(data4[0]-data1[0], origin='lower')
            plt.colorbar()
            plt.title('difference Intensity4 - Intensity1')
            plt.subplot(1, 2, 2)
            plt.imshow(data4[0]-np.roll(data1[0], shift=(1, -1), axis=(0, 1)), origin='lower')
            plt.colorbar()
            plt.title('difference Intensity4 - shifted Intensity1')
            plt.tight_layout()

            plt.show()

        # compare the three intensity data:
        # - the second one should be the same rotated bt 90 degrees
        #   (here we expect a large RMS of the difference, up to 20% of the RMS of the values)
        # - the third one should be the same shifted by 1 pixels in X
        #   (here we expect a small RMS of the difference, less than 1% of the RMS of the values)
        # - the fourth one should be the same shifted by 1 pixels in X and Y
        #   (here we expect a relatively small RMS of the difference, less than 10% of the RMS of the values)

        # differences
        diff21 = np.abs(data2 - np.rot90(data1, k=1, axes=(1,2)))
        diff31 = np.abs(data3 - np.roll(data1, shift=-1, axis=2))
        diff41 = np.abs(data4 - np.roll(data1, shift=(1, -1), axis=(1, 2)))
        # RMS of data1
        rms1 = np.sqrt(np.mean(data1**2))
        # RMS of the differences
        rms21 = np.sqrt(np.mean(diff21**2))
        rms31 = np.sqrt(np.mean(diff31**2))
        rms41 = np.sqrt(np.mean(diff41**2))

        verbose = False  # Set to True to enable verbose output
        if verbose:
            print(f"RMS Intensity1: {rms1}, RMS Intensity2: {rms21}, RMS Intensity3: {rms31}, RMS Intensity4: {rms41}")

        self.assertTrue(rms21 < 0.20 * rms1,
                        "Intensity2 data is not a 90-degree rotation of Intensity1 within 20% tolerance")
        self.assertTrue(rms31 < 0.01 * rms1,
                        "Intensity3 data is not a 1-pixel shift of Intensity1 in X direction within 1% tolerance")
        self.assertTrue(rms41 < 0.10 * rms1,
                        "Intensity4 data is not a 1-pixel shift of Intensity1 in Y direction within 10% tolerance")