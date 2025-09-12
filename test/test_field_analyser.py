import unittest
import os
import shutil
import glob
import specula
specula.init(-1,precision=1)  # Default target device

from specula import np
from specula.simul import Simul
from specula.field_analyser import FieldAnalyser
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

            # Also remove FieldAnalyser output directories
            base_name = os.path.basename(data_dir)
            for suffix in ['_PSF', '_MA', '_CUBE']:
                field_dir = os.path.join(self.datadir, base_name + suffix)
                if os.path.isdir(field_dir):
                    shutil.rmtree(field_dir)

        # Clean up copied calibration files
        if os.path.exists(self.subap_path):
            os.remove(self.subap_path)
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)
        if os.path.exists(self.phasescreen_path):
            os.remove(self.phasescreen_path)

        # Change back to original directory
        os.chdir(self.cwd)

    def test_field_analyser_psf(self):
        """Test FieldAnalyser PSF computation against saved simulation PSF"""

        verbose = False

        # Change to test directory
        os.chdir(os.path.dirname(__file__))

        # Run the simulation with both SR and PSF output
        print("Running SH SCAO simulation with PSF output...")
        yml_files = ['params_field_analyser_test.yml']
        simul = Simul(*yml_files)
        simul.run()

        # Find the most recent data directory (with timestamp)
        data_dirs = sorted(glob.glob(os.path.join(self.datadir, '2*')))
        print(f"Data directories found: {data_dirs}")
        self.assertTrue(data_dirs, "No data directory found after simulation")
        latest_data_dir = data_dirs[-1]

        # Check if res_psf.fits exists (the PSF data from simulation)
        res_psf_path = os.path.join(latest_data_dir, 'res_psf.fits')
        self.assertTrue(os.path.exists(res_psf_path), 
                    f"res_psf.fits not found in {latest_data_dir}")

        # Load the original PSF from simulation
        with fits.open(res_psf_path) as hdul:
            original_psf = hdul[0].data
            original_psf_header = hdul[0].header

        if original_psf.ndim == 3:
            original_psf = original_psf.sum(axis=0)

        # Check if res.fits exists (the modal analysis data from simulation)
        res_path = os.path.join(latest_data_dir, 'res.fits')
        self.assertTrue(os.path.exists(res_path),
                    f"res.fits not found in {latest_data_dir}")

        # Load the original modes from simulation
        with fits.open(res_path) as hdul:
            original_modes = hdul[0].data
            original_modes_header = hdul[0].header

        # Check if phase.fits exists (the phase cube data from simulation)
        phase_path = os.path.join(latest_data_dir, 'phase.fits')
        self.assertTrue(os.path.exists(phase_path),
                    f"phase.fits not found in {latest_data_dir}")

        # Load the original phase cube from simulation
        with fits.open(phase_path) as hdul:
            original_phase = hdul[0].data
            original_phase_header = hdul[0].header

        # extract the phase, discarding the amplitude
        original_phase = original_phase[:,1,:,:]

        if verbose:
            print(f"Original PSF shape: {original_psf.shape}")
            print(f"Original modes shape: {original_modes.shape}")
            print(f"Original phase cube shape: {original_phase.shape}")

        # Now test FieldAnalyser
        print("Testing FieldAnalyser computation...")

        # Setup FieldAnalyser with on-axis source only (same as simulation)
        polar_coords = np.array([[0.0, 0.0]])  # on-axis only

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=os.path.basename(latest_data_dir),
            polar_coordinates=polar_coords,
            wavelength_nm=1650,  # Same as PSF object in params
            start_time=0.0,      # Same as PSF object in params
            end_time=None,
            verbose=True
        )

        # Compute PSF using FieldAnalyser with same sampling as original
        # Extract sampling from original simulation parameters
        psf_sampling =7  # Same as 'nd' parameter in params_field_analyser_test.yml

        psf_results = analyzer.compute_field_psf(
            psf_sampling=psf_sampling,
            force_recompute=True
        )

        # Compute modal analysis
        modal_results = analyzer.compute_modal_analysis()

        # Compute phase cube
        cube_results = analyzer.compute_phase_cube()

        field_psf = psf_results['psf_list'][0]
        modes = modal_results['modal_coeffs'][0]
        phase = cube_results['phase_cubes'][0]

        field_psf = field_psf[0]
        # extract the phase, discarding the amplitude
        phase = phase[:,1,:,:]

        display = False
        if display:
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm

            # Display the PSFs for visual comparison with logarithmic scale
            plt.figure(figsize=(18, 6))

            # Calculate vmin and vmax for consistent scaling
            # Use a small fraction of the maximum to avoid issues with zeros
            vmin = max(1e-8, min(np.min(field_psf[field_psf > 0]), np.min(original_psf[original_psf > 0])))
            vmax = max(np.max(field_psf), np.max(original_psf))

            plt.subplot(1, 3, 1)
            plt.imshow(original_psf, origin='lower', cmap='hot', interpolation='nearest',
                    norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.title('Original PSF from Simulation (Log Scale)')
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(field_psf, origin='lower', cmap='hot', interpolation='nearest',
                    norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.title('FieldAnalyser PSF (Log Scale)')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(original_psf - field_psf), origin='lower', cmap='hot', interpolation='nearest',
                    norm=LogNorm(vmin=vmin, vmax=vmax))
            plt.title('Difference (Original - FieldAnalyser)')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(original_phase[-1], origin='lower', cmap='hot', interpolation='nearest')
            plt.title('Original Phase Cube (Last Slice)')
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.imshow(phase[-1], origin='lower', cmap='hot', interpolation='nearest')
            plt.title('FieldAnalyser Phase Cube (Last Slice)')
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.imshow(np.abs(original_phase[-1] - phase[-1]), origin='lower', cmap='hot', interpolation='nearest')
            plt.title('Phase Difference (Original - FieldAnalyser)')
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        # Verify we got results
        self.assertEqual(len(psf_results['psf_list']), 1, "Expected one PSF result for on-axis source")
        self.assertEqual(len(modal_results['modal_coeffs']), 1,
                        "Expected one modal analysis result for on-axis source")
        self.assertEqual(len(cube_results['phase_cubes']), 1,
                        "Expected one phase cube result for on-axis source")

        if verbose:
            print(f"FieldAnalyser PSF shape: {field_psf.shape}")
            print(f"FieldAnalyser modal coefficients shape: {modes.shape}")
            print(f"FieldAnalyser phase cube shape: {phase.shape}")

        # Compare PSF shapes
        self.assertEqual(field_psf.shape, original_psf.shape,
                        "PSF shapes should match between simulation and FieldAnalyser")
        # Compare modal coefficients shapes
        self.assertEqual(modes.shape, original_modes.shape,
                        "Modal coefficients shape should match between simulation and FieldAnalyser")
        # Compare phase cube shapes
        self.assertEqual(phase.shape, original_phase.shape,
                        "Phase cube shape should match between simulation and FieldAnalyser")

        # normalize PSF data to match original simulation
        field_psf /= field_psf.sum()  # Normalize to match original PSF
        original_psf /= original_psf.sum()  # Normalize to match original PSF

        #Compare PSFs
        np.testing.assert_allclose(
            field_psf, original_psf,
            rtol=1e-3, atol=1e-3,
            err_msg="PSF values do not match between simulation and FieldAnalyser"
        )

        # Compare modal coefficients
        np.testing.assert_allclose(
            modes, original_modes,
            rtol=1e-3, atol=1e-3,
            err_msg="Modal coefficients do not match between simulation and FieldAnalyser"
        )

        # Compare phase cube
        np.testing.assert_allclose(
            phase, original_phase,
            rtol=1e-3, atol=1e-3,
            err_msg="Phase cube values do not match between simulation and FieldAnalyser"
        )

        print(f"FieldAnalyser test successful!")

        # Verify that FieldAnalyser output files were created
        psf_output_dir = analyzer.psf_output_dir
        self.assertTrue(psf_output_dir.exists(), "PSF output directory should exist")

        psf_filename, sr_filename = analyzer._get_psf_filenames(source_dict = analyzer.sources[0])
        psf_path = psf_output_dir / f"{psf_filename}.fits"
        self.assertTrue(psf_path.exists(), f"PSF output file should exist: {psf_path}")

        print(f"FieldAnalyser PSF file saved: {psf_path}")