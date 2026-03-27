import unittest
import os
import shutil
import glob
import pickle
import yaml
import specula
specula.init(0)  # Default target device

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
                shutil.rmtree(data_dir, ignore_errors=True)

            # Also remove FieldAnalyser output directories
            base_name = os.path.basename(data_dir)
            for suffix in ['_PSF', '_MA', '_CUBE']:
                field_dir = os.path.join(self.datadir, base_name + suffix)
                if os.path.isdir(field_dir):
                    shutil.rmtree(field_dir, ignore_errors=True)

        for dirname in ['decimated_tn', 'decimated_pickle_tn', 'legacy_tn', 'legacy_pickle_tn']:
            path = os.path.join(self.datadir, dirname)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

        for path in glob.glob(os.path.join(self.datadir, 'modal_unit_*')):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)

        # Clean up copied calibration files
        if os.path.exists(self.subap_path):
            os.remove(self.subap_path)
        if os.path.exists(self.rec_path):
            os.remove(self.rec_path)
        ps_dir = os.path.dirname(self.phasescreen_path)
        ps_base = os.path.basename(self.phasescreen_path).replace('_single.fits', '_*.fits')
        for fpath in glob.glob(os.path.join(ps_dir, ps_base)):
            os.remove(fpath)

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

        print("FieldAnalyser test successful!")

        # Verify that FieldAnalyser output files were created
        psf_output_dir = analyzer.psf_output_dir
        self.assertTrue(psf_output_dir.exists(), "PSF output directory should exist")

        psf_filename, _ = analyzer._get_psf_filenames(source_dict = analyzer.sources[0])
        psf_path = psf_output_dir / f"{psf_filename}.fits"
        self.assertTrue(psf_path.exists(), f"PSF output file should exist: {psf_path}")

        print(f"FieldAnalyser PSF file saved: {psf_path}")

    def test_field_analyser_rejects_decimated_replay_inputs(self):
        tracking_number = 'decimated_tn'
        tn_dir = os.path.join(self.datadir, tracking_number)
        os.makedirs(tn_dir, exist_ok=True)

        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as outfile:
            yaml.dump(params, outfile)

        hdr = fits.Header()
        hdr['OBJ_TYPE'] = 'BaseValue'
        hdr['DOWNSAMP'] = 4
        hdr['DSMODE'] = 'SAMPLE'

        fits.HDUList([
            fits.PrimaryHDU(np.zeros((1, 1), dtype=np.float32), header=hdr),
            fits.ImageHDU(np.array([0], dtype=np.uint64), header=hdr)
        ]).writeto(os.path.join(tn_dir, 'comm.fits'), overwrite=True)

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=tracking_number,
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False
        )

        replay_params = {
            'data_source': {
                'class': 'DataSource',
                'store_dir': tn_dir,
                'data_format': 'fits',
                'outputs': ['comm']
            }
        }

        with self.assertRaisesRegex(ValueError, 'DOWNSAMP=4'):
            analyzer._validate_replay_inputs_are_not_downsampled(replay_params)

    def test_field_analyser_rejects_decimated_pickle_replay_inputs(self):
        tracking_number = 'decimated_pickle_tn'
        tn_dir = os.path.join(self.datadir, tracking_number)
        os.makedirs(tn_dir, exist_ok=True)

        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as outfile:
            yaml.dump(params, outfile)

        payload = {
            'data': np.zeros((1, 1), dtype=np.float32),
            'times': np.array([0], dtype=np.uint64),
            'hdr': {'OBJ_TYPE': 'BaseValue', 'DOWNSAMP': 3, 'DSMODE': 'SAMPLE'}
        }
        with open(os.path.join(tn_dir, 'comm.pickle'), 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=tracking_number,
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False
        )

        replay_params = {
            'data_source': {
                'class': 'DataSource',
                'store_dir': tn_dir,
                'data_format': 'pickle',
                'outputs': ['comm']
            }
        }

        with self.assertRaisesRegex(ValueError, 'DOWNSAMP=3'):
            analyzer._validate_replay_inputs_are_not_downsampled(replay_params)

    def test_field_analyser_accepts_legacy_fits_replay_inputs_without_decim(self):
        tracking_number = 'legacy_tn'
        tn_dir = os.path.join(self.datadir, tracking_number)
        os.makedirs(tn_dir, exist_ok=True)

        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as outfile:
            yaml.dump(params, outfile)

        hdr = fits.Header()
        hdr['OBJ_TYPE'] = 'BaseValue'

        fits.HDUList([
            fits.PrimaryHDU(np.zeros((1, 1), dtype=np.float32), header=hdr),
            fits.ImageHDU(np.array([0], dtype=np.uint64), header=hdr)
        ]).writeto(os.path.join(tn_dir, 'comm.fits'), overwrite=True)

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=tracking_number,
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False
        )

        replay_params = {
            'data_source': {
                'class': 'DataSource',
                'store_dir': tn_dir,
                'data_format': 'fits',
                'outputs': ['comm']
            }
        }

        analyzer._validate_replay_inputs_are_not_downsampled(replay_params)

    def test_field_analyser_accepts_legacy_pickle_replay_inputs_without_hdr(self):
        tracking_number = 'legacy_pickle_tn'
        tn_dir = os.path.join(self.datadir, tracking_number)
        os.makedirs(tn_dir, exist_ok=True)

        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as outfile:
            yaml.dump(params, outfile)

        payload = {
            'data': np.zeros((1, 1), dtype=np.float32),
            'times': np.array([0], dtype=np.uint64)
        }
        with open(os.path.join(tn_dir, 'comm.pickle'), 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=tracking_number,
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False
        )

        replay_params = {
            'data_source': {
                'class': 'DataSource',
                'store_dir': tn_dir,
                'data_format': 'pickle',
                'outputs': ['comm']
            }
        }

        analyzer._validate_replay_inputs_are_not_downsampled(replay_params)


class TestModalParamsHandling(unittest.TestCase):
    """Unit tests for the simplified modal_params pass-through to ModalAnalysis."""

    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self._created_tn_dirs = []

    def tearDown(self):
        for tn_dir in self._created_tn_dirs:
            if os.path.isdir(tn_dir):
                shutil.rmtree(tn_dir)

    def _make_analyzer(self, tn_name, pixel_pupil=8):
        tn_dir = os.path.join(self.datadir, f'modal_unit_{tn_name}')
        os.makedirs(tn_dir, exist_ok=True)
        self._created_tn_dirs.append(tn_dir)
        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': pixel_pupil, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as f:
            yaml.dump(params, f)
        return FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=f'modal_unit_{tn_name}',
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False
        )

    def _fake_replay_base(self):
        return {
            'main': {'pixel_pupil': 8},
            'prop': {'class': 'AtmoPropagation'}
        }

    # ------------------------------------------------------------------
    # _get_modal_filename
    # ------------------------------------------------------------------

    def test_modal_filename_with_ifunc_ref(self):
        """ifunc_ref appears in filename; nmodes/type_str absent."""
        analyzer = self._make_analyzer('filename_ifunc_ref')
        source = {'polar_coordinates': [15.0, 45.0]}
        fname = analyzer._get_modal_filename(source, {'ifunc_ref': 'my_ifunc', 'obsratio': 0.1})
        self.assertIn('_ifrefmy_ifunc', fname)
        self.assertIn('_obs0.10', fname)
        self.assertNotIn('nmodes', fname)

    def test_modal_filename_with_ifunc_inv_ref(self):
        """ifunc_inv_ref uses the same filename tag as ifunc_ref."""
        analyzer = self._make_analyzer('filename_ifunc_inv_ref')
        source = {'polar_coordinates': [0.0, 0.0]}
        fname = analyzer._get_modal_filename(source, {'ifunc_inv_ref': 'my_ifunc_inv'})
        self.assertIn('_ifrefmy_ifunc_inv', fname)

    def test_modal_filename_legacy_nmodes(self):
        """Legacy nmodes+type_str appear in filename as before."""
        analyzer = self._make_analyzer('filename_legacy')
        source = {'polar_coordinates': [0.0, 0.0]}
        fname = analyzer._get_modal_filename(source, {'nmodes': 50, 'type_str': 'zernike'})
        self.assertIn('_nmodes50', fname)
        self.assertIn('_zernike', fname)

    # ------------------------------------------------------------------
    # _build_replay_params_modal  (patching I/O-heavy methods)
    # ------------------------------------------------------------------

    def test_build_replay_modal_with_ifunc_ref(self):
        """ifunc_ref is passed to ModalAnalysis; no separate IFunc entry is created."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('build_ifunc_ref')
        with patch.object(analyzer, '_build_replay_params_from_datastore',
                          return_value=self._fake_replay_base()), \
             patch.object(analyzer, '_add_field_sources_to_params'):
            result = analyzer._build_replay_params_modal(
                {'ifunc_ref': 'my_ifunc', 'nmodes': 50}
            )
        ma = result['modal_analysis_0']
        self.assertEqual(ma['class'], 'ModalAnalysis')
        self.assertEqual(ma['ifunc_ref'], 'my_ifunc')
        self.assertEqual(ma['nmodes'], 50)
        self.assertNotIn('modal_analysis_ifunc', result)

    def test_build_replay_modal_with_ifunc_inv_ref(self):
        """ifunc_inv_ref is passed to ModalAnalysis."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('build_ifunc_inv_ref')
        with patch.object(analyzer, '_build_replay_params_from_datastore',
                          return_value=self._fake_replay_base()), \
             patch.object(analyzer, '_add_field_sources_to_params'):
            result = analyzer._build_replay_params_modal({'ifunc_inv_ref': 'my_inv'})
        ma = result['modal_analysis_0']
        self.assertEqual(ma['ifunc_inv_ref'], 'my_inv')
        self.assertNotIn('ifunc_ref', ma)

    def test_build_replay_modal_with_type_str(self):
        """Legacy type_str/nmodes/npixels/obsratio are forwarded verbatim."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('build_type_str')
        with patch.object(analyzer, '_build_replay_params_from_datastore',
                          return_value=self._fake_replay_base()), \
             patch.object(analyzer, '_add_field_sources_to_params'):
            result = analyzer._build_replay_params_modal(
                {'type_str': 'zernike', 'nmodes': 30, 'npixels': 8, 'obsratio': 0.2}
            )
        ma = result['modal_analysis_0']
        self.assertEqual(ma['type_str'], 'zernike')
        self.assertEqual(ma['nmodes'], 30)
        self.assertEqual(ma['npixels'], 8)
        self.assertEqual(ma['obsratio'], 0.2)
        self.assertNotIn('ifunc_ref', ma)

    # ------------------------------------------------------------------
    # compute_modal_analysis  default-setting logic
    # ------------------------------------------------------------------

    def test_no_defaults_without_ifunc(self):
        """Without explicit ifunc keys, compute_modal_analysis does not inject defaults."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('no_defaults_no_ifunc')
        modal_params = {}
        with patch.object(analyzer, '_build_replay_params_modal',
                          side_effect=RuntimeError('stop')):
            try:
                analyzer.compute_modal_analysis(modal_params=modal_params, force_recompute=True)
            except RuntimeError:
                pass
        self.assertEqual(modal_params, {})

    def test_no_defaults_with_ifunc_ref(self):
        """With ifunc_ref, type_str/nmodes/npixels defaults are NOT added."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('no_defaults_ifunc_ref')
        modal_params = {'ifunc_ref': 'my_ifunc'}
        with patch.object(analyzer, '_build_replay_params_modal',
                          side_effect=RuntimeError('stop')):
            try:
                analyzer.compute_modal_analysis(modal_params=modal_params, force_recompute=True)
            except RuntimeError:
                pass
        self.assertNotIn('type_str', modal_params)
        self.assertNotIn('nmodes', modal_params)
        self.assertNotIn('npixels', modal_params)

    def test_no_defaults_with_ifunc_inv_ref(self):
        """With ifunc_inv_ref, type_str/nmodes/npixels defaults are NOT added."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('no_defaults_ifunc_inv_ref')
        modal_params = {'ifunc_inv_ref': 'my_inv'}
        with patch.object(analyzer, '_build_replay_params_modal',
                          side_effect=RuntimeError('stop')):
            try:
                analyzer.compute_modal_analysis(modal_params=modal_params, force_recompute=True)
            except RuntimeError:
                pass
        self.assertNotIn('type_str', modal_params)
        self.assertNotIn('nmodes', modal_params)

    def test_no_defaults_with_ifunc_string(self):
        """With string-valued ifunc, defaults are not added and the key is preserved."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('no_defaults_ifunc_string')
        modal_params = {'ifunc': 'my_ifunc'}
        with patch.object(analyzer, '_build_replay_params_modal',
                          side_effect=RuntimeError('stop')):
            try:
                analyzer.compute_modal_analysis(modal_params=modal_params, force_recompute=True)
            except RuntimeError:
                pass
        self.assertEqual(modal_params.get('ifunc'), 'my_ifunc')
        self.assertNotIn('ifunc_ref', modal_params)
        self.assertNotIn('type_str', modal_params)
        self.assertNotIn('nmodes', modal_params)
        self.assertNotIn('npixels', modal_params)

    def test_no_defaults_with_ifunc_direct(self):
        """With ifunc (non-string, direct object), defaults are NOT added."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('no_defaults_ifunc_direct')
        mock_ifunc = object()  # any non-string value
        modal_params = {'ifunc': mock_ifunc}
        with patch.object(analyzer, '_build_replay_params_modal',
                          side_effect=RuntimeError('stop')):
            try:
                analyzer.compute_modal_analysis(modal_params=modal_params, force_recompute=True)
            except RuntimeError:
                pass
        self.assertNotIn('type_str', modal_params)
        self.assertNotIn('nmodes', modal_params)
        self.assertNotIn('npixels', modal_params)

    # ------------------------------------------------------------------
    # _get_modal_filename — new ifunc / ifunc_inv (no _ref) branches
    # ------------------------------------------------------------------

    def test_modal_filename_with_ifunc_string(self):
        """Direct ifunc string value appears in filename."""
        analyzer = self._make_analyzer('filename_ifunc_str')
        source = {'polar_coordinates': [0.0, 0.0]}
        fname = analyzer._get_modal_filename(source, {'ifunc': 'my_ifunc'})
        self.assertIn('_ifuncmy_ifunc', fname)

    def test_modal_filename_with_ifunc_object(self):
        """Direct ifunc non-string value produces 'custom' tag in filename."""
        analyzer = self._make_analyzer('filename_ifunc_obj')
        source = {'polar_coordinates': [0.0, 0.0]}
        fname = analyzer._get_modal_filename(source, {'ifunc': object()})
        self.assertIn('_ifunccustom', fname)

    def test_modal_filename_with_ifunc_inv_string(self):
        """Direct ifunc_inv string uses the same filename tag as ifunc."""
        analyzer = self._make_analyzer('filename_ifinv_str')
        source = {'polar_coordinates': [0.0, 0.0]}
        fname = analyzer._get_modal_filename(source, {'ifunc_inv': 'my_inv'})
        self.assertIn('_ifuncmy_inv', fname)

    # ------------------------------------------------------------------
    # _build_replay_params_modal — direct ifunc forwarded in config
    # ------------------------------------------------------------------

    def test_build_replay_modal_passes_ifunc_direct(self):
        """Direct ifunc param is forwarded verbatim into ModalAnalysis config."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('build_ifunc_direct')
        mock_ifunc = object()
        with patch.object(analyzer, '_build_replay_params_from_datastore',
                          return_value=self._fake_replay_base()), \
             patch.object(analyzer, '_add_field_sources_to_params'):
            result = analyzer._build_replay_params_modal({'ifunc': mock_ifunc})
        ma = result['modal_analysis_0']
        self.assertIs(ma['ifunc'], mock_ifunc)
        self.assertNotIn('type_str', ma)

    # ------------------------------------------------------------------
    # _build_replay_params_modal — passthrough behavior
    # ------------------------------------------------------------------

    def test_build_replay_modal_passes_unknown_key_through(self):
        """modal_params are passed as-is to ModalAnalysis configuration."""
        from unittest.mock import patch
        analyzer = self._make_analyzer('build_unknown_modal_key')
        with patch.object(analyzer, '_build_replay_params_from_datastore',
                          return_value=self._fake_replay_base()), \
             patch.object(analyzer, '_add_field_sources_to_params'):
            result = analyzer._build_replay_params_modal({'custom_key': 123})

        ma = result['modal_analysis_0']
        self.assertEqual(ma['custom_key'], 123)


class TestReplayPrecisionHandling(unittest.TestCase):
    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self._created_tn_dirs = []

    def tearDown(self):
        for tn_dir in self._created_tn_dirs:
            if os.path.isdir(tn_dir):
                shutil.rmtree(tn_dir)

    def _make_analyzer(self, tn_name):
        tn_dir = os.path.join(self.datadir, f'precision_unit_{tn_name}')
        os.makedirs(tn_dir, exist_ok=True)
        self._created_tn_dirs.append(tn_dir)
        params = {
            'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
            'prop': {'class': 'AtmoPropagation'}
        }
        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as handle:
            yaml.dump(params, handle)

        analyzer = FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=f'precision_unit_{tn_name}',
            polar_coordinates=np.array([[0.0, 0.0]]),
            verbose=False,
        )
        return analyzer, tn_dir

    def test_get_saved_replay_precision_reads_global_precision(self):
        analyzer, tn_dir = self._make_analyzer('read_global_precision')
        replay_cfg = {
            'data_source': {
                'class': 'DataSource',
                'global_precision': 1,
            }
        }
        with open(os.path.join(tn_dir, 'replay_params.yml'), 'w', encoding='utf-8') as handle:
            yaml.dump(replay_cfg, handle)

        self.assertEqual(analyzer._get_saved_replay_precision(), 1)

    def test_get_saved_replay_precision_returns_none_when_missing(self):
        analyzer, _ = self._make_analyzer('missing_global_precision')
        self.assertIsNone(analyzer._get_saved_replay_precision())

    def test_ensure_replay_precision_calls_specula_init_on_mismatch(self):
        from unittest.mock import patch
        analyzer, _ = self._make_analyzer('ensure_precision_mismatch')
        with patch('specula.field_analyser.specula.init') as mock_init, \
             patch('specula.field_analyser.specula.global_precision', 0), \
             patch('specula.field_analyser.specula.default_target_device_idx', -1), \
             patch('specula.field_analyser.specula.process_rank', None), \
             patch('specula.field_analyser.specula.process_comm', None), \
             patch('specula.field_analyser.specula.MPI_DBG', False):
            analyzer._ensure_replay_precision(1)

        mock_init.assert_called_once_with(
            device_idx=-1,
            precision=1,
            rank=None,
            comm=None,
            mpi_dbg=False,
        )

    def test_ensure_replay_precision_skips_if_already_matching(self):
        from unittest.mock import patch
        analyzer, _ = self._make_analyzer('ensure_precision_match')
        with patch('specula.field_analyser.specula.init') as mock_init, \
             patch('specula.field_analyser.specula.global_precision', 1):
            analyzer._ensure_replay_precision(1)

        mock_init.assert_not_called()


class TestFieldAnalyserWeakSpots(unittest.TestCase):
    def setUp(self):
        self.datadir = os.path.join(os.path.dirname(__file__), 'data')
        self._created_tn_dirs = []

    def tearDown(self):
        for tn_dir in self._created_tn_dirs:
            if os.path.isdir(tn_dir):
                shutil.rmtree(tn_dir, ignore_errors=True)

    def _make_analyzer(self, tn_name, polar_coordinates=None, display=False, params=None):
        if polar_coordinates is None:
            polar_coordinates = np.array([[0.0, 0.0]])
        tn_dir = os.path.join(self.datadir, f'weak_unit_{tn_name}')
        os.makedirs(tn_dir, exist_ok=True)
        self._created_tn_dirs.append(tn_dir)

        if params is None:
            params = {
                'main': {'class': 'SimulParams', 'pixel_pupil': 8, 'pixel_pitch': 1.0},
                'prop': {'class': 'AtmoPropagation'}
            }

        with open(os.path.join(tn_dir, 'params.yml'), 'w', encoding='utf-8') as handle:
            yaml.dump(params, handle)

        return FieldAnalyser(
            data_dir=self.datadir,
            tracking_number=f'weak_unit_{tn_name}',
            polar_coordinates=polar_coordinates,
            display=display,
            verbose=False,
        )

    def test_setup_sources_accepts_2xN_coordinate_format(self):
        analyzer = self._make_analyzer(
            'coords_2xn',
            polar_coordinates=np.array([[0.0, 15.0], [0.0, 90.0]])
        )
        self.assertEqual(len(analyzer.sources), 2)
        self.assertEqual(analyzer.sources[0]['polar_coordinates'], [0.0, 0.0])
        self.assertEqual(analyzer.sources[1]['polar_coordinates'], [15.0, 90.0])

    def test_add_displays_to_params_injects_phase_and_dm_displays(self):
        analyzer = self._make_analyzer('display_on', display=True)
        replay_params = {
            'prop': {'class': 'AtmoPropagation'},
            'dm0': {'class': 'DM'},
            'not_dm': {'class': 'ShSlopec'},
        }
        analyzer._add_displays_to_params(replay_params)

        self.assertIn('ph_disp', replay_params)
        self.assertIn('dm0_disp', replay_params)
        self.assertEqual(replay_params['ph_disp']['inputs']['phase'], 'prop.out_field_source_0_ef')
        self.assertEqual(replay_params['dm0_disp']['inputs']['phase'], 'dm0.out_layer')
        self.assertNotIn('not_dm_disp', replay_params)

    def test_add_displays_to_params_noop_when_disabled(self):
        analyzer = self._make_analyzer('display_off', display=False)
        replay_params = {'prop': {'class': 'AtmoPropagation'}, 'dm0': {'class': 'DM'}}
        analyzer._add_displays_to_params(replay_params)

        self.assertNotIn('ph_disp', replay_params)
        self.assertNotIn('dm0_disp', replay_params)

    def test_get_saved_replay_precision_invalid_value_returns_none(self):
        analyzer = self._make_analyzer('invalid_precision')
        tn_dir = analyzer.tn_dir
        replay_cfg = {
            'data_source': {
                'class': 'DataSource',
                'global_precision': 2,
            }
        }
        with open(os.path.join(tn_dir, 'replay_params.yml'), 'w', encoding='utf-8') as handle:
            yaml.dump(replay_cfg, handle)

        self.assertIsNone(analyzer._get_saved_replay_precision())

    def test_ensure_replay_precision_skips_when_target_device_unset(self):
        from unittest.mock import patch
        analyzer = self._make_analyzer('precision_target_none')

        with patch('specula.field_analyser.specula.init') as mock_init, \
             patch('specula.field_analyser.specula.global_precision', 0), \
             patch('specula.field_analyser.specula.default_target_device_idx', None):
            analyzer._ensure_replay_precision(1)

        mock_init.assert_not_called()

    def test_extract_modal_params_without_dm_uses_defaults_and_pixel_pupil(self):
        analyzer = self._make_analyzer(
            'extract_no_dm',
            params={
                'main': {'class': 'SimulParams', 'pixel_pupil': 16, 'pixel_pitch': 1.0},
                'prop': {'class': 'AtmoPropagation'}
            }
        )

        modal_params = analyzer._extract_modal_params_from_dm()
        self.assertEqual(modal_params['type_str'], 'zernike')
        self.assertEqual(modal_params['nmodes'], 100)
        self.assertEqual(modal_params['npixels'], 16)

    def test_extract_modal_params_merges_ifunc_ref_parameters(self):
        analyzer = self._make_analyzer(
            'extract_ifunc_ref_merge',
            params={
                'main': {'class': 'SimulParams', 'pixel_pupil': 12, 'pixel_pitch': 1.0},
                'prop': {'class': 'AtmoPropagation'},
                'dm': {
                    'class': 'DM',
                    'height': 0,
                    'ifunc_ref': 'my_ifunc',
                },
                'my_ifunc': {
                    'class': 'IFunc',
                    'nmodes': 42,
                    'type_str': 'zernike',
                    'obsratio': 0.2,
                }
            }
        )

        modal_params = analyzer._extract_modal_params_from_dm()
        self.assertEqual(modal_params['ifunc_ref'], 'my_ifunc')
        self.assertEqual(modal_params['nmodes'], 42)
        self.assertEqual(modal_params['type_str'], 'zernike')
        self.assertEqual(modal_params['obsratio'], 0.2)
