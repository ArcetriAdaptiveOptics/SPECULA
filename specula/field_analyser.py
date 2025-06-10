import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from astropy.io import fits
from copy import deepcopy

from specula.simul import Simul
from specula.processing_objects.psf import PSF

class FieldAnalyser:
    """
    Class to analyze field PSF, modal analysis, and phase cubes
    for a given tracking number in the Specula framework.
    This class replicates the functionality of the previous compute_off_axis_psf,
    compute_off_axis_modal_analysis, and compute_off_axis_cube methods,
    providing a structured way to handle field sources and their analysis.
    Attributes:
        data_dir (str): Directory containing tracking number data.
        tracking_number (str): The tracking number for the analysis.
        polar_coordinates (np.ndarray): Polar coordinates of field sources.
        wavelength_nm (float): Wavelength in nanometers.
        start_time (float): Start time for the analysis.
        end_time (Optional[float]): End time for the analysis, if applicable.
        verbose (bool): Whether to print verbose output during processing.
    """

    def __init__(self,
                 data_dir: str,
                 tracking_number: str,
                 polar_coordinates: np.ndarray,
                 wavelength_nm: float = 750.0,
                 start_time: float = 0.1,
                 end_time: Optional[float] = None,
                 verbose: bool = False):

        self.data_dir = Path(data_dir)
        self.tracking_number = tracking_number
        self.polar_coordinates = np.atleast_2d(polar_coordinates)
        self.wavelength_nm = wavelength_nm
        self.start_time = start_time
        self.end_time = end_time
        self.verbose = verbose

        # Loaded parameters
        self.params = None
        self.sources = []
        self.distances = []

        # Paths - modify to create separate directories
        self.tn_dir = self.data_dir / tracking_number
        self.base_output_dir = self.data_dir  # Base directory for analysis results

        # Create separate directories for each analysis type
        self.psf_output_dir = self.base_output_dir / f"{tracking_number}_PSF"
        self.modal_output_dir = self.base_output_dir / f"{tracking_number}_MA"
        self.cube_output_dir = self.base_output_dir / f"{tracking_number}_CUBE"

        # Verify that the tracking number directory exists
        if not self.tn_dir.exists():
            raise FileNotFoundError(f"Tracking number directory not found: {self.tn_dir}")

        self._load_simulation_params()
        self._setup_sources()

    def _load_simulation_params(self):
        """Load simulation parameters from tracking number"""
        params_file = self.tn_dir / "params.yml"
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        with open(params_file, 'r') as f:
            self.params = yaml.safe_load(f)

    def _setup_sources(self):
        """Setup field sources"""
        if self.polar_coordinates.shape[0] == 2:
            # Format: [[r1, r2, ...], [theta1, theta2, ...]]
            n_sources = self.polar_coordinates.shape[1]
            coords = self.polar_coordinates.T
        else:
            # Format: [[r1, theta1], [r2, theta2], ...]
            n_sources = self.polar_coordinates.shape[0]
            coords = self.polar_coordinates

        for i, (r, theta) in enumerate(coords):
            source_dict = {
                'polar_coordinates': [float(r), float(theta)],
                'height': float('inf'),  # star
                'magnitude': 8,
                'wavelengthInNm': self.wavelength_nm
            }
            self.sources.append(source_dict)
            self.distances.append(r)

    def _get_source_coordinates(self, source_idx: int) -> Tuple[float, float]:
        """
        Get polar coordinates (r, theta) for a specific source index

        Args:
            source_idx: Index of the source
            
        Returns:
            Tuple of (r, theta) in polar coordinates
        """
        if len(self.polar_coordinates.shape) == 2:
            if self.polar_coordinates.shape[0] == 2:
                # Format: [[r1, r2, ...], [theta1, theta2, ...]]
                r, theta = self.polar_coordinates[0, source_idx], self.polar_coordinates[1, source_idx]
            else:
                # Format: [[r1, theta1], [r2, theta2], ...]
                r, theta = self.polar_coordinates[source_idx, 0], self.polar_coordinates[source_idx, 1]
        else:
            # 1D array case
            r, theta = self.polar_coordinates[source_idx]

        return float(r), float(theta)

    def _get_psf_filenames(self, source_idx: int) -> Tuple[str, str]:
        """
        Generate PSF and SR filenames for a given source

        Args:
            source_idx: Index of the source
            pixel_size_mas: PSF pixel size in milliarcseconds
            
        Returns:
            Tuple of (psf_filename, sr_filename) without .fits extension
        """
        r, theta = self._get_source_coordinates(source_idx)
        psf_filename = f"psf_r{r:.1f}t{theta:.1f}_pix{self.psf_pixel_size_mas:.2f}mas_wl{self.wavelength_nm:.0f}nm"
        sr_filename = f"sr_r{r:.1f}t{theta:.1f}_pix{self.psf_pixel_size_mas:.2f}mas_wl{self.wavelength_nm:.0f}nm"
        return psf_filename, sr_filename

    def _get_modal_filename(self, source_idx: int, modal_params: dict) -> str:
        """
        Generate modal analysis filename for a given source
        
        Args:
            source_idx: Index of the source
            modal_params: Modal analysis parameters
            
        Returns:
            Filename without .fits extension
        """
        r, theta = self._get_source_coordinates(source_idx)
        modal_filename = f"modal_r{r:.1f}t{theta:.1f}"

        # Add modal parameters to filename
        if 'nmodes' in modal_params:
            modal_filename += f"_nmodes{modal_params['nmodes']}"
        elif 'nzern' in modal_params:
            modal_filename += f"_nzern{modal_params['nzern']}"

        if 'type_str' in modal_params:
            modal_filename += f"_{modal_params['type_str']}"

        if 'obsratio' in modal_params:
            modal_filename += f"_obs{modal_params['obsratio']:.2f}"

        return modal_filename

    def _get_cube_filename(self, source_idx: int) -> str:
        """
        Generate phase cube filename for a given source

        Args:
            source_idx: Index of the source

        Returns:
            Filename without .fits extension
        """
        r, theta = self._get_source_coordinates(source_idx)
        cube_filename = f"cube_r{r:.1f}t{theta:.1f}_wl{self.wavelength_nm:.0f}nm"
        return cube_filename

    def _build_replay_params_from_datastore(self) -> dict:
        """
        Build replay params using the existing build_replay mechanism in Simul
        but with modified DataStore input_list containing only DM commands
        """
        if self.params is None:
            raise RuntimeError("Simulation parameters not loaded")

        # Create modified params with reduced DataStore input_list
        modified_params = deepcopy(self.params)

        # Find and modify DataStore object
        datastore_obj = None
        datastore_key = None

        for key, config in modified_params.items():
            if isinstance(config, dict) and config.get('class') == 'DataStore':
                datastore_obj = config
                datastore_key = key
                break

        if datastore_obj is None:
            raise RuntimeError("No DataStore object found in original parameters")

        # Extract only DM command inputs from original input_list
        original_input_list = datastore_obj.get('inputs', {}).get('input_list', [])
        dm_command_inputs = []

        for input_ref in original_input_list:
            if isinstance(input_ref, str):
                # Keep only DM command references
                # Format: 'comm-control.out_comm' or 'comm-integrator.out_comm'
                if 'comm-' in input_ref and ('control.' in input_ref or 'integrator.' in input_ref):
                    dm_command_inputs.append(input_ref)
                # Also check for direct DM references
                elif '.out_comm' in input_ref:
                    dm_command_inputs.append(input_ref)

        if not dm_command_inputs:
            # Fallback: look for any command-related outputs
            for input_ref in original_input_list:
                if 'comm' in input_ref.lower():
                    dm_command_inputs.append(input_ref)

        if not dm_command_inputs:
            raise RuntimeError("No DM command inputs found in DataStore configuration")

        # Update DataStore with reduced input_list
        modified_params[datastore_key]['inputs']['input_list'] = dm_command_inputs

        if self.verbose:
            print(f"Original DataStore input_list: {original_input_list}")
            print(f"Reduced to DM commands only: {dm_command_inputs}")

        # Create Simul instance by bypassing the constructor
        temp_simul = object.__new__(Simul)  # Create instance without calling __init__

        # Initialize essential attributes
        temp_simul.params = modified_params
        temp_simul.verbose = self.verbose
        temp_simul.overrides = []
        temp_simul.diagram = False
        temp_simul.diagram_title = None
        temp_simul.diagram_filename = None
        temp_simul.objs = {}
        temp_simul.replay_params = {}

        # Build objects and connections (needed for build_replay)
        temp_simul.build_replay(modified_params)

        # FIX: Update DataSource store_dir to point to correct tracking number directory
        if 'data_source' in temp_simul.replay_params:
            temp_simul.replay_params['data_source']['store_dir'] = str(self.tn_dir)
            if self.verbose:
                print(f"Updated DataSource store_dir to: {self.tn_dir}")

        return temp_simul.replay_params

    def _build_replay_params_psf(self) -> dict:
        """
        Build replay_params for field PSF calculation using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        if self.verbose:
            print(f"Base replay_params keys: {list(replay_params.keys())}")

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, [
            'PSF', 'CCD', 'SH', 'ShSlopec', 'ModulatedPyramid',
            'PyrSlopec', 'Modalrec', 'ModalAnalysis'
        ])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Add PSF objects for each field source
        psf_input_list = []
        for i, source_dict in enumerate(self.sources):
            psf_name = f'psf_field_{i}'

            # Build PSF config with pixel_size_mas
            psf_config = {
                'class': 'PSF',
                'simul_params_ref': 'main',
                'wavelengthInNm': self.wavelength_nm,
                'pixel_size_mas': self.psf_pixel_size_mas,
                'start_time': self.start_time,
                'inputs': {
                    'in_ef': f'prop.out_field_source_{i}_ef'
                },
                'outputs': ['out_int_psf', 'out_int_sr']
            }

            replay_params[psf_name] = psf_config

            # Create input_list entries with desired filenames
            psf_filename, sr_filename = self._get_psf_filenames(i)
            psf_input_list.extend([
                f'{psf_filename}-{psf_name}.out_int_psf',
                f'{sr_filename}-{psf_name}.out_int_sr'
            ])

        # Add DataStore to save PSF results
        replay_params['data_store_psf'] = {
            'class': 'DataStore',
            'store_dir': str(self.psf_output_dir),
            'data_format': 'fits',
            'create_tn': False,  # Use existing directory structure
            'inputs': {
                'input_list': psf_input_list
            }
        }

        if self.verbose:
            print(f"Final replay_params keys: {list(replay_params.keys())}")
            print(f"PSF files to be saved: {psf_input_list}")

        return replay_params

    def _build_replay_params_modal(self, modal_params: dict) -> dict:
        """
        Build replay_params for field modal analysis using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, [
            'PSF', 'CCD', 'SH', 'ShSlopec', 'ModulatedPyramid',
            'PyrSlopec', 'Modalrec', 'ModalAnalysis'
        ])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Create shared IFunc/IFuncInv object if needed
        shared_ifunc_ref = None
        shared_ifunc_inv_ref = None

        if 'ifunc' in modal_params and modal_params['ifunc'] is not None:
            # Create shared IFunc object
            ifunc_config = {'class': 'IFunc'}
            ifunc_param_mapping = {
                'type_str': 'type_str', 'nmodes': 'nmodes', 'nzern': 'nzern',
                'obsratio': 'obsratio', 'diaratio': 'diaratio', 'npixels': 'npixels',
                'start_mode': 'start_mode', 'idx_modes': 'idx_modes',
                'mask': 'mask', 'tag': 'tag'
            }

            ifunc_source = modal_params['ifunc']
            for ifunc_param, config_param in ifunc_param_mapping.items():
                if ifunc_param in ifunc_source:
                    ifunc_config[config_param] = ifunc_source[ifunc_param]

            if 'npixels' not in ifunc_config:
                ifunc_config['npixels'] = replay_params['main']['pixel_pupil']

            replay_params['modal_analysis_ifunc'] = ifunc_config
            shared_ifunc_ref = 'modal_analysis_ifunc'

        elif 'ifunc_inv' in modal_params and modal_params['ifunc_inv'] is not None:
            # Create shared IFuncInv object
            ifunc_inv_config = {'class': 'IFuncInv'}
            ifunc_inv_param_mapping = {'tag': 'tag', 'mask': 'mask'}

            ifunc_inv_source = modal_params['ifunc_inv']
            for ifunc_inv_param, config_param in ifunc_inv_param_mapping.items():
                if ifunc_inv_param in ifunc_inv_source:
                    ifunc_inv_config[config_param] = ifunc_inv_source[ifunc_inv_param]

            replay_params['modal_analysis_ifunc_inv'] = ifunc_inv_config
            shared_ifunc_inv_ref = 'modal_analysis_ifunc_inv'

        else:
            # Create default IFunc
            ifunc_config = {
                'class': 'IFunc',
                'type_str': modal_params.get('type_str', 'zernike'),
                'nmodes': modal_params.get('nmodes', modal_params.get('nzern', 100)),
                'npixels': modal_params.get('npixels', replay_params['main']['pixel_pupil'])
            }

            for param in ['obsratio', 'diaratio', 'start_mode', 'idx_modes']:
                if param in modal_params:
                    ifunc_config[param] = modal_params[param]

            replay_params['modal_analysis_ifunc'] = ifunc_config
            shared_ifunc_ref = 'modal_analysis_ifunc'

        # Add ModalAnalysis for each source and build input_list
        modal_input_list = []
        for i, source_dict in enumerate(self.sources):
            modal_name = f'modal_analysis_{i}'
            modal_config = {'class': 'ModalAnalysis'}

            if shared_ifunc_ref:
                modal_config['ifunc_ref'] = shared_ifunc_ref
            elif shared_ifunc_inv_ref:
                modal_config['ifunc_inv_ref'] = shared_ifunc_inv_ref

            # Add ModalAnalysis-specific parameters
            modal_specific_params = ['dorms', 'wavelengthInNm']
            for param in modal_specific_params:
                if param in modal_params:
                    modal_config[param] = modal_params[param]

            modal_config['inputs'] = {'in_ef': f'prop.out_field_source_{i}_ef'}
            modal_config['outputs'] = ['out_modes']

            replay_params[modal_name] = modal_config

            # Create filename for this source
            modal_filename = self._get_modal_filename(i, modal_params)
            modal_input_list.append(f'{modal_filename}-{modal_name}.out_modes')

        # Add DataStore to save results
        replay_params['data_store_modal'] = {
            'class': 'DataStore',
            'store_dir': str(self.modal_output_dir),
            'data_format': 'fits',
            'create_tn': False,  # Use existing directory structure
            'inputs': {
                'input_list': modal_input_list
            }
        }

        if self.verbose:
            print(f"Modal files to be saved: {modal_input_list}")

        return replay_params

    def _build_replay_params_cube(self) -> dict:
        """
        Build replay_params for field phase cubes using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, [
            'PSF', 'CCD', 'SH', 'ShSlopec', 'ModulatedPyramid',
            'PyrSlopec', 'Modalrec', 'ModalAnalysis'
        ])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Build input_list for phase cubes
        cube_input_list = []
        for i in range(len(self.sources)):
            cube_filename = self._get_cube_filename(i)
            cube_input_list.append(f'{cube_filename}-prop.out_field_source_{i}_ef')

        # Add DataStore to save phase cubes
        replay_params['data_store_cube'] = {
            'class': 'DataStore',
            'store_dir': str(self.cube_output_dir),
            'data_format': 'fits',
            'create_tn': False,  # Use existing directory structure
            'inputs': {
                'input_list': cube_input_list
            }
        }

        if self.verbose:
            print(f"Cube files to be saved: {cube_input_list}")

        return replay_params

    def _add_field_sources_to_params(self, replay_params: dict):
        """
        Add field sources and update propagation object
        Now works with replay_params which already has proper DM inputs
        """
        # Find the propagation object
        prop_key = None
        for key, config in replay_params.items():
            if isinstance(config, dict) and config.get('class') == 'AtmoPropagation':
                prop_key = key
                break

        if prop_key is None:
            available_objects = list(replay_params.keys())
            raise KeyError(f"AtmoPropagation object not found in replay_params. "
                        f"Available objects: {available_objects}")

        if self.verbose:
            print(f"Found propagation object: '{prop_key}'")

        # Find the position of the propagation object
        keys_list = list(replay_params.keys())
        prop_index = keys_list.index(prop_key)

        # Create a new ordered dictionary
        new_params = {}

        # Add all items before the propagation object
        for key in keys_list[:prop_index]:
            new_params[key] = replay_params[key]

        # Add field sources
        for i, source_dict in enumerate(self.sources):
            source_name = f'field_source_{i}'
            new_params[source_name] = {
                'class': 'Source',
                'polar_coordinates': source_dict['polar_coordinates'],
                'magnitude': source_dict['magnitude'],
                'wavelengthInNm': source_dict['wavelengthInNm'],
                'height': source_dict['height']
            }

        # Add propagation object and remaining items
        for key in keys_list[prop_index:]:
            new_params[key] = replay_params[key]

        # Update propagation object to include all sources
        source_refs = []

        # Check if the original propagation object has source references
        original_sources = new_params[prop_key].get('source_dict_ref', [])
        if original_sources:
            source_refs.extend(original_sources)

        # Add field sources
        source_refs.extend([f'field_source_{i}' for i in range(len(self.sources))])
        new_params[prop_key]['source_dict_ref'] = source_refs

        # Update outputs to include field sources
        original_outputs = new_params[prop_key].get('outputs', [])
        output_list = list(original_outputs)

        for i in range(len(self.sources)):
            output_list.append(f'out_field_source_{i}_ef')
        new_params[prop_key]['outputs'] = output_list

        if self.verbose:
            print(f"Updated propagation object '{prop_key}':")
            print(f"  Sources: {source_refs}")
            print(f"  Outputs: {output_list}")

        # Replace the original dictionary content
        replay_params.clear()
        replay_params.update(new_params)

    def _remove_conflicting_objects(self, replay_params: dict, classes_to_remove: list):
        """
        Remove objects that would conflict with the analysis based on their class
        """
        objects_to_remove = []

        for obj_name, obj_config in replay_params.items():
            if isinstance(obj_config, dict) and 'class' in obj_config:
                if obj_config['class'] in classes_to_remove:
                    objects_to_remove.append(obj_name)

        for obj_name in objects_to_remove:
            del replay_params[obj_name]
            if self.verbose:
                print(f"Removed conflicting object: {obj_name} (class: {replay_params.get(obj_name, {}).get('class', 'unknown')})")

    def _run_simulation_with_params(self, params_dict: dict, temp_filename: str, output_dir: Path) -> Simul:
        """
        Common simulation execution logic
        """

        # Save temporary parameters
        temp_params_file = output_dir / temp_filename
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(temp_params_file, 'w') as f:
            yaml.dump(params_dict, f, default_flow_style=False, sort_keys=False)

        if self.verbose:
            print(f"Computing simulation using: {temp_params_file}")

        # Execute simulation with replay
        try:
            simul = Simul(str(temp_params_file))
            simul.run()
            return simul
        except Exception as e:
            print(f"Simulation failed: {e}")
            print(f"Temp params file saved for debugging: {temp_params_file}")
            raise
        finally:
            # Cleanup temporary file (comment out for debugging)
            # temp_params_file.unlink()
            pass

    def compute_field_psf(self,
                        psf_sampling: Optional[float] = None, 
                        psf_pixel_size_mas: Optional[float] = None,
                        force_recompute: bool = False) -> Dict:
        """
        Calculate field PSF using SPECULA's replay system
        
        Args:
            psf_sampling: PSF sampling factor (alternative to psf_pixel_size_mas)
            psf_pixel_size_mas: Desired PSF pixel size in milliarcseconds (alternative to psf_sampling)
            force_recompute: Force recomputation even if files exist
            
        Note:
            Either psf_sampling or psf_pixel_size_mas must be specified, but not both.
        """

        # Validate input parameters
        if psf_sampling is not None and psf_pixel_size_mas is not None:
            raise ValueError("Cannot specify both psf_sampling and psf_pixel_size_mas. Choose one.")

        if psf_sampling is None and psf_pixel_size_mas is None:
            psf_sampling = 7.0

        # Pupil parameters
        pixel_pitch = self.params['main']['pixel_pitch']
        pixel_pupil = self.params['main']['pixel_pupil']

        # Calculate the pixel size of the PSF in mas in both cases
        if psf_pixel_size_mas is not None:
            # compute the actual pixel size based on the provided value
            psf_sampling = PSF.calc_psf_sampling(
                pixel_pupil, 
                pixel_pitch, 
                self.wavelength_nm, 
                psf_pixel_size_mas
            )

        self.psf_sampling = psf_sampling
        self.psf_pixel_size_mas = (self.wavelength_nm * 1e-9 / (pixel_pupil*pixel_pitch) * 3600 * 180 / np.pi) \
                                    * 1000 / psf_sampling

        # Check if all individual PSF files exist
        all_exist = True
        if not force_recompute:
            for i in range(len(self.sources)):
                psf_filename, sr_filename = self._get_psf_filenames(i)
                psf_path = self.psf_output_dir / f"{psf_filename}.fits"
                sr_path = self.psf_output_dir / f"{sr_filename}.fits"

                if not psf_path.exists() or not sr_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing PSF results from: {self.psf_output_dir}")
                return self._load_psf_results()

        if self.verbose:
            print(f"Computing field PSF for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_psf()
        simul = self._run_simulation_with_params(replay_params, "temp_psf_replay_params.yml", self.psf_output_dir)

        if self.verbose:
            print(f"Actual PSF pixel size: {self.psf_pixel_size_mas:.2f} mas")

        # Extract results from DataStore (files are automatically saved)
        results = self._load_psf_results()

        return results

    def compute_modal_analysis(self, modal_params: Optional[Dict] = None, force_recompute: bool = False) -> Dict:
        """
        Calculate field modal analysis using replay system
        
        Args:
            modal_params: Dictionary with ModalAnalysis configuration parameters.
                        Can include any parameter accepted by ModalAnalysis class:
                        - type_str: 'zernike', 'kl', 'mixed', 'zonal' (if ifunc/ifunc_inv not provided)
                        - nmodes/nzern: number of modes (for type_str-based analysis)
                        - obsratio, diaratio: pupil parameters
                        - ifunc: pre-computed IFunc object
                        - ifunc_inv: pre-computed IFuncInv object
                        - npixels: override default pupil pixels
                        - mask: custom mask
                        - dorms: compute RMS flag
                        And any other ModalAnalysis parameter
                        If None, will try to extract from DM parameters if height=0,
                        otherwise defaults to {'type_str': 'zernike', 'nmodes': 100}
            force_recompute: Force recomputation even if files exist
        """
        if modal_params is None:
            modal_params = self._extract_modal_params_from_dm()

        # Check if all individual modal files exist
        all_exist = True
        if not force_recompute:
            for i in range(len(self.sources)):
                modal_filename = self._get_modal_filename(i, modal_params)
                modal_path = self.modal_output_dir / f"{modal_filename}.fits"

                if not modal_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing modal analysis from: {self.modal_output_dir}")
                return self._load_modal_results( modal_params)

        if self.verbose:
            print(f"Computing field modal analysis for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_modal(modal_params)
        simul = self._run_simulation_with_params(replay_params, "temp_modal_replay_params.yml", self.modal_output_dir)

        # Extract results from DataStore (files are automatically saved)
        results = self._load_modal_results(modal_params)

        return results

    def compute_phase_cube(self, force_recompute: bool = False) -> Dict:
        """Calculate field phase cubes using replay system"""

        # Check if all individual cube files exist
        all_exist = True
        if not force_recompute:
            for i in range(len(self.sources)):
                cube_filename = self._get_cube_filename(i)
                cube_path = self.cube_output_dir / f"{cube_filename}.fits"
                
                if not cube_path.exists():
                    all_exist = False
                    break

            if all_exist:
                if self.verbose:
                    print(f"Loading existing phase cubes from: {self.cube_output_dir}")
                return self._load_cube_results()

        if self.verbose:
            print(f"Computing field phase cubes for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_cube()
        simul = self._run_simulation_with_params(replay_params, "temp_cube_replay_params.yml", self.cube_output_dir)

        # Extract results from DataStore (files are automatically saved)
        results = self._load_cube_results()

        return results

    def _load_psf_results(self) -> Dict:
        """Extract PSF results from DataStore files"""
        results = {
            'psf_list': [],
            'sr_list': [],
            'distances': self.distances,
            'coordinates': self.polar_coordinates,
            'wavelength_nm': self.wavelength_nm,
            'pixel_size_mas': self.psf_pixel_size_mas,
            'psf_sampling': self.psf_sampling
        }

        # Load PSF and SR data from saved files
        for i in range(len(self.sources)):
            psf_filename, sr_filename = self._get_psf_filenames(i)

            # Load PSF
            psf_path = self.psf_output_dir / f"{psf_filename}.fits"
            with fits.open(psf_path) as hdul:
                results['psf_list'].append(hdul[0].data)

            # Load SR
            sr_path = self.psf_output_dir / f"{sr_filename}.fits"
            with fits.open(sr_path) as hdul:
                results['sr_list'].append(hdul[0].data)

        return results

    def _load_modal_results(self, modal_params: dict) -> Dict:
        """Load existing modal results from DataStore files"""
        results = {
            'modal_coeffs': [],
            'residual_variance': [],
            'residual_average': [],
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm,
            'modal_params': modal_params
        }

        for i in range(len(self.sources)):
            modal_filename = self._get_modal_filename(i, modal_params)
            modal_path = self.modal_output_dir / f"{modal_filename}.fits"

            with fits.open(modal_path) as hdul:
                modal_coeffs = hdul[0].data
                results['modal_coeffs'].append(modal_coeffs)

                # Calculate statistics from time series
                if len(modal_coeffs) > 0:
                    # Filter by time if needed (assuming first dimension is time)
                    results['residual_average'].append(np.mean(modal_coeffs, axis=0))
                    results['residual_variance'].append(np.var(modal_coeffs, axis=0))
                else:
                    results['residual_average'].append(np.zeros(modal_coeffs.shape[1]))
                    results['residual_variance'].append(np.zeros(modal_coeffs.shape[1]))

        return results

    def _load_cube_results(self) -> Dict:
        """Load existing cube results from DataStore files"""
        results = {
            'phase_cubes': [],
            'times': None,
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm
        }

        for i in range(len(self.sources)):
            cube_filename = self._get_cube_filename(i)
            cube_path = self.cube_output_dir / f"{cube_filename}.fits"

            with fits.open(cube_path) as hdul:
                results['phase_cubes'].append(hdul[0].data)

                if results['times'] is None and len(hdul) > 1:
                    results['times'] = hdul[1].data

        return results



    def _extract_modal_params_from_dm(self) -> Dict:
        """
        Extract modal parameters from DM configuration if available and height=0
        """
        # Default fallback parameters
        default_params = {'type_str': 'zernike', 'nmodes': 100}

        if self.params is None:
            if self.verbose:
                print("No simulation parameters loaded, using default modal params")
            return default_params

        # Look for DM configuration in params - prioritize DMs that are actually used
        dm_configs = []

        # First, find which DMs are used in AtmoPropagation
        used_dm_names = set()
        for obj_name, obj_config in self.params.items():
            if isinstance(obj_config, dict) and obj_config.get('class') == 'AtmoPropagation':
                common_layers = obj_config.get('inputs', {}).get('common_layer_list', [])
                for layer_ref in common_layers:
                    if isinstance(layer_ref, str) and '.out_layer' in layer_ref:
                        dm_name = layer_ref.split('.out_layer')[0]
                        used_dm_names.add(dm_name)

        # Find DM configurations, prioritizing used ones
        for obj_name, obj_config in self.params.items():
            if isinstance(obj_config, dict) and obj_config.get('class') == 'DM':
                priority = 0 if obj_name in used_dm_names else 1
                dm_configs.append((priority, obj_name, obj_config))

        # Sort by priority (used DMs first) and take the first one with height=0
        dm_configs.sort(key=lambda x: (x[0], x[1]))

        for priority, dm_name, dm_config in dm_configs:
            dm_height = dm_config.get('height', None)
            if dm_height == 0:
                if self.verbose:
                    print(f"Using DM '{dm_name}' (height=0) for modal parameters")
                return self._extract_params_from_dm_config(dm_config, dm_height)

        if self.verbose:
            if dm_configs:
                print("No DM with height=0 found, using default modal params")
            else:
                print("No DM configuration found in params, using default modal params")

        return default_params

    def _extract_params_from_dm_config(self, dm_config: dict, dm_height: float) -> Dict:
        """Extract parameters from a specific DM configuration"""
        modal_params = {}

        # Map DM parameters to ModalAnalysis parameters
        param_mapping = {
            'type_str': 'type_str',
            'nmodes': 'nmodes', 
            'nzern': 'nzern',
            'obsratio': 'obsratio',
            'diaratio': 'diaratio',
            'npixels': 'npixels',
            'start_mode': 'start_mode',
            'idx_modes': 'idx_modes'
        }

        # Handle IFunc reference - this is the common case
        if 'ifunc_ref' in dm_config:
            ifunc_ref = dm_config['ifunc_ref']
            if ifunc_ref in self.params:
                ifunc_config = self.params[ifunc_ref]
                if isinstance(ifunc_config, dict) and ifunc_config.get('class') == 'IFunc':
                    # Extract IFunc parameters for modal analysis
                    modal_params['ifunc'] = self._extract_ifunc_config_params(ifunc_config)
                    if self.verbose:
                        print(f"Found IFunc reference '{ifunc_ref}' in DM config")
                else:
                    if self.verbose:
                        print(f"Warning: ifunc_ref '{ifunc_ref}' does not point to an IFunc object")
            else:
                if self.verbose:
                    print(f"Warning: ifunc_ref '{ifunc_ref}' not found in params")

        # Handle direct IFunc object (less common but possible)
        elif 'ifunc' in dm_config:
            raise ValueError("This case has not been implemented yet. "
                             "Please provide an IFunc reference instead of a direct object.")

        # Handle saved IFunc objects
        if 'ifunc_object' in dm_config:
            # This is a saved IFunc loaded from file
            modal_params['ifunc'] = {'tag': dm_config['ifunc_object']}

        # Handle M2C information
        if 'm2c_ref' in dm_config:
            m2c_ref = dm_config['m2c_ref']
            if m2c_ref in self.params:
                m2c_config = self.params[m2c_ref]
                if isinstance(m2c_config, dict) and m2c_config.get('class') == 'M2C':
                    # Extract number of modes from M2C configuration if available
                    if 'nmodes' in m2c_config and 'nmodes' not in modal_params:
                        modal_params['nmodes'] = m2c_config['nmodes']
        elif 'm2c' in dm_config:
            raise ValueError("This case has not been implemented yet. "
                             "Please provide an M2C reference instead of a direct object.")

        # Copy standard parameters
        for dm_param, modal_param in param_mapping.items():
            if dm_param in dm_config:
                modal_params[modal_param] = dm_config[dm_param]

        # Ensure defaults
        if 'nmodes' not in modal_params and 'nzern' not in modal_params:
            modal_params['nmodes'] = 100
        if 'type_str' not in modal_params and 'ifunc' not in modal_params:
            modal_params['type_str'] = 'zernike'

        if self.verbose:
            print(f"Extracted modal parameters from DM config: {modal_params}")
            print(f"DM height: {dm_height} (using DM-based modal analysis)")

        return modal_params

    def _extract_ifunc_config_params(self, ifunc_config: dict) -> dict:
        """
        Extract parameters from an IFunc configuration dictionary
        """
        ifunc_params = {}

        # Map IFunc config parameters to ModalAnalysis IFunc parameters
        ifunc_param_mapping = {
            'type_str': 'type_str',
            'nmodes': 'nmodes',
            'nzern': 'nzern', 
            'npixels': 'npixels',
            'obsratio': 'obsratio',
            'diaratio': 'diaratio',
            'start_mode': 'start_mode',
            'idx_modes': 'idx_modes',
            'tag': 'tag'  # For saved IFunc objects
        }

        for ifunc_param, modal_param in ifunc_param_mapping.items():
            if ifunc_param in ifunc_config:
                ifunc_params[modal_param] = ifunc_config[ifunc_param]

        if self.verbose:
            print(f"Extracted IFunc config parameters: {ifunc_params}")

        return ifunc_params

    def _extract_ifunc_params(self, ifunc_obj) -> dict:
        """
        Extract parameters from an existing IFunc object for reconstruction
        """
        ifunc_params = {}

        if hasattr(ifunc_obj, 'influence_function'):
            # Extract shape information
            shape = ifunc_obj.influence_function.shape
            if shape[0] > shape[1]:
                ifunc_params['nmodes'] = shape[1]
                ifunc_params['npixels'] = int(np.sqrt(shape[0]))
            else:
                ifunc_params['nmodes'] = shape[0]
                ifunc_params['npixels'] = int(np.sqrt(shape[1]))

        # Try to infer type from shape or other properties
        # This is a heuristic - you might need to adjust based on your IFunc objects
        if hasattr(ifunc_obj, '_type_str'):
            ifunc_params['type_str'] = ifunc_obj._type_str
        else:
            ifunc_params['type_str'] = 'zernike'  # Default assumption

        return ifunc_params