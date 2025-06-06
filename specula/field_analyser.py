import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import yaml
from astropy.io import fits

from specula.simul import Simul
from specula.data_objects.source import Source
from specula.processing_objects.data_store import DataStore
from specula.processing_objects.data_source import DataSource
from specula.calib_manager import CalibManager

def generate_field_filename(tracking_number: str,
                             analysis_type: str,
                             polar_coordinates: np.ndarray,
                             **kwargs) -> str:
    """
    Generate standardized filename for field analysis
    
    Args:
        tracking_number: Simulation tracking number
        analysis_type: Analysis type ('psf', 'modal', 'cube')
        polar_coordinates: Polar coordinates of sources
        **kwargs: Additional parameters (wavelength, sampling, etc.)
    """
    coords = np.atleast_2d(polar_coordinates)
    if coords.shape[0] == 2:
        coords = coords.T

    coords_str = "_".join([f"r{r:.1f}t{t:.1f}" for r, t in coords])

    filename = f"{tracking_number}_{analysis_type}_{coords_str}"

    # Add specific parameters
    for key, value in kwargs.items():
        if value is not None:
            if key == 'wavelength_nm':
                filename += f"_wl{value:.0f}nm"
            elif key == 'psf_sampling':
                filename += f"_samp{value}"
            elif key == 'start_time':
                filename += f"_t{value:.1f}"
            elif key == 'end_time':
                filename += f"to{value:.1f}"
            else:
                filename += f"_{key}{value}"

    return filename + ".fits"


def check_simulation_data_completeness(tn_dir: Path) -> Dict[str, any]:
    """
    Check if all required data for a tracking number is available

    Returns:
        Dict with status of completeness, missing files, and available data
    """
    status = {
        'complete': True,
        'missing_files': [],
        'available_data': {},
        'dm_count': 0,
        'has_atmosphere': False,
        'dm_commands': False,
        'dm_command_files': []
    }

    # Parameters file
    params_file = tn_dir / "params.yml"
    if not params_file.exists():
        status['complete'] = False
        status['missing_files'].append("params.yml")
        return status

    # Load parameters to analyze configuration
    try:
        with open(params_file, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        status['complete'] = False
        status['missing_files'].append(f"params.yml (invalid YAML: {e})")
        return status

    status['available_data']['params'] = str(params_file)

    # Find DM command files based on configuration
    dm_command_files = _find_dm_command_files(params, tn_dir)

    if len(dm_command_files) == 0:
        status['complete'] = False
        status['missing_files'].append("DM command files (based on DataStore configuration)")
        status['dm_commands'] = False
        status['dm_count'] = 0
    else:
        # Check if all found command files actually exist
        existing_files = []
        for cmd_file in dm_command_files:
            file_path = tn_dir / cmd_file
            if file_path.exists():
                existing_files.append(cmd_file)
            else:
                status['complete'] = False
                status['missing_files'].append(cmd_file)

        status['available_data']['dm_commands'] = existing_files
        status['dm_commands'] = len(existing_files) > 0
        status['dm_count'] = len(existing_files)
        status['dm_command_files'] = existing_files

    return status


def _find_dm_command_files(params: dict, tn_dir: Path) -> List[str]:
    """
    Find DM command files based on configuration analysis
    
    Args:
        params: Loaded simulation parameters
        tn_dir: Tracking number directory
        
    Returns:
        List of DM command filenames that should exist
    """
    dm_command_files = []

    # Step 1: Find AtmoPropagation object and extract DM layers
    atmo_prop_config = None
    for obj_name, obj_config in params.items():
        if isinstance(obj_config, dict) and obj_config.get('class') == 'AtmoPropagation':
            atmo_prop_config = obj_config
            break

    if atmo_prop_config is None:
        return dm_command_files

    # Extract DM layer references from common_layer_list
    dm_refs = []
    common_layers = atmo_prop_config.get('inputs', {}).get('common_layer_list', [])

    for layer_ref in common_layers:
        if isinstance(layer_ref, str) and '.out_layer' in layer_ref:
            # Extract DM object name (e.g., 'dm.out_layer:-1' -> 'dm')
            dm_name = layer_ref.split('.out_layer')[0]
            dm_refs.append(dm_name)
    
    # Step 2: Find DataStore object and extract DM command inputs
    datastore_config = None
    for obj_name, obj_config in params.items():
        if isinstance(obj_config, dict) and obj_config.get('class') == 'DataStore':
            datastore_config = obj_config
            break

    if datastore_config is None:
        return dm_command_files

    # Extract command references from input_list
    input_list = datastore_config.get('inputs', {}).get('input_list', [])

    for input_ref in input_list:
        if isinstance(input_ref, str):
            # Check if this input corresponds to a DM command
            # Format: 'filename-source.output' or 'filename-obj.out_comm'
            if 'out_comm' in input_ref or any(dm_ref in input_ref for dm_ref in dm_refs):
                # Extract filename part (before the '-')
                if '-' in input_ref:
                    filename = input_ref.split('-')[0] + '.fits'
                    if filename not in dm_command_files:
                        dm_command_files.append(filename)

    # Step 3: Fallback - look for standard DM command patterns if nothing found
    if len(dm_command_files) == 0 and len(dm_refs) > 0:
        # Try standard naming patterns
        standard_patterns = ['comm.fits']
        for i, dm_ref in enumerate(dm_refs, 1):
            standard_patterns.append(f'comm{i}.fits')
            standard_patterns.append(f'{dm_ref}_comm.fits')

        for pattern in standard_patterns:
            if (tn_dir / pattern).exists():
                dm_command_files.append(pattern)

    return dm_command_files


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
        gpu (bool): Whether to use GPU for computations.
        verbose (bool): Whether to print verbose output during processing.
    """

    def __init__(self,
                 data_dir: str,
                 tracking_number: str,
                 polar_coordinates: np.ndarray,
                 wavelength_nm: float = 750.0,
                 start_time: float = 0.1,
                 end_time: Optional[float] = None,
                 gpu: bool = True,
                 verbose: bool = False):

        self.data_dir = Path(data_dir)
        self.tracking_number = tracking_number
        self.polar_coordinates = np.atleast_2d(polar_coordinates)
        self.wavelength_nm = wavelength_nm
        self.start_time = start_time
        self.end_time = end_time
        self.gpu = gpu
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

    def check_required_data(self) -> Dict[str, any]:
        """
        Verify that all necessary data is present for replay
        """
        return check_simulation_data_completeness(self.tn_dir)

    def _get_analysis_filename(self, analysis_type: str, source_idx: int, **kwargs) -> str:
        """Generate filename for analysis results"""
        # Single source filename
        base_name = f"{self.tracking_number}_{analysis_type}"

        # Add coordinate info
        r, theta = self._get_source_coordinates(source_idx)
        base_name += f"_r{r:.1f}t{theta:.1f}"

        # Add specific parameters
        if 'psf_sampling' in kwargs:
            base_name += f"_samp{kwargs['psf_sampling']}"
        if 'wavelength_nm' in kwargs:
            base_name += f"_wl{kwargs['wavelength_nm']:.0f}nm"

        # Add modal analysis specific parameters
        if analysis_type == 'modal':
            if 'nmodes' in kwargs:
                base_name += f"_nmodes{kwargs['nmodes']}"
            elif 'nzern' in kwargs:
                base_name += f"_nzern{kwargs['nzern']}"

            if 'type_str' in kwargs:
                base_name += f"_{kwargs['type_str']}"

            # Add other relevant modal parameters
            if 'obsratio' in kwargs:
                base_name += f"_obs{kwargs['obsratio']:.2f}"
            if 'diaratio' in kwargs:
                base_name += f"_dia{kwargs['diaratio']:.2f}"

        return base_name + ".fits"


    def _build_replay_params_psf(self, psf_sampling: int = 7) -> dict:
        """
        Modify replay_params for field PSF calculation
        """
        # Load existing replay_params
        replay_params_file = self.tn_dir / "replay_params.yml"
        if not replay_params_file.exists():
            raise FileNotFoundError(f"Replay params not found: {replay_params_file}")

        with open(replay_params_file, 'r') as f:
            replay_params = yaml.safe_load(f)

        if self.verbose:
            print(f"Original replay_params keys: {list(replay_params.keys())}")

        # Debug: Check if main object exists
        if 'main' not in replay_params:
            raise KeyError("'main' object not found in original replay_params.yml")

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, ['PSF','CCD','SH','ShSlopec','ModulatedPyramid',
                                                         'PyrSlopec','Modalrec','ModalAnalysis','DataStore'])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Add PSF objects for each field source
        for i, source_dict in enumerate(self.sources):
            psf_name = f'psf_field_{i}'
            replay_params[psf_name] = {
                'class': 'PSF',
                'wavelengthInNm': self.wavelength_nm,
                'nd': psf_sampling,
                'start_time': self.start_time,
                'inputs': {
                    'in_ef': f'prop.out_field_source_{i}_ef'
                },
                'outputs': ['out_psf', 'out_sr']
            }

        if self.verbose:
            print(f"Final replay_params keys: {list(replay_params.keys())}")

        return replay_params

    def _build_replay_params_modal(self, modal_params: dict) -> dict:
        """
        Modify replay_params for field modal analysis
        """
        replay_params_file = self.tn_dir / "replay_params.yml"
        with open(replay_params_file, 'r') as f:
            replay_params = yaml.safe_load(f)

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, ['PSF','CCD','SH','ShSlopec','ModulatedPyramid',
                                                        'PyrSlopec','Modalrec','ModalAnalysis','DataStore'])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Create shared IFunc/IFuncInv object if needed
        shared_ifunc_ref = None
        shared_ifunc_inv_ref = None

        if 'ifunc' in modal_params and modal_params['ifunc'] is not None:
            # Create shared IFunc object using simple parameter mapping
            ifunc_config = {'class': 'IFunc'}

            # Map parameters for IFunc
            ifunc_param_mapping = {
                'type_str': 'type_str',
                'nmodes': 'nmodes', 
                'nzern': 'nzern',
                'obsratio': 'obsratio',
                'diaratio': 'diaratio',
                'npixels': 'npixels',
                'start_mode': 'start_mode',
                'idx_modes': 'idx_modes',
                'mask': 'mask',
                'tag': 'tag'  # For saved IFunc objects
            }

            # Extract IFunc parameters from modal_params['ifunc']
            ifunc_source = modal_params['ifunc']
            for ifunc_param, config_param in ifunc_param_mapping.items():
                if ifunc_param in ifunc_source:
                    ifunc_config[config_param] = ifunc_source[ifunc_param]

            # Set default npixels if not provided
            if 'npixels' not in ifunc_config:
                ifunc_config['npixels'] = replay_params['main']['pixel_pupil']

            replay_params['modal_analysis_ifunc'] = ifunc_config
            shared_ifunc_ref = 'modal_analysis_ifunc'

        elif 'ifunc_inv' in modal_params and modal_params['ifunc_inv'] is not None:
            # Create shared IFuncInv object using simple parameter mapping
            ifunc_inv_config = {'class': 'IFuncInv'}

            # Map parameters for IFuncInv
            ifunc_inv_param_mapping = {
                'tag': 'tag',  # IFuncInv is typically loaded from saved file
                'mask': 'mask'
            }

            # Extract IFuncInv parameters from modal_params['ifunc_inv']
            ifunc_inv_source = modal_params['ifunc_inv']
            for ifunc_inv_param, config_param in ifunc_inv_param_mapping.items():
                if ifunc_inv_param in ifunc_inv_source:
                    ifunc_inv_config[config_param] = ifunc_inv_source[ifunc_inv_param]

            replay_params['modal_analysis_ifunc_inv'] = ifunc_inv_config
            shared_ifunc_inv_ref = 'modal_analysis_ifunc_inv'

        else:
            # No ifunc or ifunc_inv provided - create default IFunc
            ifunc_config = {
                'class': 'IFunc',
                'type_str': modal_params.get('type_str', 'zernike'),
                'nmodes': modal_params.get('nmodes', modal_params.get('nzern', 100)),
                'npixels': modal_params.get('npixels', replay_params['main']['pixel_pupil'])
            }

            # Add optional parameters if present
            for param in ['obsratio', 'diaratio', 'start_mode', 'idx_modes']:
                if param in modal_params:
                    ifunc_config[param] = modal_params[param]

            replay_params['modal_analysis_ifunc'] = ifunc_config
            shared_ifunc_ref = 'modal_analysis_ifunc'

        # Add ModalAnalysis for each source
        for i, source_dict in enumerate(self.sources):
            modal_name = f'modal_analysis_{i}'

            # Start with base configuration (don't copy all modal_params)
            modal_config = {
                'class': 'ModalAnalysis'
            }

            # Always use shared references - one will always exist now
            if shared_ifunc_ref:
                modal_config['ifunc_ref'] = shared_ifunc_ref
            elif shared_ifunc_inv_ref:
                modal_config['ifunc_inv_ref'] = shared_ifunc_inv_ref

            # Add optional ModalAnalysis-specific parameters that don't go in IFunc
            modal_specific_params = ['dorms', 'wavelengthInNm']  # Parameters specific to ModalAnalysis
            for param in modal_specific_params:
                if param in modal_params:
                    modal_config[param] = modal_params[param]

            # Always set inputs and outputs (these are not user-configurable)
            modal_config['inputs'] = {
                'in_ef': f'prop.out_field_source_{i}_ef'
            }
            modal_config['outputs'] = ['out_modes']

            replay_params[modal_name] = modal_config

        # Add DataStore to save results
        input_list = []
        for i in range(len(self.sources)):
            input_list.append(f'modal_res_{i}-modal_analysis_{i}.out_modes')

        replay_params['data_store_modal'] = {
            'class': 'DataStore', 
            'store_dir': str(self.modal_output_dir),
            'data_format': 'fits',
            'save_on_disk': False,
            'inputs': {
                'input_list': input_list
            }
        }

        return replay_params


    def _build_replay_params_cube(self) -> dict:
        """
        Modify replay_params for field phase cubes
        """
        replay_params_file = self.tn_dir / "replay_params.yml"
        with open(replay_params_file, 'r') as f:
            replay_params = yaml.safe_load(f)

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params, ['PSF','CCD','SH','ShSlopec','ModulatedPyramid',
                                                        'PyrSlopec','Modalrec','ModalAnalysis','DataStore'])

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Add DataStore to save phase cubes
        input_list = []
        for i in range(len(self.sources)):
            input_list.append(f'phase_cube_{i}-prop.out_field_source_{i}_ef')

        replay_params['data_store_cube'] = {
            'class': 'DataStore',
            'store_dir': str(self.cube_output_dir),  # Fixed: use cube_output_dir
            'data_format': 'fits',  # Add explicit format
            'save_on_disk': False,  # Do not save params
            'inputs': {
                'input_list': input_list
            }
        }

        return replay_params

    def _add_field_sources_to_params(self, replay_params: dict):
        """
        Add field sources and update propagation object
        Common functionality for all analysis types
        """
        # Find the position of 'prop' in the dictionary
        keys_list = list(replay_params.keys())

        if 'prop' in keys_list:
            prop_index = keys_list.index('prop')

            # Create a new ordered dictionary
            new_params = {}

            # Add all items before 'prop'
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

            # Add 'prop' and remaining items
            for key in keys_list[prop_index:]:
                new_params[key] = replay_params[key]

            # Update propagation object to include all sources
            source_refs = []
            source_refs.extend([f'field_source_{i}' for i in range(len(self.sources))])
            new_params['prop']['source_dict_ref'] = source_refs

            output_list = []
            for i in range(len(self.sources)):
                output_list.append(f'out_field_source_{i}_ef')
            new_params['prop']['outputs'] = output_list

            # Replace the original dictionary content
            replay_params.clear()
            replay_params.update(new_params)

        else:
            raise KeyError("'prop' object not found in original replay_params.yml")

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

    def compute_field_psf(self, psf_sampling: int = 7, save_results: bool = True, force_recompute: bool = False) -> Dict:
        """Calculate field PSF using SPECULA's replay system"""

        # Check if all individual PSF files exist
        all_exist = True
        for i in range(len(self.sources)):
            output_file = self.psf_output_dir / self._get_analysis_filename("psf", source_idx=i, psf_sampling=psf_sampling, wavelength_nm=self.wavelength_nm)
            if not output_file.exists():
                all_exist = False
                break

        if not force_recompute and all_exist:
            if self.verbose:
                print(f"Loading existing PSF results from: {self.psf_output_dir}")
            return self._load_psf_results(self.psf_output_dir, psf_sampling)

        # Verify necessary data
        data_status = self.check_required_data()
        if not data_status['dm_commands']:
            raise RuntimeError("DM command data not found - cannot compute PSF")

        if self.verbose:
            print(f"Computing field PSF for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_psf(psf_sampling)
        simul = self._run_simulation_with_params(replay_params, "temp_psf_replay_params.yml", self.psf_output_dir)

        # Extract and save results
        results = self._extract_psf_results_from_objects(simul, psf_sampling)
        if save_results:
            self._save_psf_results(results, psf_sampling)

        return results

    def compute_modal_analysis(self, modal_params: Optional[Dict] = None, save_results: bool = True, force_recompute: bool = False) -> Dict:
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
            save_results: Whether to save results to disk
            force_recompute: Force recomputation even if files exist
        """
        if modal_params is None:
            modal_params = self._extract_modal_params_from_dm()

        # Check if all individual modal files exist
        all_exist = True
        for i in range(len(self.sources)):
            output_file = self.modal_output_dir / self._get_analysis_filename("modal", source_idx=i, **modal_params)
            if not output_file.exists():
                all_exist = False
                break

        if not force_recompute and all_exist:
            if self.verbose:
                print(f"Loading existing modal analysis from: {self.modal_output_dir}")
            return self._load_modal_results(self.modal_output_dir, modal_params)

        if self.verbose:
            print(f"Computing field modal analysis for {len(self.sources)} sources...")
            print(f"Modal parameters: {modal_params}")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_modal(modal_params)
        simul = self._run_simulation_with_params(replay_params, "temp_modal_replay_params.yml", self.modal_output_dir)

        # Extract and save results
        results = self._extract_modal_results_from_datastore(simul, modal_params)
        if save_results:
            self._save_modal_results(results, modal_params)

        return results

    def compute_phase_cube(self, save_results: bool = True, force_recompute: bool = False) -> Dict:
        """Calculate field phase cubes using replay system"""

        # Check if all individual cube files exist
        all_exist = True
        for i in range(len(self.sources)):
            output_file = self.cube_output_dir / self._get_analysis_filename("cube", source_idx=i)
            if not output_file.exists():
                all_exist = False
                break

        if not force_recompute and all_exist:
            if self.verbose:
                print(f"Loading existing phase cubes from: {self.cube_output_dir}")
            return self._load_cube_results(self.cube_output_dir)

        # Verify necessary data
        data_status = self.check_required_data()
        if not data_status['dm_commands']:
            raise RuntimeError("DM command data not found - cannot compute phase cubes")

        if self.verbose:
            print(f"Computing field phase cubes for {len(self.sources)} sources...")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_cube()
        simul = self._run_simulation_with_params(replay_params, "temp_cube_replay_params.yml", self.cube_output_dir)

        # Extract and save results
        results = self._extract_cube_results_from_datastore(simul)
        if save_results:
            self._save_cube_results(results)

        return results

    def _extract_psf_results_from_objects(self, simul: Simul, psf_sampling: int) -> Dict:
        """
        Extract PSF results from simulation PSF objects
        """
        results = {
            'psf_list': [],
            'sr_list': [],
            'distances': self.distances,
            'coordinates': self.polar_coordinates,
            'wavelength_nm': self.wavelength_nm,
            'psf_sampling': psf_sampling,
            'pixel_scale': None  # Calculated later
        }

        # Extract PSF from each PSF object
        for i in range(len(self.sources)):
            psf_obj = simul.objs[f'psf_field_{i}']
            results['psf_list'].append(psf_obj.int_psf.value)
            results['sr_list'].append(psf_obj.int_sr.value)

        # Calculate pixel scale (as in PASSATA)
        main_params = simul.objs['main']
        pixel_pitch = main_params.pixel_pitch
        pixel_pupil = main_params.pixel_pupil

        # scale = wavelength[m] / (diameter[m]) * rad2arcsec * (1/psf_sampling)
        scale = (self.wavelength_nm * 1e-9) / (pixel_pitch * pixel_pupil) * 206264.8 * (1.0 / psf_sampling)
        results['pixel_scale'] = scale

        return results

    def _extract_modal_results_from_datastore(self, simul: Simul, modal_params: dict) -> Dict:
        """
        Extract modal analysis results from DataStore
        """
        data_store = simul.objs['data_store_modal']

        results = {
            'modal_coeffs': [],
            'residual_variance': [],
            'residual_average': [],
            'turbulence_variance': [],  # If available
            'distances': self.distances,
            'coordinates': self.polar_coordinates,
            'modal_params': modal_params,
            'wavelength_nm': self.wavelength_nm
        }

        # Extract modal coefficients for each source
        for i in range(len(self.sources)):
            modal_key = f'modal_res_{i}'
            if modal_key in data_store.storage:
                # Extract time series
                time_series = data_store.storage[modal_key]
                times = np.array(list(time_series.keys()))
                coeffs = np.array(list(time_series.values()))

                results['modal_coeffs'].append(coeffs)

                # Calculate residual variance and average (excluding settling time)
                if self.start_time > 0:
                    valid_idx = times >= self.start_time
                    if self.end_time:
                        valid_idx &= times <= self.end_time
                    valid_coeffs = coeffs[valid_idx]
                else:
                    valid_coeffs = coeffs

                if len(valid_coeffs) > 0:
                    residual_mean = np.mean(valid_coeffs, axis=0)
                    residual_var = np.var(valid_coeffs, axis=0)
                    results['residual_average'].append(residual_mean)
                    results['residual_variance'].append(residual_var)
                else:
                    results['residual_average'].append(np.zeros(coeffs.shape[1]))
                    results['residual_variance'].append(np.zeros(coeffs.shape[1]))

        return results

    def _extract_cube_results_from_datastore(self, simul: Simul) -> Dict:
        """
        Extract phase cubes from DataStore
        """
        data_store = simul.objs['data_store_cube']

        results = {
            'phase_cubes': [],
            'times': None,
            'distances': self.distances,
            'coordinates': self.polar_coordinates,
            'wavelength_nm': self.wavelength_nm
        }

        # Extract phase cubes for each source
        times = None
        for i in range(len(self.sources)):
            cube_key = f'phase_cube_{i}'
            if cube_key in data_store.storage:
                time_series = data_store.storage[cube_key]
                if times is None:
                    times = np.array(list(time_series.keys()))
                    results['times'] = times

                # Create 3D cube: [time, y, x]
                phases = list(time_series.values())
                if len(phases) > 0:
                    # Extract only amplitude and phase from ElectricField
                    if hasattr(phases[0], '__len__') and len(phases[0]) == 2:
                        # If it's [amplitude, phase], take only phase
                        phase_cube = np.array([phase[1] for phase in phases])
                    else:
                        phase_cube = np.array(phases)

                    results['phase_cubes'].append(phase_cube)

        return results

    # Methods to save and load results (using FITS)

    def _save_psf_results(self, results: Dict, psf_sampling: int):
        """Save PSF results as separate files for each source"""
        # Create directory if it doesn't exist
        self.psf_output_dir.mkdir(parents=True, exist_ok=True)

        for i, (psf_data, sr_value) in enumerate(zip(results['psf_list'], results['sr_list'])):
            filename = self.psf_output_dir / self._get_analysis_filename("psf", source_idx=i, psf_sampling=psf_sampling, wavelength_nm=self.wavelength_nm)

            # Create HDU list for this source
            primary_hdu = fits.PrimaryHDU(psf_data)

            # Add header info
            primary_hdu.header['TN'] = self.tracking_number
            primary_hdu.header['SOURCE'] = i
            primary_hdu.header['WAVELNG'] = self.wavelength_nm
            primary_hdu.header['TSTART'] = self.start_time
            primary_hdu.header['SAMPLING'] = psf_sampling
            primary_hdu.header['STREHL'] = sr_value

            # Add coordinate info
            r, theta = self._get_source_coordinates(i)
            primary_hdu.header['COORD_R'] = r
            primary_hdu.header['COORD_T'] = theta

            if self.end_time:
                primary_hdu.header['TEND'] = self.end_time
            if results.get('pixel_scale'):
                primary_hdu.header['PIXSCALE'] = results['pixel_scale']

            # Save single HDU
            primary_hdu.writeto(filename, overwrite=True)

            if self.verbose:
                print(f"PSF for source {i} saved to: {filename}")

    def _save_modal_results(self, results: Dict, modal_params: dict):
        """Save modal analysis results as separate files for each source"""
        # Create directory if it doesn't exist
        self.modal_output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(self.sources)):
            modal_coeffs = results['modal_coeffs'][i]
            residual_var = results['residual_variance'][i]
            residual_avg = results['residual_average'][i]

            filename = self.modal_output_dir / self._get_analysis_filename("modal", source_idx=i, **modal_params)

            # Create HDU list for this source
            primary_hdu = fits.PrimaryHDU()

            # Modal coefficients as primary data
            modal_hdu = fits.ImageHDU(modal_coeffs, name='MODAL_COEFFS')

            # Residual variance as second extension
            var_hdu = fits.ImageHDU(residual_var, name='RESIDUAL_VAR')

            # Residual average as third extension
            avg_hdu = fits.ImageHDU(residual_avg, name='RESIDUAL_AVG')

            hdul = fits.HDUList([primary_hdu, modal_hdu, var_hdu, avg_hdu])

            # Add header info
            primary_hdu.header['TN'] = self.tracking_number
            primary_hdu.header['SOURCE'] = i
            primary_hdu.header['WAVELNG'] = self.wavelength_nm

            # Handle different ways to specify number of modes
            if 'nmodes' in modal_params:
                primary_hdu.header['NMODES'] = modal_params['nmodes']
            elif 'nzern' in modal_params:
                primary_hdu.header['NZERN'] = modal_params['nzern']

            if 'type_str' in modal_params:
                primary_hdu.header['MODTYPE'] = modal_params['type_str']

            # Add coordinate info
            r, theta = self._get_source_coordinates(i)
            primary_hdu.header['COORD_R'] = r
            primary_hdu.header['COORD_T'] = theta

            hdul.writeto(filename, overwrite=True)

            if self.verbose:
                print(f"Modal analysis for source {i} saved to: {filename}")

    def _save_cube_results(self, results: Dict):
        """Save phase cubes as separate files for each source"""
        # Create directory if it doesn't exist
        self.cube_output_dir.mkdir(parents=True, exist_ok=True)

        for i, phase_cube in enumerate(results['phase_cubes']):
            filename = self.cube_output_dir / self._get_analysis_filename("cube", source_idx=i)

            # Create HDU list for this source
            primary_hdu = fits.PrimaryHDU(phase_cube)

            # Times as second extension
            times_hdu = fits.ImageHDU(results['times'], name='TIMES')

            hdul = fits.HDUList([primary_hdu, times_hdu])

            # Add header info
            primary_hdu.header['TN'] = self.tracking_number
            primary_hdu.header['SOURCE'] = i
            primary_hdu.header['WAVELNG'] = self.wavelength_nm

            # Add coordinate info
            r, theta = self._get_source_coordinates(i)
            primary_hdu.header['COORD_R'] = r
            primary_hdu.header['COORD_T'] = theta

            hdul.writeto(filename, overwrite=True)

            if self.verbose:
                print(f"Phase cube for source {i} saved to: {filename}")

    def _load_psf_results(self, output_dir: Path, psf_sampling: int) -> Dict:
        """Load PSF results from separate files"""
        results = {
            'psf_list': [],
            'sr_list': [],
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm,
            'psf_sampling': psf_sampling,
            'pixel_scale': None
        }

        for i in range(len(self.sources)):
            filename = output_dir / self._get_analysis_filename("psf", source_idx=i, psf_sampling=psf_sampling, wavelength_nm=self.wavelength_nm)

            if filename.exists():
                hdul = fits.open(filename)
                results['psf_list'].append(hdul[0].data)
                results['sr_list'].append(hdul[0].header.get('STREHL', 0.0))

                if results['pixel_scale'] is None:
                    results['pixel_scale'] = hdul[0].header.get('PIXSCALE', None)

                hdul.close()
            else:
                raise FileNotFoundError(f"PSF file not found: {filename}")

        return results

    def _load_modal_results(self, output_dir: Path, modal_params: dict) -> Dict:
        """Load modal analysis results from separate files"""
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
            filename = output_dir / self._get_analysis_filename("modal", source_idx=i, **modal_params)

            if filename.exists():
                hdul = fits.open(filename)
                results['modal_coeffs'].append(hdul['MODAL_COEFFS'].data)
                results['residual_variance'].append(hdul['RESIDUAL_VAR'].data)
                results['residual_average'].append(hdul['RESIDUAL_AVG'].data)
                hdul.close()
            else:
                raise FileNotFoundError(f"Modal analysis file not found: {filename}")

        return results

    def _load_cube_results(self, output_dir: Path) -> Dict:
        """Load phase cubes from separate files"""
        results = {
            'phase_cubes': [],
            'times': None,
            'coordinates': self.polar_coordinates,
            'distances': self.distances,
            'wavelength_nm': self.wavelength_nm
        }

        for i in range(len(self.sources)):
            filename = output_dir / self._get_analysis_filename("cube", source_idx=i)

            if filename.exists():
                hdul = fits.open(filename)
                results['phase_cubes'].append(hdul[0].data)

                if results['times'] is None:
                    results['times'] = hdul['TIMES'].data

                hdul.close()
            else:
                raise FileNotFoundError(f"Phase cube file not found: {filename}")

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