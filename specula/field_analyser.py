import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from astropy.io import fits
from copy import deepcopy

from specula.simul import Simul
from specula.data_objects.simul_params import SimulParams
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

        # Find DM objects and their input sources
        dm_input_sources = self._find_dm_input_sources(modified_params)

        if self.verbose:
            print(f"Found DM input sources: {dm_input_sources}")

        # Extract only DM command inputs from original DataStore input_list
        original_input_list = datastore_obj.get('inputs', {}).values()
        dm_command_inputs = []

        for input_ref in original_input_list:
            if isinstance(input_ref, str) and self._is_dm_command_in_datastore(input_ref, dm_input_sources):
                dm_command_inputs.append(input_ref)

        if not dm_command_inputs:
            raise RuntimeError(f"No DM command inputs found in DataStore configuration. "
                            f"DM input sources: {dm_input_sources}, "
                            f"Original input_list: {original_input_list}")

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

        # Update DataSource store_dir to point to correct tracking number directory
        if 'data_source' in temp_simul.replay_params:
            temp_simul.replay_params['data_source']['store_dir'] = str(self.tn_dir)
            if self.verbose:
                print(f"Updated DataSource store_dir to: {self.tn_dir}")

        return temp_simul.replay_params

    def _find_dm_input_sources(self, params: dict) -> set:
        """
        Find all objects that provide inputs to DM objects
        Returns set of object names that feed into DMs
        """
        dm_input_sources = set()

        for obj_name, obj_config in params.items():
            if isinstance(obj_config, dict) and obj_config.get('class') == 'DM':
                # Look at DM inputs to find source objects
                if 'inputs' in obj_config:
                    for input_name, output_ref in obj_config['inputs'].items():
                        if isinstance(output_ref, str):
                            # Extract source object name
                            if '.' in output_ref:
                                source_obj = output_ref.split('.')[0]
                                dm_input_sources.add(source_obj)
                                if self.verbose:
                                    print(f"DM '{obj_name}' gets input from '{source_obj}'")

        return dm_input_sources

    def _is_dm_command_in_datastore(self, input_ref: str, dm_input_sources: set) -> bool:
        """
        Check if a DataStore input reference corresponds to a DM command
        by checking if it references one of the known DM input sources
        
        Args:
            input_ref: DataStore input reference (format: 'filename-object.output')
            dm_input_sources: Set of object names that provide inputs to DMs
        """
        if '-' in input_ref:
            # DataStore format: 'filename-object.output'
            filename_part, object_output = input_ref.split('-', 1)

            if '.' in object_output:
                source_obj = object_output.split('.')[0]

                # Check if this object is one that feeds into DMs
                if source_obj in dm_input_sources:
                    if self.verbose:
                        print(f"Identified DM command: {input_ref} (source: {source_obj})")
                    return True

        return False

    def _build_replay_params_psf(self) -> dict:
        """
        Build replay_params for field PSF calculation using build_replay mechanism
        """
        # Get base replay params from DataStore mechanism
        replay_params = self._build_replay_params_from_datastore()

        if self.verbose:
            print(f"Base replay_params keys: {list(replay_params.keys())}")

        # Remove conflicting objects
        self._remove_conflicting_objects(replay_params)

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
        self._remove_conflicting_objects(replay_params)

        # Add field sources to existing parameters
        self._add_field_sources_to_params(replay_params)

        # Create simple IFunc with modal_params (let ModalAnalysis handle the complexity)
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

        # Add ModalAnalysis for each source
        modal_input_list = []
        for i, source_dict in enumerate(self.sources):
            modal_name = f'modal_analysis_{i}'
            modal_config = {
                'class': 'ModalAnalysis',
                'ifunc_ref': 'modal_analysis_ifunc',
                'inputs': {'in_ef': f'prop.out_field_source_{i}_ef'},
                'outputs': ['out_modes']
            }

            # Add ModalAnalysis-specific parameters
            for param in ['dorms', 'wavelengthInNm']:
                if param in modal_params:
                    modal_config[param] = modal_params[param]

            replay_params[modal_name] = modal_config

            # Create filename for this source
            modal_filename = self._get_modal_filename(i, modal_params)
            modal_input_list.append(f'{modal_filename}-{modal_name}.out_modes')

        # Add DataStore to save results
        replay_params['data_store_modal'] = {
            'class': 'DataStore',
            'store_dir': str(self.modal_output_dir),
            'data_format': 'fits',
            'create_tn': False,
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
        self._remove_conflicting_objects(replay_params)

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

        # Add field sources
        for i, source_dict in enumerate(self.sources):
            source_name = f'field_source_{i}'
            replay_params[source_name] = {
                'class': 'Source',
                'polar_coordinates': source_dict['polar_coordinates'],
                'magnitude': source_dict['magnitude'],
                'wavelengthInNm': source_dict['wavelengthInNm'],
                'height': source_dict['height']
            }

        prop_config = replay_params[prop_key]

        # Set only field sources
        source_refs = [f'field_source_{i}' for i in range(len(self.sources))]
        prop_config['source_dict_ref'] = source_refs

        # Set only field source outputs
        output_list = [f'out_field_source_{i}_ef' for i in range(len(self.sources))]
        prop_config['outputs'] = output_list

        if self.verbose:
            print(f"Updated propagation object '{prop_key}':")
            print(f"  Sources: {source_refs}")
            print(f"  Outputs: {output_list}")

    def _remove_conflicting_objects(self, replay_params: dict):
        """
        Remove objects that are NOT in the whitelist of allowed classes
        Uses a whitelist approach to keep only essential objects for field analysis
        """
        # Whitelist of allowed processing object classes
        allowed_processing_classes = {
            'AtmoEvolution',
            'AtmoInfiniteEvolution', 
            'AtmoPropagation',
            'AtmoRandomPhase',
            'DataSource',
            'DataStore',
            'DM',
            'ElectricFieldCombinator',
            # all generators
            'PushPullGenerator',
            'RandomGenerator',
            'ScheduleGenerator',
            'TimeHistoryGenerator',
            'VibrationGenerator',
            'WaveGenerator'
            # Add other processing objects as needed
        }

        # Whitelist of allowed data object classes
        allowed_data_classes = {
            'Source',
            'ElectricField',
            'IFunc',
            'Pupilstop',
            'Layer',
            'SimulParams'
            # Add other data objects as needed
        }

        # Combined whitelist
        allowed_classes = allowed_processing_classes | allowed_data_classes

        objects_to_remove = []

        for obj_name, obj_config in replay_params.items():
            # Skip non-dict objects
            if not isinstance(obj_config, dict):
                continue

            # If object has a 'class' field, check if it's in whitelist
            if 'class' in obj_config:
                obj_class = obj_config['class']

                # Remove if NOT in whitelist
                if obj_class not in allowed_classes:
                    objects_to_remove.append(obj_name)
                    if self.verbose:
                        print(f"Removing non-whitelisted object: {obj_name} (class: {obj_class})")

        # Remove the objects
        for obj_name in objects_to_remove:
            del replay_params[obj_name]

        if self.verbose:
            remaining_classes = set()
            for obj_config in replay_params.values():
                if isinstance(obj_config, dict) and 'class' in obj_config:
                    remaining_classes.add(obj_config['class'])

            print(f"Removed {len(objects_to_remove)} objects")
            print(f"Remaining classes: {sorted(remaining_classes)}")

    def _run_simulation_with_params(self, params_dict: dict, output_dir: Path) -> Simul:
        """
        Common simulation execution logic using minimal temporary file
        """
        import tempfile
        import os

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Computing simulation with parameters to be saved by DataStore in: {output_dir}")

        # Create minimal temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.dump(params_dict, temp_file, default_flow_style=False, sort_keys=False)
            temp_params_file = temp_file.name

        try:
            # Create Simul instance normally (this initializes all required attributes)
            simul = Simul(temp_params_file)
            simul.run()
            return simul
        except Exception as e:
            print(f"Simulation failed: {e}")
            print(f"Check DataStore output in: {output_dir}")
            print(f"Temp params file for debugging: {temp_params_file}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_params_file)
            except:
                pass  # File cleanup failure is not critical

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

        # Get simul_params from main configuration
        main_config = self.params.get('main', {})
        if not main_config:
            raise RuntimeError("No 'main' configuration found in parameters")

        # Create a temporary SimulParams object to initialize PSF
        temp_simul_params = SimulParams(pixel_pitch = self.params['main']['pixel_pitch'],
                                        pixel_pupil = self.params['main']['pixel_pupil'])

        temp_psf = PSF(
            simul_params=temp_simul_params,
            wavelengthInNm=self.wavelength_nm,
            nd=psf_sampling,
            pixel_size_mas=psf_pixel_size_mas,
            start_time=self.start_time
        )
        self.psf_sampling = temp_psf.nd
        self.psf_pixel_size_mas = temp_psf.psf_pixel_size

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
        simul = self._run_simulation_with_params(replay_params, self.psf_output_dir)

        if self.verbose:
            print(f"Actual PSF pixel size: {self.psf_pixel_size_mas:.2f} mas")

        # Extract results from DataStore (files are automatically saved)
        results = self._load_psf_results()

        return results

    def compute_modal_analysis(self, modal_params: Optional[Dict] = None, force_recompute: bool = False) -> Dict:
        """
        Calculate field modal analysis using replay system

        Args:
            modal_params: Simple dictionary with basic parameters:
                        - type_str: 'zernike', 'kl', etc. (default: 'zernike')
                        - nmodes/nzern: number of modes (default: 100)
                        - obsratio, diaratio: pupil parameters (optional)
                        - dorms: compute RMS flag (optional)
                        If None, attempts to extract from DM configuration
            force_recompute: Force recomputation even if files exist
        """
        if modal_params is None:
            modal_params = self._extract_modal_params_from_dm()

        # Validate and set defaults
        if 'nmodes' not in modal_params and 'nzern' not in modal_params:
            modal_params['nmodes'] = 100
        if 'type_str' not in modal_params:
            modal_params['type_str'] = 'zernike'

        # Check if files exist
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
                return self._load_modal_results(modal_params)

        if self.verbose:
            print(f"Computing field modal analysis for {len(self.sources)} sources...")
            print(f"Modal parameters: {modal_params}")

        # Setup replay parameters and run simulation
        replay_params = self._build_replay_params_modal(modal_params)
        simul = self._run_simulation_with_params(replay_params, self.modal_output_dir)

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
        simul = self._run_simulation_with_params(replay_params, self.cube_output_dir)

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
        Extract modal parameters from DM configuration with simple fallback
        """
        # Try to find a DM with height=0 and extract basic parameters
        if self.params is None:
            return {'type_str': 'zernike', 'nmodes': 100}

        # Look for DM with height=0
        for obj_name, obj_config in self.params.items():
            if isinstance(obj_config, dict) and obj_config.get('class') == 'DM':
                if obj_config.get('height', None) == 0:
                    # Extract simple parameters
                    modal_params = {}

                    # Direct copy of relevant parameters
                    for param in ['type_str', 'nmodes', 'nzern', 'obsratio', 'diaratio']:
                        if param in obj_config:
                            modal_params[param] = obj_config[param]

                    # If we have an ifunc_ref, try to get nmodes from it
                    if 'ifunc_ref' in obj_config and obj_config['ifunc_ref'] in self.params:
                        ifunc_config = self.params[obj_config['ifunc_ref']]
                        if isinstance(ifunc_config, dict):
                            for param in ['nmodes', 'nzern', 'type_str', 'obsratio']:
                                if param in ifunc_config and param not in modal_params:
                                    modal_params[param] = ifunc_config[param]

                    # Ensure we have basic parameters
                    if 'nmodes' not in modal_params and 'nzern' not in modal_params:
                        modal_params['nmodes'] = 100
                    if 'type_str' not in modal_params:
                        modal_params['type_str'] = 'zernike'

                    if self.verbose:
                        print(f"Extracted modal parameters from DM '{obj_name}': {modal_params}")

                    return modal_params

        # Fallback to defaults
        if self.verbose:
            print("No suitable DM found, using default modal parameters")

        return {'type_str': 'zernike', 'nmodes': 100}