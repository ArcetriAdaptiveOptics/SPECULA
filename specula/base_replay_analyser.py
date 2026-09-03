
from pathlib import Path
from typing import Optional
import yaml
from astropy.io import fits
import specula

from specula.simul import Simul
from specula.log import get_specula_logger


class BaseReplayAnalyser:
    """
    Shared replay machinery for classes that recompute results from a past
    simulation run by targeting one or more objects in its params.yml with
    Simul.build_targeted_replay, without re-running the whole simulation.

    Subclasses provide what to target and what analysis objects to attach;
    this base class handles locating the tracking number directory, loading
    params.yml, replay precision/downsampling checks, and running the replay
    simulation from a generated params dict.
    """

    def __init__(self,
                 data_dir: str,
                 tracking_number: str,
                 start_time: float = 0.1,
                 end_time: Optional[float] = None,
                 display: bool = False,
                 log_level: Optional[str] = None,
                 on_missing_downstream_consumers: str = 'error'):

        self.data_dir = Path(data_dir)
        self.tracking_number = tracking_number
        self.start_time = start_time
        self.end_time = end_time
        self.display = display
        if on_missing_downstream_consumers not in ('error', 'warn', 'ignore'):
            raise ValueError(
                "on_missing_downstream_consumers must be one of 'error', 'warn', 'ignore', "
                f"got {on_missing_downstream_consumers!r}"
            )
        self.on_missing_downstream_consumers = on_missing_downstream_consumers
        self.logger = get_specula_logger(__name__)
        if log_level is not None:
            self.logger.setLevel(log_level)

        self.params = None
        self.replay_precision = None

        self.tn_dir = self.data_dir / tracking_number
        self.base_output_dir = self.data_dir

        if not self.tn_dir.exists():
            raise FileNotFoundError(f"Tracking number directory not found: {self.tn_dir}")

        self._load_simulation_params()

    def _load_simulation_params(self):
        """Load simulation parameters from tracking number"""
        params_file = self.tn_dir / "params.yml"
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")

        with open(params_file, 'r') as f:
            self.params = yaml.safe_load(f)

    def _make_output_dir(self, suffix: str) -> Path:
        return self.base_output_dir / f"{self.tracking_number}_{suffix}"

    def _build_replay_params_from_datastore(self, *target_object_names) -> dict:
        """
        Build replay params targeting the given object(s), using
        Simul.build_targeted_replay, then apply the saved replay precision
        and re-inject any recorded RandomGenerator seeds.
        """
        simul = Simul('dummy.yaml')
        replay_params = simul.build_targeted_replay(
            self.params, *target_object_names, set_store_dir=str(self.tn_dir),
            on_missing_downstream_consumers=self.on_missing_downstream_consumers)
        self._validate_replay_inputs_are_not_downsampled(replay_params)
        simul.inject_recorded_seeds(replay_params, self._get_saved_replay_seeds())
        replay_precision = self._get_saved_replay_precision()
        self.replay_precision = replay_precision
        if replay_precision is None:
            self.logger.debug('Did not find saved replay precision; using current SPECULA precision state')
        else:
            self.logger.debug(f'Loaded replay precision={replay_precision} from replay_params.yml')
        self._ensure_replay_precision(replay_precision)
        return replay_params

    def _get_saved_replay_seeds(self) -> dict:
        replay_params_file = self.tn_dir / 'replay_params.yml'
        if not replay_params_file.exists():
            return {}

        with open(replay_params_file, 'r', encoding='utf-8') as handle:
            saved_replay_params = yaml.safe_load(handle) or {}

        data_source_cfg = saved_replay_params.get('data_source', {})
        if not isinstance(data_source_cfg, dict):
            return {}

        random_seeds = data_source_cfg.get('random_seeds', None)
        if not isinstance(random_seeds, dict):
            return {}

        return random_seeds

    def _get_saved_replay_precision(self) -> Optional[int]:
        replay_params_file = self.tn_dir / 'replay_params.yml'
        if not replay_params_file.exists():
            return None

        with open(replay_params_file, 'r', encoding='utf-8') as handle:
            saved_replay_params = yaml.safe_load(handle) or {}

        data_source_cfg = saved_replay_params.get('data_source', {})
        if not isinstance(data_source_cfg, dict):
            return None

        precision = data_source_cfg.get('global_precision', None)
        if precision is None:
            return None

        precision = int(precision)
        if precision not in (0, 1):
            self.logger.warning(f'invalid global_precision={precision} in replay_params.yml; ignoring it')
            return None

        return precision

    def _validate_replay_inputs_are_not_downsampled(self, replay_params: dict):
        data_source = replay_params.get('data_source')
        if not data_source:
            return

        data_format = data_source.get('data_format', 'fits')
        if data_format not in ('fits', 'pickle'):
            return

        store_dir = Path(data_source.get('store_dir', self.tn_dir))
        extension = '.fits' if data_format == 'fits' else '.pickle'

        for output_name in data_source.get('outputs', []):
            file_path = store_dir / f'{output_name}{extension}'
            if data_format == 'fits':
                with fits.open(file_path) as hdul:
                    header = hdul[0].header
            else:
                import pickle
                with open(file_path, 'rb') as handle:
                    payload = pickle.load(handle)
                header = payload.get('hdr', {})

            downsampling = int(header.get('DOWNSAMP', 1))
            if downsampling > 1:
                raise ValueError(
                    f'Replay does not support downsampled replay inputs: '
                    f'{file_path.name} was saved with DOWNSAMP={downsampling}'
                )

            if 'DOWNSAMP' not in header:
                self.logger.warning(f'replay input {file_path.name} has no DOWNSAMP metadata; assuming DOWNSAMP=1')

    def _ensure_replay_precision(self, replay_precision: Optional[int]):
        if replay_precision not in (0, 1):
            return

        if specula.global_precision == replay_precision:
            return

        if specula.default_target_device_idx is None:
            self.logger.warning('SPECULA not initialized yet, cannot enforce replay precision automatically')
            return

        specula.init(
            device_idx=specula.default_target_device_idx,
            precision=replay_precision,
            rank=specula.process_rank,
            comm=specula.process_comm,
        )

    def _add_extra_objects_to_params(self, params_dict: dict):
        """
        Override to inject extra objects (e.g. displays) into params_dict
        before running the replay simulation. No-op by default.
        """
        pass

    def _run_simulation_with_params(self, params_dict: dict, output_dir: Path) -> Simul:
        """
        Common simulation execution logic using minimal temporary file
        """
        import tempfile
        import os

        output_dir.mkdir(parents=True, exist_ok=True)

        self._add_extra_objects_to_params(params_dict)

        self.logger.debug(f"Computing simulation with parameters to be saved by DataStore in: {output_dir}")

        # Create minimal temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as temp_file:
            yaml.dump(params_dict, temp_file, default_flow_style=False, sort_keys=False)
            temp_params_file = temp_file.name

        try:
            # Create Simul instance normally (this initializes all required attributes)
            simul = Simul(temp_params_file)
            simul.run(start_time=self.start_time, end_time=self.end_time)
            return simul
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            self.logger.error(f"Check DataStore output in: {output_dir}")
            self.logger.error(f"Temp params file for debugging: {temp_params_file}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_params_file)
            except:
                pass  # File cleanup failure is not critical

    def _read_fits_primary_and_times(self, path: Path):
        """
        Read a FITS file's primary HDU data, plus an optional second HDU
        holding a time vector (as saved by DataStore), if present.
        """
        with fits.open(path) as hdul:
            data = hdul[0].data   # pylint: disable=no-member
            times = hdul[1].data if len(hdul) > 1 else None   # pylint: disable=no-member
        return data, times
