
import os
import numpy as np
from astropy.io import fits

from collections import OrderedDict, defaultdict
import pickle
import yaml

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.lib import utils


class DataStore(BaseProcessingObj):
    """
    Data storage processing object.
    Stores input values over time, optionally downsampling them on a per-input
    basis, and saves them to disk at the end of the run.
    """
    def __init__(self,
                store_dir: str,         # TODO ="",
                split_size: int=0,
                first_suffix: int=0,
                data_format: str='fits',
                start_time: float=0,
                create_tn: bool=True,
                downsample_factor: int=1,
                downsample_factor_by_input: dict=None):
        """
        Parameters
        ----------
        store_dir : str
            Base directory where data will be stored. A subdirectory with a timestamp
            will be created inside this directory to hold the data files.
        split_size : int, optional
            If > 0, creates a new subdirectory and saves one chunk every
            ``split_size`` trigger iterations after ``start_time``.
            Default is 0 (no splitting, all data in one folder).
        first_suffix : int, optional
            Starting suffix for split folders. Default is 0.
        data_format : str, optional
            Format for saved data files. Supported values are 'fits' and 'pickle'.
            Default is 'fits'.
        start_time : float, optional
            Time in seconds to wait before starting to store data.
            Default is 0 (store from the beginning).
        create_tn : bool, optional
            If True, creates a timestamped subdirectory for storing data.
            Default is True.
        downsample_factor : int, optional
            Store one sample every ``N`` received samples for all inputs.
            The downsampling is sample-based and tracked independently for each
            input, so an input that updates less frequently is counted only when
            it produces a new sample. Default is 1 (store every sample).
        downsample_factor_by_input : dict, optional
            Per-input downsampling factors. Keys are the DataStore input names,
            values are integers >= 1. When using ``input_list``, the key is the
            alias before the dash, e.g. ``'comm'`` for ``'comm-control.out_comm'``.
            This option is mutually exclusive with ``downsample_factor != 1``.
        """
        super().__init__()
        self.data_filename = ''
        self.today = utils.make_tn()
        self.tn_dir = store_dir
        self.tn_dir_orig = store_dir     # Extra copy needed when suffix is used
        self.data_format = data_format
        self.create_tn = create_tn
        self.replay_params = None
        self.iter_counter = 0
        self.split_size = split_size
        self.first_suffix = first_suffix
        self.start_time = self.seconds_to_t(start_time)
        self.downsample_factor = self._validate_downsample_factor(
            downsample_factor,
            'downsample_factor'
        )
        if downsample_factor_by_input is not None and self.downsample_factor != 1:
            raise ValueError('downsample_factor_by_input requires downsample_factor == 1')
        self.downsample_factor_by_input = {
            key: self._validate_downsample_factor(value, f'downsample_factor_by_input[{key!r}]')
            for key, value in (downsample_factor_by_input or {}).items()
        }
        self.input_sample_counters = defaultdict(int)
        self.init_storage()

    @staticmethod
    def _validate_downsample_factor(value, name):
        value = int(value)
        if value < 1:
            raise ValueError(f'{name} must be >= 1')
        return value

    def _should_store_input(self, input_name):
        every = self.downsample_factor_by_input.get(input_name, self.downsample_factor)
        sample_idx = self.input_sample_counters[input_name]
        self.input_sample_counters[input_name] += 1
        return sample_idx % every == 0

    def _downsampling_for_input(self, input_name):
        return self.downsample_factor_by_input.get(input_name, self.downsample_factor)

    def update_header_with_storage_metadata(self, header, input_name):
        header['DOWNSAMP'] = (self._downsampling_for_input(input_name),
                              'Stored one sample every N received samples')
        header['DSMODE'] = ('SAMPLE', 'Downsampling mode used by DataStore')

    def init_storage(self):
        self.storage = defaultdict(OrderedDict)

    def setParams(self, params):
        self.params = params

    def setReplayParams(self, replay_params):
        self.replay_params = replay_params

    def save_pickle(self):
        times = {k: np.array(list(v.keys()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict) and k is not None}
        data = {k: np.array(list(v.values()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict) and k is not None}

        for k, v in times.items():
            try:
                if k not in self.inputs or self.inputs[k] is None:
                    if self.verbose:
                        print(f"Warning: skipping key '{k}' - not in inputs or value is None")
                    continue

                filename = os.path.join(self.tn_dir, k + '.pickle')
                hdr = self.inputs[k].get(target_device_idx=-1).get_fits_header()
                self.update_header_with_storage_metadata(hdr, input_name=k)
                with open(filename, 'wb') as handle:
                    data_to_save = {'data': data[k], 'times': times[k], 'hdr': hdr}
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                if self.verbose:
                    print(f"Error saving pickle file for key '{k}': {str(e)}")
                continue

    def save_params(self):
        filename = os.path.join(self.tn_dir, 'params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.params, outfile,  default_flow_style=False, sort_keys=False)

        # Check if replay_params exists before using it
        if hasattr(self, 'replay_params') and self.replay_params is not None:
            self.replay_params['data_source']['store_dir'] = self.tn_dir
            self.replay_params['data_source']['global_precision'] = int(self.precision)
            filename = os.path.join(self.tn_dir, 'replay_params.yml')
            with open(filename, 'w') as outfile:
                yaml.dump(self.replay_params, outfile, default_flow_style=False, sort_keys=False)
        else:
            # Skip saving replay_params if not available
            if self.verbose:
                print("Warning: replay_params not available, skipping replay_params.yml creation")

    def save_fits(self):
        times = {k: np.array(list(v.keys()), dtype=np.uint64)
            for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict)}

        for k,v in times.items():
            try:
                if k not in self.local_inputs or self.local_inputs[k] is None:
                    if self.verbose:
                        print(f"Warning: skipping key '{k}'"
                              f"- not in local_inputs or value is None")
                    continue

                filename = os.path.join(self.tn_dir, k + '.fits')
                hdr = self.local_inputs[k].get_fits_header()
                self.update_header_with_storage_metadata(hdr, input_name=k)
                hdu_time = fits.ImageHDU(times[k], header=hdr)
                hdu_data = fits.PrimaryHDU(data[k], header=hdr)
                hdul = fits.HDUList([hdu_data, hdu_time])
                hdul.writeto(filename, overwrite=True)
                hdul.close()  # Force close for Windows

            except Exception as e:
                if self.verbose:
                    print(f"Error saving FITS file for key '{k}': {str(e)}")
                continue

    def create_TN_folder(self, suffix=''):
        iter = None
        while True:
            tn = f'{self.today}'
            fullpath = os.path.join(self.tn_dir_orig, tn) + suffix
            if iter is not None:
                fullpath += f'.{iter}'
            if not os.path.exists(fullpath):
                os.makedirs(fullpath)
                break
            if iter is None:
                iter = 0
            else:
                iter += 1
        self.tn_dir = fullpath

    def trigger_code(self):
        if self.current_time < self.start_time:
            return

        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                if not self._should_store_input(k):
                    continue
                value = item.get_value()
                v = cpuArray(value, force_copy=True)
                self.storage[k][self.current_time] = v

        # If we are saving a split TN, check whether it is time to save a new chunk
        # In case, clear the storage dictionary to restart with an empty one.
        self.iter_counter += 1
        if self.split_size > 0:
            if self.iter_counter % self.split_size == 0:
                self.create_TN_folder(
                    suffix=f'_{self.iter_counter - self.split_size + self.first_suffix}'
                )
                self.save()
                self.init_storage()

    def setup(self):
        # We check that all input items
        for k, _input in self.inputs.items():
            item = _input.get(target_device_idx=self.target_device_idx)
            if item is not None and not hasattr(item, 'get_value'):
                raise TypeError(f"Error: don't know how to buffer an object of type {type(item)}")

    def save(self):
        self.save_params()
        if self.data_format == 'pickle':
            self.save_pickle()
        elif self.data_format == 'fits':
            self.save_fits()
        else:
            raise TypeError(f"Error: unsupported file format {self.data_format}")

    def finalize(self):

        # Perform an additional trigger to ensure all data is captured,
        # including any calculations done in other objects' finalize() methods
        self.trigger_code()

        if self.split_size == 0:
            if self.create_tn:
                self.create_TN_folder()
            self.save()
