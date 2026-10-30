
import os
import numpy as np

from collections import OrderedDict, defaultdict
import pickle
import yaml

import iteras
from iteras import cpuArray
from iteras.base_processing_obj import BaseProcessingObj
from iteras.lib import utils
from iteras.lib import storage


class DataStore(BaseProcessingObj):
    """
    Data storage processing object.
    Stores input values over time, optionally downsampling them on a per-input
    basis, and saves them to disk at the end of the run.
    """
    def __init__(self,
                store_dir: str,
                split_size: int=0,
                first_suffix: int=0,
                data_format: str=None,
                start_time: float=0,
                create_tn: bool=True,
                downsample_factor: int=1,
                downsample_factor_by_input: dict=None,
                target_device_idx: int=None,
                precision: int=None):
        """
        Parameters
        ----------
        store_dir : str
            Base directory where data will be stored. A subdirectory with a timestamp
            will be created inside this directory to hold the data files.
        split_size : int [1], optional
            If > 0, creates a new subdirectory and saves one chunk every
            ``split_size`` trigger iterations after ``start_time``.
            Default is 0 (no splitting, all data in one folder).
        first_suffix : int [1], optional
            Starting suffix for split folders. Default is 0.
        data_format : str, optional
            Format for saved data files: 'fits', 'pickle', 'npz', 'npy' or
            'hdf5' (see iteras.lib.storage).
            Default is the value of the ITERAS_DATA_FORMAT environment variable,
            or 'fits' if it is not set.
        start_time : float [s], optional
            Time in seconds to wait before starting to store data.
            Default is 0 (store from the beginning).
        create_tn : bool
            If True, creates a timestamped subdirectory for storing data.
            Default is True.
        downsample_factor : int [1], optional
            Store one sample every ``N`` received samples for all inputs.
            The downsampling is sample-based and tracked independently for each
            input, so an input that updates less frequently is counted only when
            it produces a new sample. Default is 1 (store every sample).
        downsample_factor_by_input : dict [1], optional
            Per-input downsampling factors. Keys are the DataStore input names,
            values are integers >= 1. When using ``input_list``, the key is the
            alias before the dash, e.g. ``'comm'`` for ``'comm-control.out_comm'``.
            This option is mutually exclusive with ``downsample_factor != 1``.
        target_device_idx : int [1], optional
            Device of the object. The data is always stored in host memory.
        precision : int [1], optional
            Precision (0 = double, 1 = single).
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.data_filename = ''
        self.today = utils.make_tn()
        self.tn_dir = store_dir
        self.tn_dir_orig = store_dir     # Extra copy needed when suffix is used
        if data_format is None:
            data_format = storage.default_format()
        storage.check_format(data_format)
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

    @classmethod
    def input_names(cls):
        return {}

    @classmethod
    def output_names(cls):
        return {}

    @classmethod
    def accepts_input_list(cls):
        return True

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

    @property
    def storage(self):
        '''Stored values, by input name and time (the pending GPU copies are completed first)'''
        self._store_pending_copies()
        return self._storage

    @storage.setter
    def storage(self, value):
        # GPU values being copied to the host: (input, time, pinned buffer, event)
        self._pending_copies = []
        # Page-locked buffer of each input, where its GPU values are copied
        self._staging = {}
        # Host block of each input where its values are stored: [array, next free row]
        self._blocks = {}
        self._storage = value

    def set_simulation_params(self, params):
        self.params = params

    def set_replay_params(self, replay_params):
        self.replay_params = replay_params

    # Previous names, kept for compatibility
    setParams = set_simulation_params
    setReplayParams = set_replay_params

    def save_pickle(self):
        times = {k: np.array(list(v.keys()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict) and k is not None}
        data = {k: np.array(list(v.values()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict) and k is not None}

        for k, v in times.items():
            try:
                if k not in self.inputs or self.inputs[k] is None:
                    self.logger.warning(f"skipping key '{k}' - not in inputs or value is None")
                    continue

                filename = os.path.join(self.tn_dir, k + '.pickle')
                hdr = self.inputs[k].get(target_device_idx=-1).get_fits_header()
                self.update_header_with_storage_metadata(hdr, input_name=k)
                with open(filename, 'wb') as handle:
                    data_to_save = {'data': data[k], 'times': times[k], 'hdr': hdr}
                    pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:
                self.logger.error(f"Error saving pickle file for key '{k}': {str(e)}")
                continue

    def save_params(self):
        filename = os.path.join(self.tn_dir, 'params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.params, outfile,  default_flow_style=False, sort_keys=False)

        # Check if replay_params exists before using it
        if hasattr(self, 'replay_params') and self.replay_params is not None:
            self.replay_params['data_source']['store_dir'] = self.tn_dir
            self.replay_params['data_source']['global_precision'] = int(self.precision)
            # The format actually used, which may come from ITERAS_DATA_FORMAT
            self.replay_params['data_source']['data_format'] = self.data_format
            filename = os.path.join(self.tn_dir, 'replay_params.yml')
            with open(filename, 'w') as outfile:
                yaml.dump(self.replay_params, outfile, default_flow_style=False, sort_keys=False)
        else:
            # Skip saving replay_params if not available
            self.logger.warning("replay_params not available, skipping replay_params.yml creation")

    def save_data(self, data_format=None):
        '''
        Save the stored inputs, one file per input, in *data_format* (default:
        the format of this DataStore): the data, the times and the header of
        the input data object (see iteras.lib.storage). For FITS files, the data
        is in the primary HDU, with the header, and the times in the first
        extension. The pickle format is saved by save_pickle().
        '''
        data_format = data_format or self.data_format
        times = {k: np.array(list(v.keys()), dtype=np.uint64)
            for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype)
            for k, v in self.storage.items() if isinstance(v, OrderedDict)}

        for k in times:
            try:
                if k not in self.local_inputs or self.local_inputs[k] is None:
                    self.logger.warning(f"Warning: skipping key '{k}'"
                              f"- not in local_inputs or value is None")
                    continue

                hdr = self.local_inputs[k].get_fits_header()
                self.update_header_with_storage_metadata(hdr, input_name=k)
                storage.write_object(storage.filename(self.tn_dir, k, data_format), hdr,
                                     {'data': data[k], 'times': times[k]}, layout='primary')

            except Exception as e:
                self.logger.error(f"Error saving {data_format} file for key '{k}': {str(e)}")
                continue

    def save_fits(self):
        self.save_data('fits')

    def save_npz(self):
        self.save_data('npz')

    def save_npy(self):
        self.save_data('npy')

    def save_hdf5(self):
        self.save_data('hdf5')

    def create_TN_folder(self, suffix=''):
        iter = None
        while True:
            tn = f'{self.today}'
            fullpath = os.path.join(self.tn_dir_orig, tn) + suffix
            if iter is not None:
                fullpath += f'.{iter}'
            # mkdir is atomic: an exists() check before it would race with other
            # processes that pick the same timestamp-based name
            try:
                os.makedirs(fullpath)
                break
            except FileExistsError:
                pass
            if iter is None:
                iter = 0
            else:
                iter += 1
        self.tn_dir = fullpath

    def _copy_to_host(self, k, value):
        '''
        Store a copy of *value* for input *k* at the current time. A GPU value is
        copied asynchronously into page-locked memory, so that the simulation does
        not wait for the GPU here; the copy is moved to the storage by
        _store_pending_copies(), at the next step or when the storage is read.
        '''
        cp = iteras.cp
        if cp is None or not isinstance(value, cp.ndarray):
            self._storage[k][self.current_time] = cpuArray(value, force_copy=True)
            return
        pinned = self._staging.get(k)
        if pinned is None or pinned.shape != value.shape or pinned.dtype != value.dtype:
            import cupyx
            pinned = self._staging[k] = cupyx.empty_pinned(value.shape, dtype=value.dtype)
        with value.device:
            stream = cp.cuda.get_current_stream()
            value.get(stream=stream, out=pinned, blocking=False)
            event = stream.record()
        # Placeholder, to keep the order of the times in the storage
        self._storage[k][self.current_time] = None
        self._pending_copies.append((k, self.current_time, pinned, event))

    def _store_pending_copies(self):
        '''Wait for the copies of GPU values started before, and store them'''
        for k, t, pinned, event in self._pending_copies:
            event.synchronize()
            self._storage[k][t] = self._host_copy(k, pinned)
        self._pending_copies = []

    # Size of the host blocks where the values are stored. A new array for each
    # value would be slower to fill, its memory being new at each step; numpy
    # asks for huge pages for arrays of a few MB.
    BLOCK_BYTES = 4 * 1024 * 1024

    def _host_copy(self, k, value):
        '''Copy of *value* (a host array) in the next free row of the block of input *k*'''
        block = self._blocks.get(k)
        if block is None or block[1] == len(block[0]) or block[0].shape[1:] != value.shape \
                or block[0].dtype != value.dtype:
            rows = max(1, self.BLOCK_BYTES // max(value.nbytes, 1))
            block = self._blocks[k] = [np.empty((rows,) + value.shape, dtype=value.dtype), 0]
        array, row = block
        block[1] += 1
        out = array[row:row + 1].reshape(value.shape)
        out[...] = value
        return out

    def trigger_code(self):
        if self.current_time < self.start_time:
            return

        # The copies of the previous step are complete by now, usually
        self._store_pending_copies()
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                if not self._should_store_input(k):
                    continue
                self._copy_to_host(k, item.get_value())

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
        elif self.data_format in storage.OBJECT_FORMATS:
            self.save_data()
        else:
            raise TypeError(f"Error: unsupported file format {self.data_format}")

    def finalize(self):

        # Perform an additional trigger to ensure all data is captured,
        # including any calculations done in other objects' finalize() methods.
        # Inputs from another device are copies: copy them again first (not those
        # from other processes, which send nothing after their finalize())
        for name, input_obj in self.inputs.items():
            if all(item.remote_rank is None for item in input_obj.input_values):
                self.local_inputs[name] = input_obj.get(self.target_device_idx)
        self.trigger_code()

        if self.split_size == 0:
            if self.create_tn:
                self.create_TN_folder()
            else:
                os.makedirs(self.tn_dir_orig,exist_ok=True)
            self.save()

    def check_input_names(self):
        # DataStore inputs are added dynamically via input_list;
        # skip the static input_names validation.
        pass
