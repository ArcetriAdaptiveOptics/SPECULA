
import os
import numpy as np
from astropy.io import fits

from collections import OrderedDict, defaultdict
import pickle
import yaml
import time

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj


class DataStore(BaseProcessingObj):
    '''Data storage object'''

    def __init__(self,
                store_dir: str,         # TODO ="",
                split_size: int=0,
                first_suffix: int=0,
                data_format: str='fits',
                start_time: float=0,
                create_tn: bool=True):
        super().__init__()
        self.data_filename = ''
        self.today = time.strftime("%Y%m%d_%H%M%S")
        self.tn_dir = store_dir
        self.tn_dir_orig = store_dir     # Extra copy needed when suffix is used
        self.data_format = data_format
        self.create_tn = create_tn
        self.replay_params = None
        self.iter_counter = 0
        self.split_size = split_size
        self.first_suffix = first_suffix
        self.start_time = self.seconds_to_t(start_time)
        self.init_storage()

    def init_storage(self):
        self.storage = defaultdict(OrderedDict)

    def setParams(self, params):
        self.params = params

    def setReplayParams(self, replay_params):
        self.replay_params = replay_params

    def save_pickle(self):
        times = {k: np.array(list(v.keys()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        for k,v in times.items():            
            filename = os.path.join(self.tn_dir,k+'.pickle')
            hdr = self.inputs[k].get(target_device_idx=-1).get_fits_header()
            with open(filename, 'wb') as handle:
                data_to_save = {'data': data[k], 'times': times[k], 'hdr':hdr}
                pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_params(self):
        filename = os.path.join(self.tn_dir, 'params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.params, outfile,  default_flow_style=False, sort_keys=False)

        # Check if replay_params exists before using it
        if hasattr(self, 'replay_params') and self.replay_params is not None:
            self.replay_params['data_source']['store_dir'] = self.tn_dir
            filename = os.path.join(self.tn_dir, 'replay_params.yml')
            with open(filename, 'w') as outfile:
                yaml.dump(self.replay_params, outfile, default_flow_style=False, sort_keys=False)
        else:
            # Skip saving replay_params if not available
            if self.verbose:
                print("Warning: replay_params not available, skipping replay_params.yml creation")

    def save_fits(self):
        times = {k: np.array(list(v.keys()), dtype=np.uint64) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}

        for k,v in times.items():

            filename = os.path.join(self.tn_dir,k+'.fits')
            hdr = self.local_inputs[k].get_fits_header()
            hdu_time = fits.ImageHDU(times[k], header=hdr)
            hdu_data = fits.PrimaryHDU(data[k], header=hdr)
            hdul = fits.HDUList([hdu_data, hdu_time])
            hdul.writeto(filename, overwrite=True)
            hdul.close()  # Force close for Windows

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
                value = item.get_value()
                v = cpuArray(value, force_copy=True)
                self.storage[k][self.current_time] = v
        
        # If we are saving a split TN, check whether it is time to save a new chunk
        # In case, clear the storage dictionary to restart with an empty one.
        self.iter_counter += 1
        if self.split_size > 0:
            if self.iter_counter % self.split_size == 0:
                self.create_TN_folder(suffix=f'_{self.iter_counter - self.split_size + self.first_suffix}')
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
