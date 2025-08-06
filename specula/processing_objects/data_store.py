
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
                data_format: str='fits',
                create_tn: bool=True):
        super().__init__()
        self.data_filename = ''
        self.tn_dir = store_dir
        self.data_format = data_format
        self.storage = defaultdict(OrderedDict)
        self._create_tn = create_tn
        self.replay_params = None

    def setParams(self, params):
        self.params = params

    def setReplayParams(self, replay_params):
        self.replay_params = replay_params

    def save_pickle(self, compress=False):
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

    def save_fits(self, compress=False):
        times = {k: np.array(list(v.keys()), dtype=np.uint64) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}

        for k,v in times.items():

            filename = os.path.join(self.tn_dir,k+'.fits')
            hdr = self.local_inputs[k].get_fits_header()
            hdu_time = fits.ImageHDU(times[k], header=hdr)
            hdu_data = fits.PrimaryHDU(data[k], header=hdr)
            hdul = fits.HDUList([hdu_data, hdu_time])
            hdul.writeto(filename, overwrite=True)

    def create_TN_folder(self):
        today = time.strftime("%Y%m%d_%H%M%S")
        iter = None
        while True:
            tn = f'{today}'
            prefix = os.path.join(self.tn_dir, tn)
            if iter is not None:
                prefix += f'.{iter}'
            if not os.path.exists(prefix):
                os.makedirs(prefix)
                break
            if iter is None:
                iter = 0
            else:
                iter += 1
        self.tn_dir = prefix

    def trigger_code(self):
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                if hasattr(item, 'get_value'):
                    value = item.get_value()
                    v = cpuArray(value, force_copy=True)
                else:
                    raise TypeError(f"Error: don't know how to save an object of type {type(item)}")
                self.storage[k][self.current_time] = v

    def finalize(self):

        # Perform an additional trigger to ensure all data is captured,
        # including any calculations done in other objects' finalize() methods
        self.trigger_code()

        if self._create_tn:
            self.create_TN_folder()
        self.save_params()
        if self.data_format == 'pickle':
            self.save_pickle()
        elif self.data_format == 'fits':
            self.save_fits()
        else:
            raise TypeError(f"Error: unsupported file format {self.data_format}")
