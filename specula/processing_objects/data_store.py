from astropy.io import fits

import os
import numpy as np

from collections import OrderedDict
import pickle
import yaml
import time

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes


class DataStore(BaseProcessingObj):
    '''Data storage object'''

    def __init__(self,
                store_dir: str,         # TODO ="",
                starting_time: float=0, 
                data_format: str='fits'):
        super().__init__()
        self.items = {}
        self.storage = {}
        self.data_filename = ''
        self.starting_time = starting_time
        self.tn_dir = store_dir
        self.data_format = data_format
        
    def setParams(self, params):
        self.params = params

    def setReplayParams(self, replay_params):
        self.replay_params = replay_params

    def add(self, data_obj, name=None):
        if name is None:
            name = data_obj.__class__.__name__
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        self.items[name] = data_obj
        self.storage[name] = OrderedDict()

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

        self.replay_params['data_source']['store_dir'] = self.tn_dir

        filename = os.path.join(self.tn_dir, 'replay_params.yml')
        with open(filename, 'w') as outfile:
            yaml.dump(self.replay_params, outfile,  default_flow_style=False, sort_keys=False)

    def save_fits(self, compress=False):
        times = {k: np.array(list(v.keys()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}
        data = {k: np.array(list(v.values()), dtype=self.dtype) for k, v in self.storage.items() if isinstance(v, OrderedDict)}        
        
        for k,v in times.items():
        
            filename = os.path.join(self.tn_dir,k+'.fits')
            hdr = self.inputs[k].get(target_device_idx=-1).get_fits_header()
            hdu_time = fits.ImageHDU(times[k], header=hdr)
            hdu_data = fits.PrimaryHDU(data[k], header=hdr)
            hdul = fits.HDUList([hdu_data, hdu_time])
            hdul.writeto(filename, overwrite=True)


    def create_TN_folder(self):
        today = time.strftime("%Y%m%d_%H%M%S")
        while True:
            tn = f'{today}'
            prefix = os.path.join(self.tn_dir, tn)
            if not os.path.exists(prefix):
                os.makedirs(prefix)
                break            
        self.tn_dir = prefix        

    def trigger_code(self):
        if self.current_time_seconds < self.starting_time:
            return

        for k, item in self.items.items():
            if item is not None and item.generation_time == self.current_time:
                if isinstance(item, BaseValue):
                    v = cpuArray(item.value)
                elif isinstance(item, Slopes):
                    v = cpuArray(item.slopes)
                elif isinstance(item, Pixels):
                    v = cpuArray(item.pixels)
                elif isinstance(item, ElectricField):
                    v = np.stack( (cpuArray(item.A), cpuArray(item.phaseInNm)) )
                else:
                    raise TypeError(f"Error: don't know how to save an object of type {type(item)}")
                self.storage[k][self.current_time] = v

    def finalize(self):        
        self.create_TN_folder()
        self.save_params()
        if self.data_format == 'pickle':
            self.save_pickle()
        elif self.data_format == 'fits':
            self.save_fits()
        else:
            raise TypeError(f"Error: unsupported file format {self.data_format}")