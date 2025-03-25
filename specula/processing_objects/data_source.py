from astropy.io import fits

import os
import pickle

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.lib.utils import import_class


class DataSource(BaseProcessingObj):
    '''Data source object'''

    def __init__(self,
                outputs: list=[],
                store_dir: str="",
                data_format: str='fits'):
        super().__init__()
        self.items = {}
        self.storage = {}
        self.data_filename = ''
        self.tn_dir = store_dir
        self.data_format = data_format
        self.headers = {}
        self.obj_type = {}

        for aout in outputs:            
            self.loadFromFile(aout)
        for k in self.storage.keys():
            if not isinstance(self.obj_type[k], BaseValue):
                self.outputs[k] = import_class(self.obj_type[k]).from_header(self.headers[k])
            else:
                self.outputs[k] = BaseValue()

    def loadFromFile(self, name):
        if name in self.items:
            raise ValueError(f'Storing already has an object with name {name}')
        if self.data_format=='fits':
            self.load_fits(name)
        elif self.data_format=='pickle':
            self.load_pickle(name)

    def load_pickle(self, name):
        filename = os.path.join(self.tn_dir,name + '.pickle')
        with open( filename, 'rb') as handle:
            unserialized_data = pickle.load(handle)
        times = unserialized_data['times']
        data = unserialized_data['times']
        self.storage[name] = { t:data.data[i] for i, t in enumerate(times.data.tolist())}

    def load_fits(self, name):
        filename = os.path.join(self.tn_dir, name+'.fits')
        self.headers[name] = fits.getheader(filename)
        hdul = fits.open(filename)        
        times = hdul[1]
        data = hdul[0]
        self.storage[name] = { t:data.data[i] for i, t in enumerate(times.data.tolist())}
        self.obj_type[name] = self.headers[name]['OBJ_TYPE']

    def size(self, name, dimensions=False):
        if not self.has_key(name):
            print(f'The key: {name} is not stored in the object!')
            return -1
        h = self.storage[name]
        return h.shape if not dimensions else h.shape[dimensions]

    def trigger_code(self):
        for k in self.storage.keys():            
            self.outputs[k].set_value(self.outputs[k].xp.array(self.storage[k][self.current_time]))
            self.outputs[k].generation_time = self.current_time

        