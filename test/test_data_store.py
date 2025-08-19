
import os
import shutil

import yaml
import specula
from specula.simul import Simul
specula.init(0)  # Default target device

from astropy.io import fits
import numpy as np
import unittest

from test.specula_testlib import cpu_and_gpu


class TestDataStore(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_store')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
       shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @cpu_and_gpu
    def test_data_store(self, target_device_idx, xp):
        params = {'main': {'class': 'SimulParams', 'root_dir': self.tmp_dir,
                           'time_step': 0.1, 'total_time': 0.2},
                  'generator': {'class': 'WaveGenerator', 'target_device_idx': target_device_idx, 'amp': 1, 'freq': 2},
                  'store': {'class': 'DataStore', 'store_dir': self.tmp_dir,
                            'inputs': {'input_list': ['gen-generator.output']},
                            }
                  }
        filename = os.path.join(self.tmp_dir, 'test_data_store.yaml')
        with open(filename, 'w') as outfile:
            yaml.dump(params, outfile)
        
        simul = Simul(filename)
        simul.run()

        # Find last TN in tmp_dir
        tn_dirs = sorted([d for d in os.listdir(self.tmp_dir) if d.startswith('2')])
        last_tn_dir = os.path.join(self.tmp_dir, tn_dirs[-1])

        # Read gen.fits file from last_tn_dir and compare with [1,2]
        gen_file = os.path.join(last_tn_dir, 'gen.fits')
        assert os.path.exists(gen_file), f"File {gen_file} does not exist"
        gen_data = fits.getdata(gen_file)
        np.testing.assert_array_almost_equal(gen_data, np.array([[0], [0.9510565162951535]]))

        # Make sure times are in int64
        gen_times = fits.getdata(gen_file, ext=1)
        assert gen_times.dtype == np.uint64

        # Make sure replay_params.yml exists
        replay_file = os.path.join(last_tn_dir, 'replay_params.yml')
        assert os.path.exists(replay_file), f"File {replay_file} does not exist"
