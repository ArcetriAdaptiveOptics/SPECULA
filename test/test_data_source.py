
import os
import shutil

import specula
from specula.processing_objects.data_source import DataSource
specula.init(0)  # Default target device

from astropy.io import fits
import numpy as np
import unittest

from test.specula_testlib import cpu_and_gpu


class TestDataSource(unittest.TestCase):

    # Test that data source can read back files and output them correctly in its trigger method
    def setUp(self):
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp_data_source')
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
    
    # Create a fits file "gen.fits" for testing, with the expected output
    def _create_test_files(self):
        gen_file = os.path.join(self.tmp_dir, 'gen.fits')
        data = np.array(([3], [4]))
        times = np.array([0, 1], dtype=np.uint64)
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'BaseValue'
        data_hdu = fits.PrimaryHDU(data, header=hdr)
        time_hdu = fits.ImageHDU(times, header=hdr)
        hdul = fits.HDUList([data_hdu, time_hdu])
        hdul.writeto(gen_file, overwrite=True)

    def test_data_source(self):
        self._create_test_files()
        source = DataSource(store_dir=self.tmp_dir,
                            outputs=['gen'],
                            data_format='fits')
        
        gen = source.outputs['gen']

        source.check_ready(0)
        source.setup()
        source.trigger()
        source.post_trigger()
        assert gen.value == 3

        source.check_ready(1)
        source.setup()
        source.trigger()
        source.post_trigger()
        assert gen.value == 4
