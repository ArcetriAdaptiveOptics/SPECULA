

import specula
specula.init(0)  # Default target device

import os
import shutil
import unittest
from astropy.io import fits

from specula import np
from specula.calib_manager import CalibManager

class TestCalibManager(unittest.TestCase):

    def setUp(self):
        self.rootdir = os.path.join(os.path.dirname(__file__), 'data', 'test_calib')
        self.subdir = 'm2c'
        self.filename = os.path.join(self.rootdir, self.subdir, 'test_calibmanager.fits')

        cm = CalibManager(root_dir='dummy')
        real_data_dir = cm._subdirs['data']
        # Prepare calib/m2c and calib/data directory
        os.makedirs(os.path.join(self.rootdir, self.subdir), exist_ok=True)
        os.makedirs(os.path.join(self.rootdir, real_data_dir), exist_ok=True)

    def tearDown(self):
        try:
            shutil.rmtree(self.rootdir, ignore_errors=True)
        except FileNotFoundError:
            pass

    def test_calibmanager_filename(self):
        '''Test that the *filename()* method correctly joins subdir and name'''

        name = 'bar.fits'

        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs[self.subdir]
        
        expected = os.path.join(self.rootdir, real_subdir, name)

        assert calib_manager.filename(subdir=self.subdir, name=name) == expected


    def test_calibmanager_filename_fits(self):
        '''Test that the *filename()* method correctly adds .fits'''

        name = 'bar'
        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs[self.subdir]
        
        expected = os.path.join(self.rootdir, real_subdir, name)+'.fits'

        assert calib_manager.filename(subdir=self.subdir, name=name) == expected

    def test_calibmanager_writefits(self):

        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs[self.subdir]
        name = 'bar'
        data = np.arange(2)
        calib_manager.writefits(self.subdir, name, data=data)

        expected = os.path.join(self.rootdir, real_subdir, name)+'.fits'
        np.testing.assert_array_equal(fits.getdata(expected), data)

    def test_calibmanager_readfits(self):

        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs[self.subdir]
        name = 'bar'
        data = np.arange(2)

        expected = os.path.join(self.rootdir, real_subdir, name)+'.fits'
        fits.writeto(expected, data)

        np.testing.assert_array_equal(calib_manager.readfits(self.subdir, name), data)

    def test_calibmanager_writedata(self):

        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs['data']
        name = 'bar'
        data = np.arange(2)
        calib_manager.write_data(name, data=data)

        expected = os.path.join(self.rootdir, real_subdir, name)+'.fits'
        np.testing.assert_array_equal(fits.getdata(expected), data)

    def test_calibmanager_readdata(self):

        calib_manager = CalibManager(self.rootdir)
        real_subdir = calib_manager._subdirs['data']
        name = 'bar'
        data = np.arange(2)

        expected = os.path.join(self.rootdir, real_subdir, name)+'.fits'
        fits.writeto(expected, data)

        np.testing.assert_array_equal(calib_manager.read_data(name), data)