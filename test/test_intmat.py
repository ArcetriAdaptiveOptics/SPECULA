

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula import np
from specula import cpuArray
from specula.data_objects.intmat import Intmat
from specula.data_objects.recmat import Recmat
from test.specula_testlib import cpu_and_gpu

class TestIntmat(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_im.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        im_data = xp.arange(10).reshape((5,2))
        im = Intmat(im_data, target_device_idx=target_device_idx)
        
        im.save(self.filename)
        im2 = Intmat.restore(self.filename)

        np.testing.assert_array_equal(cpuArray(im.intmat), cpuArray(im2.intmat))
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_init_with_defaults(self, target_device_idx, xp):
        mat = xp.zeros((3, 3))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        assert xp.allclose(intmat.intmat, mat)
        assert intmat.slope_mm is None
        assert intmat.slope_rms is None
        assert intmat.pupdata_tag == ''
        assert intmat.subapdata_tag == ''
        assert intmat.norm_factor == 0.0

    @cpu_and_gpu
    def test_get_and_set_value(self, target_device_idx, xp):
        mat = xp.ones((4, 4))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        new_mat = xp.full((4, 4), 7.0)
        intmat.set_value(new_mat)
        assert xp.allclose(intmat.get_value(), new_mat)

    @cpu_and_gpu
    def test_set_value_shape_mismatch_raises(self, target_device_idx, xp):
        mat = xp.ones((3, 3))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        with self.assertRaises(AssertionError):
            intmat.set_value(xp.ones((2, 2)))

    @cpu_and_gpu
    def test_reduce_size_and_slopes_and_set_start_mode(self, target_device_idx, xp):
        mat = xp.arange(30).reshape(6, 5)
        intmat = Intmat(mat, target_device_idx=target_device_idx)

        # Reduce modes
        intmat.reduce_size(2)
        assert intmat.intmat.shape == (6, 3)

        # Reduce slopes
        intmat.reduce_slopes(1)
        assert intmat.intmat.shape == (5, 3)

        # Set start mode
        intmat.set_start_mode(1)
        assert intmat.intmat.shape == (5, 2)

    @cpu_and_gpu
    def test_reduce_size_and_slopes_raises(self, target_device_idx, xp):
        mat = xp.ones((5, 5))
        intmat = Intmat(mat, target_device_idx=target_device_idx)

        with self.assertRaises(ValueError):
            intmat.reduce_size(5)
        with self.assertRaises(ValueError):
            intmat.reduce_slopes(5)
        with self.assertRaises(ValueError):
            intmat.set_start_mode(5)

    @cpu_and_gpu
    def test_nmodes_and_nslopes_properties(self, target_device_idx, xp):
        mat = xp.zeros((7, 9))
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        assert intmat.nmodes == 9
        assert intmat.nslopes == 7

    @cpu_and_gpu
    def test_generate_rec_and_pseudo_invert(self, target_device_idx, xp):
        mat = xp.eye(5)
        intmat = Intmat(mat, target_device_idx=target_device_idx)
        rec = intmat.generate_rec()
        assert isinstance(rec, Recmat)
        assert xp.allclose(rec.recmat, xp.linalg.pinv(mat))

    @cpu_and_gpu
    def test_build_from_slopes(self, target_device_idx, xp):
        times = [0, 1, 2]
        slopes = {
            t: xp.array([1.0, 2.0, 3.0]) for t in times
        }
        disturbance = {
            t: xp.array([1.0, -1.0, 1.0]) for t in times
        }
        intmat = Intmat(xp.zeros((3, 3)), target_device_idx=target_device_idx)
        im = intmat.build_from_slopes(slopes, disturbance)
        assert isinstance(im, Intmat)
        assert im.intmat.shape == (3, 3)
        assert xp.all(im.intmat[:, 0] != 0)
