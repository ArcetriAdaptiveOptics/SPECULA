import specula
specula.init(0)  # Default target device

import os
import unittest
import numpy as np
from astropy.io import fits

from specula.data_objects.lenslet import Lenslet
from test.specula_testlib import cpu_and_gpu

class TestLenslet(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_lenslet.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        lenslet = Lenslet(n_lenses=4, target_device_idx=target_device_idx)
        
        lenslet.save(self.filename)
        lenslet2 = Lenslet.restore(self.filename)

        assert lenslet.n_lenses == lenslet2.n_lenses
        assert lenslet2.n_lenses == 4
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_init_default_and_custom(self, target_device_idx, xp):
        # Default: 1 lens
        lens1 = Lenslet(target_device_idx=target_device_idx)
        assert lens1.n_lenses == 1
        assert lens1.dimx == 1
        assert lens1.dimy == 1
        assert xp.allclose(lens1._lenses[0][0][:2], xp.array([0.0, 0.0]))

        # Custom: 2x2 lenses
        lens2 = Lenslet(2, target_device_idx=target_device_idx)
        assert lens2.n_lenses == 2
        assert lens2.dimx == 2
        assert lens2.dimy == 2
        assert isinstance(lens2._lenses[0][0], list)
        assert len(lens2._lenses[0][0]) == 3  # [x, y, size]

    @cpu_and_gpu
    def test_get_method_returns_correct_values(self, target_device_idx, xp):
        lens = Lenslet(3, target_device_idx=target_device_idx)
        x, y, size = lens.get(0, 0)
        assert np.isclose(size, 2.0 / 3)

    @cpu_and_gpu
    def test_get_value_and_set_value_raise(self, target_device_idx, xp):
        lens = Lenslet(2, target_device_idx=target_device_idx)
        with self.assertRaises(NotImplementedError):
            lens.get_value()
        with self.assertRaises(NotImplementedError):
            lens.set_value(None)

    @cpu_and_gpu
    def test_dimx_and_dimy_properties(self, target_device_idx, xp):
        lens = Lenslet(4, target_device_idx=target_device_idx)
        assert lens.dimx == 4
        assert lens.dimy == 4

        # Edge case: empty lenses
        lens._lenses = []
        assert lens.dimx == 0
        assert lens.dimy == 0

    @cpu_and_gpu
    def test_get_fits_header_contents(self, target_device_idx, xp):
        lens = Lenslet(5, target_device_idx=target_device_idx)
        hdr = lens.get_fits_header()
        assert isinstance(hdr, fits.Header)
        assert hdr["VERSION"] == 1
        assert hdr["N_LENSES"] == 5

    @cpu_and_gpu
    def test_from_header_with_invalid_version_raises(self, target_device_idx, xp):
        hdr = fits.Header()
        hdr["VERSION"] = 999
        hdr["N_LENSES"] = 4
        with self.assertRaises(ValueError):
            Lenslet.from_header(hdr, target_device_idx=target_device_idx)
