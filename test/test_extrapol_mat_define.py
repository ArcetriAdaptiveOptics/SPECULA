

import specula
specula.init(0)

import os
import unittest
import numpy as np
from astropy.io import fits

from specula.lib.extrapolate_edge_pixel_mat_define import extrapolate_edge_pixel_mat_define

class TestExtrapolMatDefine(unittest.TestCase):
    '''
    This test is CPU only, since it is a helper matrix
    that is loaded on CPU once calculated.
    '''
    def test_extrapolmat_define(self):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        mask = fits.getdata(os.path.join(datadir, 'mask.fits'))
        mat1, mat2, id1pix, idx2pix = extrapolate_edge_pixel_mat_define(mask, do_ext_2_pix=False)
        ref1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=0)
        ref2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=1)
        
        np.testing.assert_array_almost_equal(mat1, ref1)
        np.testing.assert_array_almost_equal(mat2, ref2)

    def test_extrapolmat_define_doExt2Pix(self):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        mask = fits.getdata(os.path.join(datadir, 'mask.fits'))
        mat1, mat2, id1pix, idx2pix = extrapolate_edge_pixel_mat_define(mask, do_ext_2_pix=True)
        ref1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=0)
        ref2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=1)
        
        np.testing.assert_array_almost_equal(mat1, ref1)
        np.testing.assert_array_almost_equal(mat2, ref2)

