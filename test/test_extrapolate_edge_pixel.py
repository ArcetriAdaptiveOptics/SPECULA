

import specula
specula.init(0)

import os
import unittest
import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.lib.extrapolate_edge_pixel import extrapolate_edge_pixel

from test.specula_testlib import cpu_and_gpu

class TestExtrapolateEdgePixel(unittest.TestCase):

    @cpu_and_gpu
    def test_extrapolage_edge_pixel(self, target_device_idx, xp):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'phase.fits'))
        ref = fits.getdata(os.path.join(datadir, 'extrapolated1.fits'))
        mat1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=0).astype(int)
        mat2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixFalse.fits'), ext=1).astype(int)
        
        idx_1pix = np.where(mat1 >= 0)
        idx_2pix = np.where(mat2[idx_1pix] >= 0)
    
        xp_idx_1pix = tuple(map(xp.array, idx_1pix))
        xp_idx_2pix = tuple(map(xp.array, idx_2pix))
    
        test = extrapolate_edge_pixel(xp.array(phase), xp.array(mat1), xp.array(mat2), xp_idx_1pix, xp_idx_2pix, xp=xp)
        np.testing.assert_array_almost_equal(cpuArray(test), ref)

    @cpu_and_gpu
    def test_extrapolage_edge_pixel_doExt2Pix(self, target_device_idx, xp):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'phase.fits'))
        ref = fits.getdata(os.path.join(datadir, 'extrapolated2.fits'))
        mat1 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=0).astype(int)
        mat2 = fits.getdata(os.path.join(datadir, 'extrapol_array_doExt2PixTrue.fits'), ext=1).astype(int)

        idx_1pix = np.where(mat1 >= 0)
        idx_2pix = np.where(mat2[idx_1pix] >= 0)
    
        xp_idx_1pix = tuple(map(xp.array, idx_1pix))
        xp_idx_2pix = tuple(map(xp.array, idx_2pix))
        
        test = extrapolate_edge_pixel(xp.array(phase), xp.array(mat1), xp.array(mat2), xp_idx_1pix, xp_idx_2pix, xp=xp)
        np.testing.assert_array_almost_equal(cpuArray(test), ref)

