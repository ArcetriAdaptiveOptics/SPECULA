import os
import unittest
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import np
from specula import cpuArray

from specula.lib.make_xy import make_xy
from specula.lib.interp2d import Interp2D

from test.specula_testlib import cpu_and_gpu

class TestInterp2D(unittest.TestCase):

    @cpu_and_gpu
    def test_interp2d(self, target_device_idx, xp):
        
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        phase = fits.getdata(os.path.join(datadir, 'input_phase.fits'))
        ref_phase = fits.getdata(os.path.join(datadir, 'ref_phase.fits'))
        
        half_pixel_layer = [240.5, 240.5]
        pixel_position = [0.0930127, 0]
        pixel_pupil = 480
        pixel_pupmeta = 479.84003
        
        xx, yy = make_xy(pixel_pupil, pixel_pupmeta/2., xp=xp)
        xx1 = xx + half_pixel_layer[0] + pixel_position[0]
        yy1 = yy + half_pixel_layer[1] + pixel_position[1]
        interpolator = Interp2D(phase.shape, (pixel_pupil, pixel_pupil), xx=xx1, yy=yy1,
                      rotInDeg=0, xp=xp, dtype=xp.float32)

        output_phase = interpolator.interpolate(xp.array(phase))
    
        test_phase = cpuArray(output_phase)
        
        # TODO one single pixel has value with 15% difference
        test_phase[394,365] = ref_phase[394,365]
        # Then a few pixels with 3%
        np.testing.assert_allclose(test_phase, ref_phase, rtol=4e-2)

    @cpu_and_gpu
    def test_interp2d_boundary_check(self, target_device_idx, xp):
        '''
        Test that interp2d does not read beyond the input array boundaries,
        using as input a small array that is a slice of a larger one, the latter
        filled with NaNs outside the slice range. We check that the output
        contains no NaNs.
        '''

        array = xp.empty(shape=(20,20), dtype=xp.float32)
        array[:] = np.nan
        array[:10] = 1

        interpolator = Interp2D(input_shape=(10, 20), output_shape=(5,5), rotInDeg=45, xp=xp, dtype=xp.float32)
        result = interpolator.interpolate(array[:10])
        assert xp.isnan(result).sum() == 0

    @cpu_and_gpu
    def test_interp2d_wrong_shape(self, target_device_idx, xp):
        '''
        Test that the interpolate method raises if the
        array to be interpolated has the wrong shape
        '''
        interpolator = Interp2D(input_shape=(10, 10), output_shape=(5,5), rotInDeg=45, xp=xp, dtype=xp.float32)
        with self.assertRaises(ValueError):
            _ = interpolator.interpolate(xp.zeros((20,20)))
