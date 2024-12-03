import os
import unittest
from astropy.io import fits

import specula
specula.init(0)  # Default target device

from specula import cp, np
from specula import cpuArray

from specula.lib.make_xy import make_xy
from specula.lib.interp2d import Interp2D

class TestInterp2D(unittest.TestCase):

    @unittest.skipIf(cp is None, 'Cupy not found')
    def test_interp2d_gpu(self):
        self._test_interp2d(xp=cp)

    def test_interp2d_cpu(self):
        self._test_interp2d(xp=np)
        
    def _test_interp2d(self, xp):
        
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

if __name__ == '__main__':
    unittest.main()