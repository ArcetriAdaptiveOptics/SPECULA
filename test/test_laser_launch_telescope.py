

import specula
specula.init(0)  # Default target device

import os
import unittest

from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from test.specula_testlib import cpu_and_gpu

class TestLaserLaunchTelescope(unittest.TestCase):
   
    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_llt.fits')

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        llt = LaserLaunchTelescope(
                            spot_size = 1.0,
                            tel_position = [2, 3, 4],
                            beacon_focus= 88e3,
                            beacon_tt = [5.0, 6.0],
                            target_device_idx = target_device_idx
                            )

        
        llt.save(self.filename)
        llt2 = LaserLaunchTelescope.restore(self.filename)

        assert llt.spot_size == llt2.spot_size
        assert llt.tel_pos == llt2.tel_pos
        assert llt.beacon_focus == llt2.beacon_focus
        assert llt.beacon_tt == llt2.beacon_tt
        
    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

