import specula
specula.init(0)  # Default target device

import os
import unittest
from astropy.io import fits

from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from test.specula_testlib import cpu_and_gpu
from specula.data_objects.simul_params import SimulParams

class TestLaserLaunchTelescope(unittest.TestCase):

    def setUp(self):
        datadir = os.path.join(os.path.dirname(__file__), 'data')
        self.filename = os.path.join(datadir, 'test_llt.fits')
        self.filename_no_fits = os.path.join(datadir, 'test_llt')
        self.simul_params = SimulParams(pixel_pupil=10, pixel_pitch=1.0, zenithAngleInDeg=30.0)

    @cpu_and_gpu
    def test_save_restore_roundtrip(self, target_device_idx, xp):

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        llt = LaserLaunchTelescope(
                simul_params=self.simul_params,
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
        assert llt.beacon_tt == llt2.beacon_tt
        self.assertAlmostEqual(llt.beacon_focus, llt2.beacon_focus, places=3)

    def tearDown(self):
        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

    @cpu_and_gpu
    def test_save_appends_fits_extension(self, target_device_idx, xp):
        """Test that save() automatically appends '.fits' if missing."""

        try:
            os.unlink(self.filename)
        except FileNotFoundError:
            pass

        llt = LaserLaunchTelescope(target_device_idx=target_device_idx)

        # Save without extension
        llt.save(self.filename_no_fits)

        # Check that file exists with '.fits' appended
        self.assertTrue(os.path.exists(self.filename_no_fits + ".fits"))

    @cpu_and_gpu
    def test_init_with_defaults(self, target_device_idx, xp):
        """Test initializing LaserLaunchTelescope with default values."""
        llt = LaserLaunchTelescope(target_device_idx=target_device_idx)

        self.assertEqual(llt.spot_size, 0.0)
        self.assertEqual(llt.tel_pos, [])
        self.assertEqual(llt.beacon_focus, 90e3)
        self.assertEqual(llt.beacon_tt, [0.0, 0.0])

    @cpu_and_gpu
    def test_init_with_custom_values(self, target_device_idx, xp):
        """Test initializing LaserLaunchTelescope with custom parameters."""
        llt = LaserLaunchTelescope(
            simul_params=self.simul_params,
            spot_size=2.5,
            tel_position=[1.0, 2.0, 3.0],
            beacon_focus=85000.0,
            beacon_tt=[0.1, 0.2],
            target_device_idx=target_device_idx,
        )

        airmass = 1.0 / xp.cos(xp.radians(self.simul_params.zenithAngleInDeg))

        self.assertEqual(llt.spot_size, 2.5)
        self.assertEqual(llt.tel_pos, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(llt.beacon_focus, 85000.0*airmass, places=3)
        self.assertEqual(llt.beacon_tt, [0.1, 0.2])

    @cpu_and_gpu
    def test_get_fits_header(self, target_device_idx, xp):
        """Test FITS header creation from LaserLaunchTelescope."""
        llt = LaserLaunchTelescope(
            simul_params=self.simul_params,
            spot_size=1.5,
            tel_position=[10.0, 20.0, 30.0],
            beacon_focus=95000.0,
            beacon_tt=[0.3, 0.4],
            target_device_idx=target_device_idx,
        )

        airmass = 1.0 / xp.cos(xp.radians(self.simul_params.zenithAngleInDeg))

        hdr = llt.get_fits_header()
        self.assertEqual(hdr["VERSION"], 1)
        self.assertEqual(hdr["SPOTSIZE"], 1.5)
        self.assertEqual(hdr["TELPOS_X"], 10.0)
        self.assertEqual(hdr["TELPOS_Y"], 20.0)
        self.assertEqual(hdr["TELPOS_Z"], 30.0)
        self.assertAlmostEqual(hdr["BEAC_FOC"], 95000.0*airmass, places=3)
        self.assertEqual(hdr["BEAC_TT0"], 0.3)
        self.assertEqual(hdr["BEAC_TT1"], 0.4)

    @cpu_and_gpu
    def test_from_header_invalid_version(self, target_device_idx, xp):
        """Test from_header raises ValueError for unknown versions."""
        hdr = fits.Header()
        hdr["VERSION"] = 99  # Invalid version
        hdr["SPOTSIZE"] = 1.0
        hdr["TELPOS_X"] = 0.0
        hdr["TELPOS_Y"] = 0.0
        hdr["TELPOS_Z"] = 0.0
        hdr["BEAC_FOC"] = 90000.0
        hdr["BEAC_TT0"] = 0.0
        hdr["BEAC_TT1"] = 0.0

        with self.assertRaises(ValueError):
            LaserLaunchTelescope.from_header(hdr, target_device_idx=target_device_idx)
