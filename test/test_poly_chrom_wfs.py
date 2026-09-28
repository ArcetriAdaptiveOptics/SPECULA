import unittest

import specula
specula.init(0)  # Default target device

from specula.processing_objects.poly_chrom_wfs import PolyChromWFS
from test.specula_testlib import cpu_and_gpu


class TestPolyChromWFS(unittest.TestCase):

    @cpu_and_gpu
    def test_unit_tilt_dtype_follows_precision(self, target_device_idx, xp):
        """Unit tilt coordinate arrays (from make_xy) must follow the object's own dtype."""
        wfs32 = PolyChromWFS(wavelengthInNm=[500.0, 600.0], ccd_side=8,
                             flux_factor=[1.0, 1.0],
                             precision=1, target_device_idx=target_device_idx)
        tilt_x32, tilt_y32 = wfs32._create_unit_tilts(in_ef_size=8, in_ef_pixel_pitch=0.1)
        self.assertEqual(tilt_x32.dtype, xp.float32)
        self.assertEqual(tilt_y32.dtype, xp.float32)

        wfs64 = PolyChromWFS(wavelengthInNm=[500.0, 600.0], ccd_side=8,
                             flux_factor=[1.0, 1.0],
                             precision=0, target_device_idx=target_device_idx)
        tilt_x64, tilt_y64 = wfs64._create_unit_tilts(in_ef_size=8, in_ef_pixel_pitch=0.1)
        self.assertEqual(tilt_x64.dtype, xp.float64)
        self.assertEqual(tilt_y64.dtype, xp.float64)
