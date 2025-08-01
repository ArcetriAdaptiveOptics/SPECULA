import specula
specula.init(0)

import unittest
import numpy as np
from specula.processing_objects.extended_source import ExtendedSource

class TestExtendedSource(unittest.TestCase):

    debug_plot = True  # Set to True to enable plotting for debugging

    def test_point_source(self):
        src = ExtendedSource(
            polar_coordinate=(0.0, 0.0),
            height=np.inf,
            magnitude=10.0,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            d_tel=8.0,
            source_type='POINT_SOURCE',
            size_obj=None,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertEqual(src.npoints, 1)
        self.assertTrue(np.allclose(src.coeff_flux, 1.0))

    def test_tophat_cartesian(self):
        src = ExtendedSource(
            polar_coordinate=(0.0, 0.0),
            height=np.inf,
            magnitude=10.0,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            d_tel=8.0,
            source_type='TOPHAT',
            size_obj=1.0,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)

    def test_gauss_cartesian(self):
        src = ExtendedSource(
            polar_coordinate=(0.0, 0.0),
            height=np.inf,
            magnitude=10.0,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            d_tel=8.0,
            source_type='GAUSS',
            size_obj=1.0,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)