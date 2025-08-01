import specula
specula.init(0)

import unittest
import numpy as np
from specula.processing_objects.extended_source import ExtendedSource
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from specula.data_objects.simul_params import SimulParams

class TestExtendedSource(unittest.TestCase):

    debug_plot = True  # Set to True to enable plotting for debugging
    simul_params = SimulParams(
        pixel_pupil=160,
        pixel_pitch=0.05,
        zenithAngleInDeg=30.0
    )

    def test_point_source(self):
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
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
            simul_params=self.simul_params,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
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
            simul_params=self.simul_params,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            source_type='GAUSS',
            size_obj=1.0,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)

    def test_gauss_cartesian_3d(self):
        src = ExtendedSource(
            simul_params=self.simul_params,
            focus_height=90000.0,
            layer_height=[70000.0, 80000.0, 90000.0, 100000.0, 110000.0],
            intensity_profile=[0.1, 0.23, 0.34, 0.23, 0.1],
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            source_type='GAUSS',
            size_obj=1.0,
            sampling_type='CARTESIAN'
        )
        src.compute()
        if self.debug_plot:
            src.plot_source()
        self.assertGreater(src.npoints, 1)
        self.assertAlmostEqual(np.sum(src.coeff_flux), 1.0, places=6)
        
    def test_extended_source_in_pyramid(self):
        # Create an extended source
        src = ExtendedSource(
            simul_params=self.simul_params,
            wavelength_in_nm=500.0,
            multiples_fwhm=2.0,
            source_type='GAUSS',
            size_obj=1.0,
            sampling_type='CARTESIAN'
        )
        src.compute()

        # Pass it to the pyramid
        pyr = ModulatedPyramid(
            simul_params=self.simul_params,
            wavelengthInNm=500.0,
            fov=2.0,
            pup_diam=30,
            output_resolution=80,
            mod_amp=3.0,
            extended_source=src
        )

        # Check that the extended source is loaded and parameters are consistent
        self.assertTrue(pyr.extended_source_in_on)
        self.assertIs(pyr.extended_source, src)
        self.assertEqual(pyr.mod_steps, src.npoints)
        self.assertEqual(pyr.ttexp.shape[0], src.npoints)
        self.assertEqual(pyr.flux_factor_vector.shape[0], src.npoints)
        self.assertAlmostEqual(float(np.sum(specula.cpuArray(pyr.flux_factor_vector))), 1.0, places=6)
        self.assertEqual(pyr.ttexp.shape[1:], pyr.tilt_x.shape)

        # Optionally, plot for debug
        if self.debug_plot:
            src.plot_source()