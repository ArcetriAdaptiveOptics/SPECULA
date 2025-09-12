import specula
specula.init(0)  # Default target device

import unittest
from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.poly_crom_pyramid import PolyCromPyramid
from specula.data_objects.simul_params import SimulParams
from test.specula_testlib import cpu_and_gpu
import numpy as np

class TestPolyCromPyramid(unittest.TestCase):

    @cpu_and_gpu
    def test_basic_output_shape(self, target_device_idx, xp):
        t = 1
        wavelengths = [500, 600]
        flux_factors = [1.0, 1.0]
        xy_tilts = [[0, 0], [0, 0]]

        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)
        poly_pyr = PolyCromPyramid(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            simul_params=simul_params,
            fov=2.0,
            pup_diam=30,
            output_resolution=80,
            mod_amp=3.0,
            mod_type='circular',
            xy_tilts_in_arcsec=xy_tilts,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_pyr.inputs['in_ef'].set(ef)
        poly_pyr.setup()
        poly_pyr.check_ready(t)
        poly_pyr.trigger()
        poly_pyr.post_trigger()

        output_intensity = poly_pyr.outputs['out_i']
        expected_shape = (80, 80)
        self.assertEqual(output_intensity.i.shape, expected_shape)

    @cpu_and_gpu
    def test_flux_conservation(self, target_device_idx, xp):
        t = 1
        wavelengths = [500, 600, 700]
        flux_factors = [0.3, 0.4, 0.3]
        xy_tilts = [[0, 0], [0, 0], [0, 0]]

        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)
        poly_pyr = PolyCromPyramid(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            simul_params=simul_params,
            fov=2.0,
            pup_diam=30,
            output_resolution=80,
            mod_amp=3.0,
            mod_type='circular',
            xy_tilts_in_arcsec=xy_tilts,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_pyr.inputs['in_ef'].set(ef)
        poly_pyr.setup()
        poly_pyr.check_ready(t)
        poly_pyr.trigger()
        poly_pyr.post_trigger()

        total_output_flux = xp.sum(poly_pyr.outputs['out_i'].i)
        expected_flux = 100 * ef.masked_area()

        np.testing.assert_allclose(cpuArray(total_output_flux), cpuArray(expected_flux), rtol=1e-2)

    @cpu_and_gpu
    def test_wavelength_contributions(self, target_device_idx, xp):
        t = 1
        wavelengths = [500, 600]
        flux_factors = [0.5, 1.5]
        xy_tilts = [[0, 0], [0, 0]]

        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)
        poly_pyr = PolyCromPyramid(
            wavelengthInNm=wavelengths,
            flux_factor=flux_factors,
            simul_params=simul_params,
            fov=2.0,
            pup_diam=30,
            output_resolution=80,
            mod_amp=3.0,
            mod_type='circular',
            xy_tilts_in_arcsec=xy_tilts,
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        ef.generation_time = t

        poly_pyr.inputs['in_ef'].set(ef)
        poly_pyr.setup()
        poly_pyr.check_ready(t)
        poly_pyr.trigger()
        poly_pyr.post_trigger()

        contrib_0 = poly_pyr.get_wavelength_contribution(0)
        contrib_1 = poly_pyr.get_wavelength_contribution(1)
        total = xp.sum(contrib_0) + xp.sum(contrib_1)
        ratio = xp.sum(contrib_1) / total
        expected_ratio = flux_factors[1] / sum(flux_factors)
        np.testing.assert_allclose(cpuArray(ratio), expected_ratio, rtol=1e-2)