import specula
specula.init(0)  # Default target device

import unittest
from unittest import mock

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula.processing_objects.modulated_double_roof import ModulatedDoubleRoof
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
from test.specula_testlib import cpu_and_gpu


class TestModulatedDoubleRoof(unittest.TestCase):

    @cpu_and_gpu
    def test_flat_wavefront_output_size(self, target_device_idx, xp):
        """Test that ModulatedDoubleRoof produces correct output dimensions for flat wavefront"""

        # Test parameters
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        wavelength_nm = 500
        fov = 2.0
        pup_diam = 30
        pup_dist = 36
        output_resolution = 80
        mod_amp = 10.0
        ref_S0 = 100

        # Create simulation parameters
        simul_params = SimulParams(
            pixel_pupil=pixel_pupil,
            pixel_pitch=pixel_pitch
        )

        # Create ModulatedDoubleRoof sensor
        double_roof = ModulatedDoubleRoof(
            simul_params=simul_params,
            wavelengthInNm=wavelength_nm,
            fov=fov,
            pup_diam=pup_diam,
            pup_dist=pup_dist,
            output_resolution=output_resolution,
            mod_amp=mod_amp,
            target_device_idx=target_device_idx
        )

        # Create flat wavefront (no phase)
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=ref_S0, target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        ef.generation_time = t

        # Connect input
        double_roof.inputs['in_ef'].set(ef)

        # Setup and run
        double_roof.setup()
        double_roof.check_ready(t)
        double_roof.trigger()
        double_roof.post_trigger()

        # Get output intensity
        intensity = double_roof.outputs['out_i']

        plot_debug = False
        if plot_debug:
            import matplotlib.pyplot as plt
            plt.imshow(intensity.i)
            plt.title("Output Intensity")
            plt.colorbar()
            plt.show()

        # Test 1: Check that both roofs exist
        self.assertTrue(hasattr(double_roof, 'roof1_tlt'), "roof1_tlt should be created")
        self.assertTrue(hasattr(double_roof, 'roof2_tlt'), "roof2_tlt should be created")

        # Test 2: Check roof dimensions are equal
        self.assertEqual(double_roof.roof1_tlt.shape, double_roof.roof2_tlt.shape,
                        "Both roofs should have the same dimensions")

        # Test 3: Check output dimensions
        expected_shape = (output_resolution, output_resolution)
        self.assertEqual(intensity.i.shape, expected_shape,
                        f"Output intensity shape {intensity.i.shape} doesn't match expected {expected_shape}")

        # Test 4: Check that output is positive (intensities should be non-negative)
        self.assertTrue(xp.all(intensity.i >= 0), "Intensity values should be non-negative")

        # Test 5: Check flux conservation (total intensity should match input)
        total_flux = xp.sum(intensity.i)
        expected_flux = ref_S0 * ef.masked_area()
        np.testing.assert_allclose(cpuArray(total_flux), cpuArray(expected_flux),
                                 rtol=0.1, atol=1e-6,
                                 err_msg="Total flux is not conserved")

        # Test 6: Check that we have 4 sub-pupils (basic structure test)
        # The intensity should have 4 distinct bright regions
        max_intensity = xp.max(intensity.i)
        threshold = max_intensity * 0.1  # 10% of max intensity
        bright_pixels = float(xp.sum(intensity.i > threshold))

        # Should have a reasonable number of bright pixels for 4 sub-pupils
        min_expected_pixels = 4 * (pup_diam // 4) ** 2  # Very rough estimate
        self.assertGreater(bright_pixels, min_expected_pixels,
                          "Not enough bright pixels for 4 sub-pupils")

        # Test 7:  Test quadrant structure
        h, w = intensity.i.shape
        mid_h, mid_w = h // 2, w // 2

        # Extract quadrants
        q1 = intensity.i[:mid_h, :mid_w]       # Top-left
        q2 = intensity.i[:mid_h, mid_w:]       # Top-right
        q3 = intensity.i[mid_h:, :mid_w]       # Bottom-left
        q4 = intensity.i[mid_h:, mid_w:]       # Bottom-right

        # Each quadrant should have some significant intensity
        total_intensity = cpuArray(xp.sum(intensity.i))
        q1_intensity = cpuArray(xp.sum(q1))
        q2_intensity = cpuArray(xp.sum(q2))
        q3_intensity = cpuArray(xp.sum(q3))
        q4_intensity = cpuArray(xp.sum(q4))

        # Each quadrant should have at least 5% of total intensity
        min_quadrant_fraction = 0.05
        self.assertGreater(q1_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 1 has insufficient intensity")
        self.assertGreater(q2_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 2 has insufficient intensity")
        self.assertGreater(q3_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 3 has insufficient intensity") 
        self.assertGreater(q4_intensity, total_intensity * min_quadrant_fraction,
                          "Quadrant 4 has insufficient intensity")

        print(f"Test passed: output shape = {intensity.i.shape}, "
              f"total flux = {cpuArray(total_flux):.1f}, "
              f"bright pixels = {cpuArray(bright_pixels)}")

        print(f"Quadrant intensities: Q1={cpuArray(q1_intensity):.1f}, "
              f"Q2={cpuArray(q2_intensity):.1f}, Q3={cpuArray(q3_intensity):.1f}, "
              f"Q4={cpuArray(q4_intensity):.1f}")

    @cpu_and_gpu
    def test_psf_same_as_modulated_pyramid(self, target_device_idx, xp):
        """The PSF before the focal plane mask does not depend on the mask, so for
        the same electric field and modulation it must be the same as the one of
        ModulatedPyramid (it was counted twice, once per roof)."""
        t = 1
        pixel_pupil = 120
        pixel_pitch = 0.05
        simul_params = SimulParams(pixel_pupil=pixel_pupil, pixel_pitch=pixel_pitch)
        params = dict(simul_params=simul_params, wavelengthInNm=500, fov=2.0, pup_diam=30,
                      pup_dist=36, output_resolution=80, mod_amp=3.0,
                      target_device_idx=target_device_idx)

        # Tilted wavefront, so that the PSF is not centered
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=100,
                           target_device_idx=target_device_idx)
        ef.A = make_mask(pixel_pupil)
        x = xp.arange(pixel_pupil, dtype=ef.phaseInNm.dtype)
        ef.phaseInNm[:] = 20.0 * (x[None, :] - pixel_pupil / 2) * ef.A
        ef.generation_time = t

        psf = {}
        for name, cls in (('roof', ModulatedDoubleRoof), ('pyr', ModulatedPyramid)):
            wfs = cls(**params)
            wfs.inputs['in_ef'].set(ef)
            wfs.setup()
            wfs.check_ready(t)
            wfs.trigger()
            wfs.post_trigger()
            psf[name] = cpuArray(wfs.outputs['out_psf_bfm'].value)

        self.assertGreater(psf['pyr'].max(), 0)
        np.testing.assert_allclose(psf['roof'], psf['pyr'], rtol=1e-5,
                                   atol=1e-6 * psf['pyr'].max())

class TestDoubleRoofGeometry(unittest.TestCase):
    """Roof phase maps (roof1_tlt, roof2_tlt): geometry and defects.
    Pixel coordinates: u = col - (N/2 - 0.5), v = row - (N/2 - 0.5);
    roof 1 has its edge at u = 0, roof 2 at v = 0."""

    def _make(self, target_device_idx, **kwargs):
        simul_params = SimulParams(pixel_pupil=64, pixel_pitch=0.1)
        params = dict(simul_params=simul_params, wavelengthInNm=750, fov=2.0, pup_diam=24,
                      pup_dist=32, output_resolution=64, mod_amp=3.0,
                      target_device_idx=target_device_idx)
        params.update(kwargs)
        return ModulatedDoubleRoof(**params)

    @staticmethod
    def _roofs(obj):
        """Roofs in tilt pixels (tilt_scale removed), on the host, in float64"""
        return [np.asarray(cpuArray(r), dtype=np.float64) * obj.tilt_scale
                for r in (obj.roof1_tlt, obj.roof2_tlt)]

    @staticmethod
    def _grids(n):
        row, col = np.mgrid[0:n, 0:n]
        return col - (n / 2 - 0.5), row - (n / 2 - 0.5), row, col

    @cpu_and_gpu
    def test_geometry_no_defects(self, target_device_idx, xp):
        """Without defects roof 1 = |u| - 0.5 + 2 (N/2 - 1 - row/2), roof 2 the same
        with v and col, and roof 2 is exactly roof 1 rotated by 90 degrees."""
        obj = self._make(target_device_idx)
        r1, r2 = self._roofs(obj)
        n = r1.shape[0]
        self.assertEqual(n, 192)
        u, v, row, col = self._grids(n)
        exp1 = np.abs(u) - 0.5 + 2 * (n // 2 - 1 - 0.5 * row)
        exp2 = np.abs(v) - 0.5 + 2 * (n // 2 - 1 - 0.5 * col)
        np.testing.assert_allclose(r1, exp1, atol=1e-4)
        np.testing.assert_allclose(r2, exp2, atol=1e-4)
        np.testing.assert_array_equal(cpuArray(obj.roof2_tlt),
                                      np.rot90(cpuArray(obj.roof1_tlt)))

    @cpu_and_gpu
    def test_edge_defect_only_along_own_edge(self, target_device_idx, xp):
        """pyr_edge_def_ld=1 -> strip of half-width 1.5 px along the own edge only:
        roof 1 modifies exactly the 4 columns |u| <= 1.5 (all rows), roof 2 exactly
        the 4 rows |v| <= 1.5. The strip along the other axis (which is not an edge
        of that roof) must stay untouched."""
        base1, base2 = self._roofs(self._make(target_device_idx))
        with mock.patch('numpy.random.rand', lambda n: np.full(n, -1.0)):
            obj = self._make(target_device_idx, pyr_edge_def_ld=1.0)
        self.assertEqual(obj.fft_res, 3.0)
        r1, r2 = self._roofs(obj)
        n = r1.shape[0]
        u, v, _, _ = self._grids(n)

        mod1 = ~np.isclose(r1, base1, atol=1e-4)
        mod2 = ~np.isclose(r2, base2, atol=1e-4)
        np.testing.assert_array_equal(mod1, np.abs(u) <= 1.5)
        np.testing.assert_array_equal(mod2, np.abs(v) <= 1.5)
        self.assertEqual(mod1.sum(), 4 * n)
        self.assertEqual(mod2.sum(), 4 * n)
        # Central rows of roof 1 away from its edge are untouched
        self.assertFalse(mod1[(np.abs(v) <= 1.5) & (np.abs(u) > 1.5)].any())
        # Modified pixels hold -max(roof) (random factor mocked to -1)
        np.testing.assert_allclose(r1[mod1], -base1.max(), rtol=1e-5)
        np.testing.assert_allclose(r2[mod2], -base2.max(), rtol=1e-5)

    @cpu_and_gpu
    def test_truncated_edge(self, target_device_idx, xp):
        """pyr_tip_maya_ld=4 -> r = 6 px: the face planes are cut at -0.5 + 6 = 5.5, giving a
        flat strip of the 12 columns with |u| - 0.5 < 5.5 (roof 1), unchanged elsewhere,
        and a profile across the edge that stays continuous (steps <= 1 px)."""
        obj = self._make(target_device_idx, pyr_tip_maya_ld=4.0)
        r1, r2 = self._roofs(obj)
        n = r1.shape[0]
        u, v, row, col = self._grids(n)
        tilt1 = 2 * (n // 2 - 1 - 0.5 * row)
        tilt2 = 2 * (n // 2 - 1 - 0.5 * col)

        np.testing.assert_allclose(r1 - tilt1, np.maximum(np.abs(u) - 0.5, 5.5), atol=1e-4)
        np.testing.assert_allclose(r2 - tilt2, np.maximum(np.abs(v) - 0.5, 5.5), atol=1e-4)
        flat = np.abs(u) - 0.5 < 5.5
        self.assertEqual(flat[0].sum(), 12)
        np.testing.assert_allclose((r1 - tilt1)[flat], 5.5, atol=1e-4)
        # Continuity across the edge along a row
        self.assertLessEqual(np.abs(np.diff((r1 - tilt1)[0])).max(), 1.0 + 1e-4)

    @cpu_and_gpu
    def test_finite_support(self, target_device_idx, xp):
        """pyr_max_side_ld=20 -> support radius 30 px: both roofs are 0 where
        sqrt(u^2 + v^2) > 30 and identical to the no-defect roofs inside."""
        base1, base2 = self._roofs(self._make(target_device_idx))
        obj = self._make(target_device_idx, pyr_max_side_ld=20.0)
        r1, r2 = self._roofs(obj)
        n = r1.shape[0]
        u, v, _, _ = self._grids(n)
        outside = np.sqrt(u ** 2 + v ** 2) > 30
        self.assertTrue(outside.any() and (~outside).any())
        for r, base in ((r1, base1), (r2, base2)):
            self.assertTrue(np.all(r[outside] == 0))
            np.testing.assert_allclose(r[~outside], base[~outside], atol=1e-4)

    @cpu_and_gpu
    def test_tip_defect_not_supported(self, target_device_idx, xp):
        """A roof has no tip: pyr_tip_def_ld != 0 must raise ValueError."""
        with self.assertRaises(ValueError):
            self._make(target_device_idx, pyr_tip_def_ld=1.0)
