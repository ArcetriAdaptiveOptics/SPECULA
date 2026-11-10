import specula
specula.init(0)  # Default target device

import unittest
from specula import cpuArray
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.poly_crom_pyramid import PolyCromPyramid
from specula.processing_objects.modulated_pyramid import ModulatedPyramid
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

    @cpu_and_gpu
    def test_ld_parameters_are_fixed_angles(self, target_device_idx, xp):
        """
        mod_amp and pyr_*_ld are given at wavelengthInNm[ref_wavelength_idx]
        and rescaled by lambda_ref / lambda_i. Expected: value_i * lambda_i is
        the same for all wavelengths (fixed angle on the sky) and equal to
        value * lambda_ref; the instance at the reference wavelength has
        exactly the given value. Positive and negative indices are checked.
        """
        wavelengths = [500, 750, 1000]
        given = dict(mod_amp=3.0, pyr_roof_ld=2.0, pyr_edge_def_ld=0.4,
                     pyr_tip_def_ld=0.6, pyr_tip_maya_ld=0.8, pyr_max_side_ld=20.0)
        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)

        for ref_idx in (0, 1, -1):
            poly_pyr = PolyCromPyramid(
                wavelengthInNm=wavelengths,
                flux_factor=[1.0, 1.0, 1.0],
                simul_params=simul_params,
                fov=2.0,
                pup_diam=30,
                output_resolution=80,
                ref_wavelength_idx=ref_idx,
                xy_tilts_in_arcsec=[[0, 0]] * 3,
                target_device_idx=target_device_idx,
                **given
            )
            lambda_ref = wavelengths[ref_idx]
            instances = poly_pyr._wfs_instances
            self.assertEqual(len(instances), 3)
            for name, value in given.items():
                angles = [getattr(inst, name) * wl for inst, wl in zip(instances, wavelengths)]
                np.testing.assert_allclose(angles, value * lambda_ref, rtol=1e-12,
                                           err_msg=f'{name}, ref_wavelength_idx={ref_idx}')
                # Exact value at the reference wavelength
                self.assertEqual(getattr(instances[ref_idx], name), value)

    @cpu_and_gpu
    def test_ref_wavelength_idx_out_of_range(self, target_device_idx, xp):
        """ref_wavelength_idx outside [-n, n-1] raises ValueError."""
        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)
        for bad_idx in (3, -4):
            with self.assertRaises(ValueError):
                PolyCromPyramid(
                    wavelengthInNm=[500, 600, 700],
                    flux_factor=[1.0, 1.0, 1.0],
                    simul_params=simul_params,
                    fov=2.0,
                    pup_diam=30,
                    output_resolution=80,
                    ref_wavelength_idx=bad_idx,
                    xy_tilts_in_arcsec=[[0, 0]] * 3,
                    target_device_idx=target_device_idx
                )

    @cpu_and_gpu
    def test_modulation_radius_is_same_angle_at_all_wavelengths(self, target_device_idx, xp):
        """
        With mod_amp = 4 lambda_ref/D (lambda_ref = 600 nm), the time-averaged
        focal-plane PSF (out_psf_tot) of a circular pupil is a ring of the same
        angular radius at 600 and 900 nm, i.e. 4 lambda_ref/D, which is
        (r_pix / fft_res * lambda_i / lambda_ref must coincide for the two).
        The ring radius is estimated as the flux-weighted median radius around
        the centre (N/2-0.5, N/2-0.5): the mean radius is biased high (by
        ~0.5 lambda/D) by the diffraction wings, the median gives ~4.1 for both
        wavelengths. Without the rescaling the 900 nm instance would give ~6.
        """
        t = 1
        wavelengths = [600, 900]
        simul_params = SimulParams(pixel_pupil=120, pixel_pitch=0.05)
        poly_pyr = PolyCromPyramid(
            wavelengthInNm=wavelengths,
            flux_factor=[1.0, 1.0],
            simul_params=simul_params,
            fov=2.0,
            pup_diam=30,
            output_resolution=80,
            mod_amp=4.0,
            mod_type='circular',
            xy_tilts_in_arcsec=[[0, 0], [0, 0]],
            target_device_idx=target_device_idx
        )

        ef = ElectricField(120, 120, 0.05, S0=100, target_device_idx=target_device_idx)
        yy, xx = np.mgrid[:120, :120]
        ef.A = xp.asarray(np.hypot(yy - 59.5, xx - 59.5) < 60, dtype=ef.A.dtype)
        ef.generation_time = t

        poly_pyr.inputs['in_ef'].set(ef)
        poly_pyr.setup()
        poly_pyr.check_ready(t)
        poly_pyr.trigger()
        poly_pyr.post_trigger()

        radii = []
        for inst, wl in zip(poly_pyr._wfs_instances, wavelengths):
            psf = cpuArray(inst.outputs['out_psf_tot'].value).astype(np.float64)
            n = psf.shape[0]
            y, x = np.mgrid[:n, :n]
            r = np.hypot(y - (n / 2 - 0.5), x - (n / 2 - 0.5)).ravel()
            r = r / inst.fft_res * wl / wavelengths[0]       # [lambda_ref/D]
            order = np.argsort(r)
            cum = np.cumsum(psf.ravel()[order]) / psf.sum()
            radii.append(r[order][np.searchsorted(cum, 0.5)])

        # Same ring radius at both wavelengths, close to the requested 4 lambda_ref/D
        self.assertAlmostEqual(radii[0], radii[1], delta=0.1)
        for radius in radii:
            self.assertAlmostEqual(radius, 4.0, delta=0.3)

    def _build(self, target_device_idx, wavelengths=(500, 750, 1000), **kwargs):
        simul_params = SimulParams(pixel_pupil=64, pixel_pitch=0.1)
        params = dict(wavelengthInNm=list(wavelengths), flux_factor=[1.0] * len(wavelengths),
                      simul_params=simul_params, fov=2.0, pup_diam=24, pup_dist=32,
                      output_resolution=64, ref_wavelength_idx=1, mod_amp=3.0,
                      mod_type='circular', target_device_idx=target_device_idx)
        params.update(kwargs)
        return PolyCromPyramid(**params)

    @cpu_and_gpu
    def test_default_mod_step_from_shortest_wavelength(self, target_device_idx, xp):
        """Default mod_step is the same for all instances and equals the
        ModulatedPyramid default for the largest amplitude in lambda/D,
        3.0*750/500 = 4.5 -> 36 steps."""
        poly = self._build(target_device_idx)
        self.assertEqual(ModulatedPyramid.default_mod_step(3.0 * 750 / 500, 'circular'), 36)
        self.assertEqual([p.mod_steps for p in poly._wfs_instances], [36, 36, 36])

    @cpu_and_gpu
    def test_explicit_mod_step_checked_at_shortest_wavelength(self, target_device_idx, xp):
        """Explicit mod_step=20 is too small for 4.5 lambda/D at 500 nm
        (needs >= around(2*pi*4.5) = 28) and raises; 28 is accepted and shared
        by all instances."""
        with self.assertRaises(ValueError):
            self._build(target_device_idx, mod_step=20)
        poly = self._build(target_device_idx, mod_step=28)
        self.assertEqual([p.mod_steps for p in poly._wfs_instances], [28, 28, 28])

    @cpu_and_gpu
    def test_fp_obs_same_angle_at_all_wavelengths(self, target_device_idx, xp):
        """fp_obs=3 lambda_ref/D is a fixed angle: the obstruction diameter in
        lambda_i/D times lambda_i is the same for all instances and about
        3*lambda_ref."""
        wavelengths = (500, 750, 1000)
        fp_obs = 3.0
        poly = self._build(target_device_idx, wavelengths=wavelengths, fp_obs=fp_obs)
        products = []
        for pyr, wl in zip(poly._wfs_instances, wavelengths):
            fp_obs_i = fp_obs * 750 / wl
            mask = cpuArray(pyr.fp_mask)
            n = mask.shape[0]
            y, x = np.mgrid[:n, :n]
            r = np.hypot(y - (n / 2 - 0.5), x - (n / 2 - 0.5))
            area = np.count_nonzero((mask == 0) & (r <= fp_obs_i * pyr.fft_res / 2 + 2))
            diam_ld = 2 * np.sqrt(area / np.pi) / pyr.fft_res
            products.append(diam_ld * wl)
        for prod in products:
            self.assertAlmostEqual(prod / products[0], 1.0, delta=0.1)
            self.assertAlmostEqual(prod / (fp_obs * 750), 1.0, delta=0.1)
