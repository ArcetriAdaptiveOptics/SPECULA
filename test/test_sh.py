import specula
specula.init(0)  # Default target device

import tempfile
import unittest
from unittest import mock

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.sh import SH, choose_fov_resolution
from test.specula_testlib import cpu_and_gpu


class TestSH(unittest.TestCase):

    @cpu_and_gpu
    def test_sh_flux(self, target_device_idx, xp):

        ref_S0 = 100
        t = 1

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=20,
                subap_npx=6,
                target_device_idx=target_device_idx)

        ef = ElectricField(120,120,0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t

        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']

        np.testing.assert_almost_equal(xp.sum(intensity.i), ref_S0 * ef.masked_area())

    @cpu_and_gpu
    def test_pixelscale(self, target_device_idx, xp):
        '''
        Test that pixelscale is correctly handled, by comparing spots from a flat 
        wavefront and from a tilted one. The introduced tilt corresponds to exactly 1 pixel,
        and we verify that the resulting intensity field is indeed shifted by 1 pixel
        in the correct direction
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        flat = sh.outputs['out_i'].i.copy()

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2 * (1-1/pixel_pupil), tilt_value / 2 * (1-1/pixel_pupil), pixel_pupil)

        # Tilted wavefront
        ef.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        ef.generation_time = t+1

        sh.check_ready(t+1)
        sh.trigger()
        sh.post_trigger()
        tilted = sh.outputs['out_i'].i.copy()

        flat_shifted = np.roll(flat, (0, 1))

        # Remove the left column edges on each subap (comparison is invalid after roll)
        flat_shifted[:, ::sh_npix] = 0
        tilted[:, ::sh_npix] = 0

        # import matplotlib.pyplot as plt
        # plt.imshow(cpuArray(tilted))
        # plt.figure()
        # plt.imshow(cpuArray(flat_shifted))
        # plt.show()

        np.testing.assert_array_almost_equal(cpuArray(tilted), cpuArray(flat_shifted), decimal=4)

    @cpu_and_gpu
    def test_zeros_cache(self, target_device_idx, xp):
        '''
        Test that arrays are re-used between SH instances on the same target
        '''
        t = 1
        pxscale_arcsec = 0.5
        pixel_pupil = 120
        pixel_pitch = 0.05
        sh_npix = 6

        # clear cache before test
        SH._SH__zeros_cache.clear()

        sh1 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh2 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=20,
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)

        sh3 = SH(wavelengthInNm=500,
                subap_wanted_fov= sh_npix * pxscale_arcsec,
                sensor_pxscale=pxscale_arcsec,
                subap_on_diameter=30,  # Different
                subap_npx=sh_npix,
                target_device_idx=target_device_idx)


        # Flat wavefront
        ef = ElectricField(pixel_pupil, pixel_pupil,
                           pixel_pitch, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh1.inputs['in_ef'].set(ef)
        sh2.inputs['in_ef'].set(ef)
        sh3.inputs['in_ef'].set(ef)

        sh1.setup()
        sh2.setup()
        sh3.setup()

        # Test 1: sh1 and sh2 should share arrays (same geometry, same rank)
        assert id(sh1._wf3) == id(sh2._wf3), "sh1 and sh2 should share _wf3"
        assert id(sh1._psfimage) == id(sh2._psfimage), "sh1 and sh2 should share _psfimage"
        assert id(sh1.ef_row) == id(sh2.ef_row), "sh1 and sh2 should share ef_row"

        # Test 2: sh3 should NOT share with sh1/sh2 (different geometry)
        assert id(sh1._wf3) != id(sh3._wf3), "sh3 should have different _wf3 (different geometry)"

        # Test 4: Check cache size
        cache_size = len(SH._SH__zeros_cache)
        self.assertGreater(cache_size, 0, "Cache should have entries")

        # We should have entries for:
        # - sh1/sh2 (shared, rank 0, geometry 20)
        # - sh3 (separate, rank 0, geometry 30)
        # Each geometry allocates 3 arrays (_wf3, ef_row, _psfimage),
        # psf_shifted is only allocated when a convolution kernel is used.
        # So expected: 2 geometries × 3 arrays = 6 entries
        assert cache_size == 6
        print(f"Cache has {cache_size} entries")

    @cpu_and_gpu
    def test_oversampling_alignment(self, target_device_idx, xp):
        '''
        Test that the new float oversampling logic correctly aligns the phase size
        to be a multiple of (2 * n_lenses), even if the input size is irregular.
        '''
        t = 1
        wl = 500 # nm
        # We simulate a case where the pupil is 105 pixels and we want 10 subaps.
        # Modulus required = 2 * 10 = 20.
        # 105 is NOT divisible by 20. Next multiple is 120.
        # Expected oversampling = 120 / 105 = 1.142857...

        pixel_pupil = 105 # Irregular size
        n_lenses = 10

        sh = SH(wavelengthInNm=wl,
                subap_wanted_fov=2.0,
                sensor_pxscale=0.5,
                subap_on_diameter=n_lenses,
                subap_npx=4,
                fov_ovs_coeff=1.0, # No forced coeff, we want to test the automatic adjustment
                target_device_idx=target_device_idx)

        # Create the irregular electric field
        ef = ElectricField(pixel_pupil, pixel_pupil, 0.05, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()

        # 1. Check if the oversampling factor is a float > 1.0
        self.assertGreater(sh._fov_ovs, 1.0, "Oversampling should be > 1.0 to fix alignment")

        # 2. Verify the math: 105 * ovs should be exactly 120
        calculated_size = pixel_pupil * sh._fov_ovs
        self.assertAlmostEqual(calculated_size, 120.0, places=5,
                               msg=f"Expected 120 total pixels, got {calculated_size}")

        # 3. Verify internal pixel count
        # _ovs_np_sub = 120 // 10 = 12
        # This represents the full subaperture width in pixels (120 pixels / 10 subaps).
        self.assertEqual(sh._ovs_np_sub, 12,
                         "Internal subap pixel count should match total/n_lenses")

    @cpu_and_gpu
    def test_oversampling_forced_coeff(self, target_device_idx, xp):
        '''
        Test that providing a specific fov_ovs_coeff works and still respects
        the geometry constraints (multiple of 2*n_lenses).
        '''
        t = 1
        pixel_pupil = 100
        n_lenses = 10
        # Modulus = 20.

        # We force coefficient = 1.5
        # Target minimum size = 100 * 1.5 = 150.
        # 150 is NOT divisible by 20 (150/20 = 7.5).
        # Next multiple of 20 is 160.
        # Expected final oversampling = 160 / 100 = 1.6

        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=2.0,
                sensor_pxscale=0.5,
                subap_on_diameter=n_lenses,
                subap_npx=4,
                fov_ovs_coeff=1.5, # FORCE THIS
                target_device_idx=target_device_idx)

        ef = ElectricField(pixel_pupil, pixel_pupil, 0.05, S0=1,
                           target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()

        # Check that we respected the forced coeff (at least)
        self.assertGreaterEqual(sh._fov_ovs, 1.5, "Should respect minimum forced coefficient")

        # Check that we adjusted for geometry (1.6 expected)
        self.assertAlmostEqual(sh._fov_ovs, 1.6, places=5,
                               msg="Should have adjusted 1.5 -> 1.6 for geometry alignment")

        # Verify final size
        final_size = pixel_pupil * sh._fov_ovs
        self.assertAlmostEqual(final_size % 20, 0, places=5,
                               msg="Final size must be divisible by 20")

    @unittest.skipIf(specula.cp is None, 'GPU not available')
    def test_shared_interpolated_ef_in_cuda_graph(self):
        '''
        Two GPU SH objects with the same geometry share the interpolated
        field, which is computed inside their CUDA graphs. Over several steps
        with a changing input, each must give the same result as a CPU SH
        (no graph) running alone.
        '''
        n = 40
        yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
        pupil = (np.hypot(xx, yy) < n / 2).astype(float)
        rng = np.random.default_rng(1)
        phases = [rng.normal(size=(n, n)) * 80 for _ in range(3)]

        def make(target_device_idx, rot):
            sh = SH(wavelengthInNm=589, subap_wanted_fov=4.0, sensor_pxscale=0.5,
                    subap_on_diameter=5, subap_npx=8, rotAnglePhInDeg=rot,
                    target_device_idx=target_device_idx)
            return sh

        def run(shs, ef):
            out = []
            for sh in shs:
                sh.inputs['in_ef'].set(ef)
                sh.setup()
            for t, phase in enumerate(phases):
                ef.A[:] = ef.to_xp(pupil)
                ef.phaseInNm[:] = ef.to_xp(phase)
                ef.generation_time = t
                for sh in shs:
                    sh.check_ready(t)
                    sh.trigger()
                    sh.post_trigger()
                    out.append(cpuArray(sh.outputs['out_i'].i).copy())
            return out

        gpu_shs = [make(0, 0.0), make(0, 7.0)]
        gpu_ef = ElectricField(n, n, 0.1, S0=10, target_device_idx=0)
        gpu_out = run(gpu_shs, gpu_ef)

        assert gpu_shs[0].ef_interpolator.interpolated_ef() is \
               gpu_shs[1].ef_interpolator.interpolated_ef()
        assert all(sh.cuda_graph is not None for sh in gpu_shs)

        for k, rot in enumerate([0.0, 7.0]):
            cpu_ef = ElectricField(n, n, 0.1, S0=10, target_device_idx=-1)
            cpu_out = run([make(-1, rot)], cpu_ef)
            for t in range(len(phases)):
                np.testing.assert_allclose(gpu_out[t * 2 + k], cpu_out[t],
                                           rtol=1e-4, atol=1e-6 * cpu_out[t].max())

    @unittest.skipIf(specula.cp is None, 'GPU not available')
    def test_cuda_graph_recaptured_after_interpolator_update(self):
        '''
        The interpolation parameters are frozen in the CUDA graph: after
        update_interpolator_parameters(), the GPU SH must give the same result
        as a CPU SH (no graph) created with the new parameters.
        '''
        n = 40
        yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
        pupil = (np.hypot(xx, yy) < n / 2).astype(float)
        phase = np.random.default_rng(2).normal(size=(n, n)) * 80

        def make(target_device_idx, rot):
            sh = SH(wavelengthInNm=589, subap_wanted_fov=4.0, sensor_pxscale=0.5,
                    subap_on_diameter=5, subap_npx=8, rotAnglePhInDeg=rot,
                    target_device_idx=target_device_idx)
            ef = ElectricField(n, n, 0.1, S0=10, target_device_idx=target_device_idx)
            ef.A[:] = ef.to_xp(pupil)
            ef.phaseInNm[:] = ef.to_xp(phase)
            sh.inputs['in_ef'].set(ef)
            sh.setup()
            return sh, ef

        def step(sh, ef, t):
            ef.generation_time = t
            sh.check_ready(t)
            sh.trigger()
            sh.post_trigger()
            return cpuArray(sh.outputs['out_i'].i).copy()

        gpu_sh, gpu_ef = make(0, 0.0)
        out_before = step(gpu_sh, gpu_ef, 0)
        step(gpu_sh, gpu_ef, 1)

        gpu_sh.update_interpolator_parameters(rotAnglePhInDeg=7.0)
        out_after = step(gpu_sh, gpu_ef, 2)

        cpu_sh, cpu_ef = make(-1, 7.0)
        out_ref = step(cpu_sh, cpu_ef, 0)

        np.testing.assert_allclose(out_after, out_ref, rtol=1e-4, atol=1e-6 * out_ref.max())
        assert not np.allclose(out_before, out_ref, rtol=1e-4, atol=1e-6 * out_ref.max())

    @cpu_and_gpu
    def test_wf3_not_shared_with_different_subap_size(self, target_device_idx, xp):
        '''
        Two SH with the same number of subaps and FFT size, but a different
        number of pixels per subap, must not share the padded _wf3 buffer:
        the one with the larger subaps would write into the padding of the other.
        '''
        t = 1

        def make(pixel_pupil, fov_ovs_coeff):
            sh = SH(wavelengthInNm=589,
                    subap_wanted_fov=3.0,
                    sensor_pxscale=0.5,
                    subap_on_diameter=10,
                    subap_npx=6,
                    fov_ovs_coeff=fov_ovs_coeff,
                    target_device_idx=target_device_idx)
            ef = ElectricField(pixel_pupil, pixel_pupil, 0.05, S0=1,
                               target_device_idx=target_device_idx)
            ef.generation_time = t
            sh.inputs['in_ef'].set(ef)
            sh.setup()
            return sh

        def run(sh):
            sh.check_ready(t)
            sh.trigger()
            sh.post_trigger()
            return cpuArray(sh.outputs['out_i'].i).copy()

        SH._SH__zeros_cache.clear()
        sh_a = make(40, 3.0)
        sh_b = make(60, 1.0)
        assert sh_a._fft_size == sh_b._fft_size
        assert sh_a._ovs_np_sub > sh_b._ovs_np_sub
        assert sh_a._wf3 is not sh_b._wf3

        out_b = run(sh_b)
        run(sh_a)
        np.testing.assert_array_equal(run(sh_b), out_b)

    @cpu_and_gpu
    def test_oversampled_size_not_truncated(self, target_device_idx, xp):
        '''
        Test that the oversampled size is not truncated by float rounding:
        with 4 subaps and a 47 pixel pupil, the oversampled size is 96,
        but int(47 * (96 / 47)) == 95.
        '''
        t = 1
        ref_S0 = 100
        sh = SH(wavelengthInNm=500,
                subap_wanted_fov=3,
                sensor_pxscale=0.5,
                subap_on_diameter=4,
                subap_npx=6,
                target_device_idx=target_device_idx)

        ef = ElectricField(47, 47, 0.05, S0=ref_S0, target_device_idx=target_device_idx)
        ef.generation_time = t
        sh.inputs['in_ef'].set(ef)

        sh.setup()
        self.assertEqual(sh._ovs_ef_size, 96)
        self.assertEqual(sh.ef_row.shape[1], 96)

        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        intensity = sh.outputs['out_i']
        np.testing.assert_almost_equal(xp.sum(intensity.i), ref_S0 * ef.masked_area(), decimal=3)

    def test_sensor_pxscale_effective(self):
        '''
        The simulated sensor pixel scale is available in arcsec, and a warning
        is logged when it differs from the requested one by more than 1%
        '''
        def make(sensor_pxscale):
            sh = SH(wavelengthInNm=500,
                    subap_wanted_fov=6 * sensor_pxscale,
                    sensor_pxscale=sensor_pxscale,
                    subap_on_diameter=10,
                    subap_npx=6,
                    target_device_idx=-1)
            return sh, ElectricField(80, 80, 0.05, S0=1, target_device_idx=-1)

        # 0.27% difference: no warning
        sh, ef = make(0.3)
        with self.assertNoLogs('specula.SH', level='WARNING'):
            sh._set_in_ef(ef)
        self.assertAlmostEqual(sh.sensor_pxscale_effective, 0.3, delta=0.3 * 0.01)
        self.assertAlmostEqual(sh.subap_real_fov_arcsec, 6 * sh.sensor_pxscale_effective)

        # 1.8% difference: warning
        sh, ef = make(0.7)
        with self.assertLogs('specula.SH', level='WARNING') as logs:
            sh._set_in_ef(ef)
        self.assertGreater(abs(sh.sensor_pxscale_effective - 0.7) / 0.7, 0.01)
        self.assertTrue(any('Effective sensor pixel scale' in line for line in logs.output))

    def _lgs_sh(self, target_device_idx, xp, data_dir):
        '''
        LGS SH with sodium profile inputs, after setup(). Returns the SH, the
        sodium intensity input and a function that runs one step, refreshing
        all inputs as generators do, and returns a copy of the kernels.
        '''
        llt = LaserLaunchTelescope(simul_params=SimulParams(zenithAngleInDeg=30.0),
                                   spot_size=1.0, tel_position=[5.0, 0.0, 0.0],
                                   target_device_idx=target_device_idx)
        altitude = BaseValue(value=xp.array([85e3, 90e3, 95e3], dtype=xp.float32),
                             target_device_idx=target_device_idx)
        intensity = BaseValue(value=xp.array([0.2, 0.6, 0.2], dtype=xp.float32),
                              target_device_idx=target_device_idx)
        ef = ElectricField(80, 80, 0.1, S0=100, target_device_idx=target_device_idx)

        sh = SH(wavelengthInNm=589,
                subap_wanted_fov=6,
                sensor_pxscale=0.5,
                subap_on_diameter=10,
                subap_npx=12,
                laser_launch_tel=llt,
                data_dir=data_dir,
                target_device_idx=target_device_idx)
        sh.inputs['in_ef'].set(ef)
        sh.inputs['sodium_altitude'].set(altitude)
        sh.inputs['sodium_intensity'].set(intensity)
        sh.setup()

        def step(t):
            for obj in (ef, altitude, intensity):
                obj.generation_time = t
            sh.check_ready(t)
            sh.trigger()
            sh.post_trigger()
            return cpuArray(sh._kernelobj.kernels).copy()

        return sh, intensity, step

    @cpu_and_gpu
    def test_kernels_follow_in_place_sodium_update(self, target_device_idx, xp):
        '''
        Generators update the sodium profile in place at every step. The SH must
        detect the new values and recompute its kernels. On CPU, cpuArray() does
        not copy: the SH must keep its own copy of the last profile to compare with.
        '''
        with tempfile.TemporaryDirectory() as data_dir:
            _, intensity, step = self._lgs_sh(target_device_idx, xp, data_dir)
            kernels1 = step(1)
            intensity.value[:] = xp.array([0.5, 0.3, 0.2], dtype=xp.float32)
            kernels2 = step(2)

        self.assertFalse(np.allclose(kernels1, kernels2))

    @cpu_and_gpu
    def test_kernels_not_prepared_if_sodium_unchanged(self, target_device_idx, xp):
        '''
        Generators refresh the sodium profile at every step, even if its values
        do not change: the kernels must be prepared only at the first step.
        '''
        with tempfile.TemporaryDirectory() as data_dir:
            sh, _, step = self._lgs_sh(target_device_idx, xp, data_dir)
            with mock.patch.object(sh._kernelobj, 'prepare_for_sh',
                                   wraps=sh._kernelobj.prepare_for_sh) as prepare:
                for t in range(1, 5):
                    step(t)

        self.assertEqual(prepare.call_count, 1)

    @unittest.skipIf(specula.cp is None, 'GPU not available')
    def test_shared_kernels_in_cuda_graph(self):
        '''
        Two GPU SH objects with the same kernel share it. When the sodium
        profile of one of them changes, it gets new kernels and must capture
        its CUDA graph again: its output must match an SH created with the
        new profile, while the other one is unaffected.
        '''
        xp = specula.cp
        new_profile = xp.array([0.5, 0.3, 0.2], dtype=xp.float32)

        def out(sh):
            return cpuArray(sh.outputs['out_i'].i).copy()

        with tempfile.TemporaryDirectory() as data_dir:
            sh1, _, step1 = self._lgs_sh(0, xp, data_dir)
            sh2, intensity2, step2 = self._lgs_sh(0, xp, data_dir)
            for t in (1, 2):
                step1(t)
                step2(t)
            self.assertIs(sh1._kernelobj.kernels, sh2._kernelobj.kernels)
            self.assertIsNotNone(sh2.cuda_graph)
            out1 = out(sh1)

            intensity2.value[:] = new_profile
            for t in (3, 4):
                step1(t)
                step2(t)
            self.assertIsNot(sh1._kernelobj.kernels, sh2._kernelobj.kernels)

            ref_sh, ref_intensity, ref_step = self._lgs_sh(0, xp, data_dir)
            ref_intensity.value[:] = new_profile
            ref_step(1)

            np.testing.assert_allclose(out(sh1), out1, rtol=1e-5)
            np.testing.assert_allclose(out(sh2), out(ref_sh), rtol=1e-4,
                                       atol=1e-6 * out(ref_sh).max())
            self.assertFalse(np.allclose(out(sh2), out1, rtol=1e-4))

    def test_choose_fov_resolution(self):
        '''
        The resolution is turbulence_pxscale / k. Here all candidates k = 3...12
        give the exact FoV: with 6 pixels the first one (k=3) has the smallest
        L.C.M. / k ratio and is chosen, while with 8 pixels the L.C.M. criterion
        prefers k=4 (L.C.M. 16 instead of 24).
        '''
        turb = 0.5
        self.assertEqual(choose_fov_resolution(turb, 0.2, 2.0, 6), turb / 3)
        self.assertEqual(choose_fov_resolution(turb, 0.2, 2.0, 8), turb / 4)

        # The resolution is always finer than the sensor pixel scale
        for sensor_pxscale in (0.1, 0.2, 0.3, 0.45):
            res = choose_fov_resolution(turb, sensor_pxscale, 2.0, 8)
            self.assertLess(res, sensor_pxscale)
            self.assertAlmostEqual(turb / res, round(turb / res))
