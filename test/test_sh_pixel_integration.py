import specula
specula.init(0)  # Default target device

import tempfile
import unittest
import unittest.mock

from specula import np
from specula import cpuArray

from specula.data_objects.electric_field import ElectricField
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.processing_objects.sh import SH
from specula.lib.pixel_integration import pixel_integration_matrix, integrate_pixels
from test.specula_testlib import cpu_and_gpu

WAVELENGTH_NM = 500.0
N_LENSES = 4
PUPIL_PX = 80
# lambda / d = 0.912 arcsec for a subaperture of 20 pupil pixels
PITCH = WAVELENGTH_NM * 1e-9 * 206264.80624709636 / 0.912 / 20


class TestSHPixelIntegration(unittest.TestCase):

    @staticmethod
    def _phase_nm(rms_nm=300.0, seed=1):
        """Smooth random phase: low-order 2D polynomial with seeded coefficients"""
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[:PUPIL_PX, :PUPIL_PX] / (PUPIL_PX - 1) * 2 - 1
        ph = np.zeros((PUPIL_PX, PUPIL_PX))
        for i in range(4):
            for j in range(4 - i):
                ph += rng.standard_normal() * x**i * y**j
        return ph / ph.std() * rms_nm

    def _make_sh(self, target_device_idx, wavefront, npx, pxscale, fov, **kwargs):
        sh = SH(wavelengthInNm=WAVELENGTH_NM, subap_wanted_fov=fov, sensor_pxscale=pxscale,
                subap_on_diameter=N_LENSES, subap_npx=npx,
                target_device_idx=target_device_idx, precision=0, **kwargs)
        ef = ElectricField(PUPIL_PX, PUPIL_PX, PITCH, S0=100, target_device_idx=target_device_idx,
                           precision=0)
        if wavefront is not None:
            ef.phaseInNm[:] = ef.to_xp(wavefront)
        ef.generation_time = 1
        sh.inputs['in_ef'].set(ef)
        sh.setup()
        return sh, ef

    @staticmethod
    def _run(sh, t=1):
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()
        return cpuArray(sh.outputs['out_i'].i).copy()

    @staticmethod
    def _subaps(img, npx):
        n = img.shape[0] // npx
        return img.reshape(n, npx, n, npx).swapaxes(1, 2).reshape(n * n, npx, npx)

    @staticmethod
    def _normalized(subaps):
        return subaps / subaps.sum(axis=(1, 2), keepdims=True)

    @staticmethod
    def _rel_l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    def _reference_image(self, sh, npx, pxscale, stop, q):
        """
        Detector image of every subaperture from a direct evaluation of the DTFT of
        the field used by the SH, on q x q sub-sample centres per detector pixel.
        """
        n = sh._ovs_np_sub
        period = sh._fft_period_arcsec
        wf = sh.ef_interpolator.interpolated_ef()
        field = cpuArray(wf.ef_at_lambda(WAVELENGTH_NM))
        # sub-sample centres along one axis (same for both axes)
        sub = (np.arange(npx * q) + 0.5) / q
        theta = (sub - npx / 2) * pxscale
        w = np.exp(-2j * np.pi * np.outer(theta, np.arange(n)) / period)
        weight = (pxscale / q) ** 2
        inside = np.ones_like(theta) if stop is None else (np.abs(theta) <= stop / 2).astype(float)
        out = np.zeros((N_LENSES * npx, N_LENSES * npx))
        for iy in range(N_LENSES):
            for ix in range(N_LENSES):
                e = field[iy * n:(iy + 1) * n, ix * n:(ix + 1) * n]
                f = w @ e @ w.T
                p = np.abs(f) ** 2 * inside[:, None] * inside[None, :] * weight
                out[iy * npx:(iy + 1) * npx, ix * npx:(ix + 1) * npx] = \
                    p.reshape(npx, q, npx, q).sum(axis=(1, 3))
        return out

    def _compare_with_reference(self, target_device_idx, npx, pxscale, fov, q, tol=1e-4):
        sh, _ = self._make_sh(target_device_idx, self._phase_nm(), npx, pxscale, fov)
        img = self._run(sh)
        stop = fov if fov < npx * pxscale else None
        ref = self._reference_image(sh, npx, pxscale, stop, q)
        a = self._normalized(self._subaps(img, npx))
        b = self._normalized(self._subaps(ref, npx))
        self.assertLess(self._rel_l2(a, b), tol)

    @cpu_and_gpu
    def test_coarse_pixels(self, target_device_idx, xp):
        self._compare_with_reference(target_device_idx, 4, 1.275, 5.1, q=64)

    @cpu_and_gpu
    def test_fine_pixels(self, target_device_idx, xp):
        # pixel scale below lambda / (2 d) = 0.456 arcsec
        self._compare_with_reference(target_device_idx, 8, 0.3, 2.4, q=64)

    @cpu_and_gpu
    def test_odd_number_of_pixels(self, target_device_idx, xp):
        self._compare_with_reference(target_device_idx, 5, 1.275, 6.375, q=64)

    @cpu_and_gpu
    def test_square_stop_smaller_than_sensor_fov(self, target_device_idx, xp):
        # The reference has a sharp edge: the midpoint rule limits it to ~3e-5 with q = 512
        self._compare_with_reference(target_device_idx, 4, 1.275, 3.9, q=512)

    @cpu_and_gpu
    def test_result_independent_of_fft_size(self, target_device_idx, xp):
        npx, pxscale, fov = 4, 1.275, 5.1
        phase = self._phase_nm()
        sh, _ = self._make_sh(target_device_idx, phase, npx, pxscale, fov)
        default = self._run(sh)
        n = sh._ovs_np_sub

        original = SH._set_in_ef

        def set_in_ef_4n(self_, in_ef):
            original(self_, in_ef)
            self_._fft_size = 4 * self_._ovs_np_sub
            self_._fov_resolution_arcsec = self_._fft_period_arcsec / self_._fft_size

        SH._set_in_ef = set_in_ef_4n
        try:
            sh4, _ = self._make_sh(target_device_idx, phase, npx, pxscale, fov)
        finally:
            SH._set_in_ef = original
        self.assertEqual(sh4._fft_size, 4 * n)
        big = self._run(sh4)
        self.assertLess(self._rel_l2(big, default), 1e-10)

    def _gaussian_reference(self, sh, npx, pxscale, fwhm):
        """
        Detector image of every subaperture for the PSF convolved with a Gaussian
        spot, computed independently of the SH kernel code: exact PSF Fourier
        coefficients (FFT of the field zero-padded by 2), multiplied by the
        analytic Fourier transform of the Gaussian, integrated over the pixels.
        """
        n = sh._ovs_np_sub
        period = sh._fft_period_arcsec
        n_fft = 2 * n
        field = cpuArray(sh.ef_interpolator.interpolated_ef().ef_at_lambda(WAVELENGTH_NM))
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        f = np.fft.fftfreq(n_fft) * n_fft / period
        khat = np.exp(-2 * np.pi ** 2 * sigma ** 2 * f ** 2)
        g = pixel_integration_matrix(n_fft, period, pxscale, npx)
        out = np.zeros((N_LENSES * npx, N_LENSES * npx))
        for iy in range(N_LENSES):
            for ix in range(N_LENSES):
                pad = np.zeros((n_fft, n_fft), dtype=complex)
                pad[:n, :n] = field[iy * n:(iy + 1) * n, ix * n:(ix + 1) * n]
                psf = np.abs(np.fft.fft2(pad)) ** 2
                conv = np.real(np.fft.ifft2(np.fft.fft2(psf) * khat[:, None] * khat[None, :]))
                out[iy * npx:(iy + 1) * npx, ix * npx:(ix + 1) * npx] = g @ conv @ g.T
        return out

    @cpu_and_gpu
    def test_lgs_gaussian_kernel_vs_analytic_reference(self, target_device_idx, xp):
        """
        LGS path: the kernel convolution, the N/2 roll of the integration matrix and the
        kernel sampling factor, against an analytic Gaussian spot. Spot sizes from much
        wider than the FFT sample to comparable with it (where aliasing would matter).
        The tolerance is limited by issue #837 (kernels narrower by (N-1)/N).
        """
        npx, pxscale, fov = 4, 1.275, 5.1
        phase = self._phase_nm()
        for fwhm in (1.8, 0.5, 0.25):
            with self.subTest(fwhm=fwhm), tempfile.TemporaryDirectory() as data_dir:
                llt = LaserLaunchTelescope(spot_size=fwhm, target_device_idx=target_device_idx,
                                           precision=0)
                sh, _ = self._make_sh(target_device_idx, phase, npx, pxscale, fov,
                                      laser_launch_tel=llt, data_dir=data_dir)
                img = self._normalized(self._subaps(self._run(sh), npx))
                ref = self._normalized(self._subaps(self._gaussian_reference(sh, npx, pxscale, fwhm), npx))
                # The kernels are narrower than requested by (N-1)/N (issue #837): the
                # error is ~1/N. With that fixed it is below 1e-7: then use 1e-6 here.
                self.assertLess(self._rel_l2(img, ref), 1.5 / sh._fft_size)

    @cpu_and_gpu
    def test_negative_pixels_are_clipped(self, target_device_idx, xp):
        # Rounding in G @ PSF @ G.T can give slightly negative dark pixels: emulate
        # them by removing the mean of each integrated row of subapertures
        def integrate_with_offset(*args, **kwargs):
            img = integrate_pixels(*args, **kwargs)
            return img - img.mean()

        with unittest.mock.patch('specula.processing_objects.sh.integrate_pixels',
                                 integrate_with_offset):
            sh, _ = self._make_sh(target_device_idx, self._phase_nm(), 4, 1.275, 5.1)
            img = self._run(sh)
        self.assertTrue(np.any(img == 0))
        self.assertGreaterEqual(img.min(), 0)

    @cpu_and_gpu
    def test_flat_wavefront_symmetry_and_flux(self, target_device_idx, xp):
        npx = 4
        sh, ef = self._make_sh(target_device_idx, None, npx, 1.275, 5.1)
        img = self._run(sh)
        subaps = self._normalized(self._subaps(img, npx))
        np.testing.assert_allclose(subaps, subaps[:, :, ::-1], atol=1e-12)
        np.testing.assert_allclose(subaps, subaps[:, ::-1, :], atol=1e-12)
        np.testing.assert_allclose(img.sum(), float(ef.S0 * ef.masked_area()), rtol=1e-12)


if __name__ == '__main__':
    unittest.main()
