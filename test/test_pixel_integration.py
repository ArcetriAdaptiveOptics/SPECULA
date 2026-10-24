import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.lib.pixel_integration import pixel_integration_matrix, integrate_pixels
from test.specula_testlib import cpu_and_gpu


class TestPixelIntegrationMatrix(unittest.TestCase):

    def test_shape_and_dtype(self):
        g = pixel_integration_matrix(32, 10.0, 1.0, 5)
        self.assertEqual(g.shape, (5, 32))
        self.assertFalse(np.iscomplexobj(g))

    def test_constant_intensity_gives_pixel_widths(self):
        for n_fft, npx in [(32, 4), (30, 5), (64, 6)]:
            g = pixel_integration_matrix(n_fft, 20.0, 1.275, npx)
            np.testing.assert_allclose(g @ np.ones(n_fft), np.full(npx, 1.275), atol=1e-12)

    def test_constant_intensity_with_stop_gives_clipped_widths(self):
        g = pixel_integration_matrix(32, 20.0, 1.0, 4, stop=3.0)
        np.testing.assert_allclose(g @ np.ones(32), [0.5, 1, 1, 0.5], atol=1e-12)

    def test_cosine_matches_analytic_integral(self):
        n_fft, period, pxscale, npx = 64, 12.0, 1.1, 5
        g = pixel_integration_matrix(n_fft, period, pxscale, npx)
        theta = np.arange(n_fft) * period / n_fft
        a = (np.arange(npx) - npx / 2) * pxscale
        b = a + pxscale
        for k0 in (1, 3, 10, 30):
            val = g @ np.cos(2 * np.pi * k0 * theta / period)
            c = period / (2 * np.pi * k0)
            ref = c * (np.sin(2 * np.pi * k0 * b / period) - np.sin(2 * np.pi * k0 * a / period))
            np.testing.assert_allclose(val, ref, atol=1e-12)

    def test_cosine_with_stop_matches_analytic_integral(self):
        n_fft, period, pxscale, npx, stop = 64, 12.0, 1.1, 4, 3.5
        g = pixel_integration_matrix(n_fft, period, pxscale, npx, stop=stop)
        theta = np.arange(n_fft) * period / n_fft
        a = np.clip((np.arange(npx) - npx / 2) * pxscale, -stop / 2, stop / 2)
        b = np.clip((np.arange(npx) + 1 - npx / 2) * pxscale, -stop / 2, stop / 2)
        k0 = 5
        c = period / (2 * np.pi * k0)
        ref = c * (np.sin(2 * np.pi * k0 * b / period) - np.sin(2 * np.pi * k0 * a / period))
        np.testing.assert_allclose(g @ np.cos(2 * np.pi * k0 * theta / period), ref, atol=1e-12)


class TestIntegratePixels(unittest.TestCase):

    @cpu_and_gpu
    def test_integrate_pixels_matches_matrix_products(self, target_device_idx, xp):
        """g @ I[b] @ g.T for a batch, also with a non-contiguous input,
        on CPU and GPU (raw kernel), against numpy matrix products"""
        rng = np.random.default_rng(3)
        nb, n_fft, npx = 5, 37, 6
        g = rng.standard_normal((npx, n_fft))
        intensity = rng.random((nb, n_fft, n_fft))
        expected = g[None] @ intensity @ g.T[None]
        for strided in (False, True):
            with self.subTest(strided=strided):
                inten = xp.asarray(intensity.swapaxes(1, 2).copy()).swapaxes(1, 2) if strided \
                    else xp.asarray(intensity)
                tmp = xp.zeros((nb, npx, n_fft))
                work = xp.zeros((nb, npx, npx))
                out = integrate_pixels(xp.asarray(g), inten, tmp, work, xp)
                np.testing.assert_allclose(cpuArray(out), expected, rtol=1e-12, atol=1e-12)

    @cpu_and_gpu
    def test_raw_kernel_rejects_unsupported_layouts(self, target_device_idx, xp):
        """The GPU kernel reads g row-major and uses the dtype of the output"""
        if xp is np:
            self.skipTest('the checks apply to the GPU kernel only')
        nb, n_fft, npx = 2, 8, 3
        g = xp.ones((npx, n_fft))
        intensity = xp.ones((nb, n_fft, n_fft))
        tmp = xp.zeros((nb, npx, n_fft))
        work = xp.zeros((nb, npx, npx))
        with self.assertRaises(ValueError):
            integrate_pixels(xp.ones((n_fft, npx)).T, intensity, tmp, work, xp)
        with self.assertRaises(ValueError):
            integrate_pixels(g, intensity.astype(xp.float32), tmp, work, xp)
