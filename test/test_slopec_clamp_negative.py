"""
Tests of the ``clamp_negative`` option of ShSlopec and PyrSlopec.

With ``clamp_negative=True`` (default) the pixels that are negative after the
threshold subtraction are set to zero. With ``clamp_negative=False`` they are
kept. The tests use pixel frames built directly (no optical simulation): a known
noiseless signal plus zero-mean Gaussian read-out noise, with many subapertures
having identical statistics in one frame and a fixed RNG seed. The expected
values are analytic:

* Without clamp the estimators are linear: the mean slope is the noiseless
  slope and the variance of A+B-C-D is 4 sigma^2 / norm^2.
* With clamp each pixel (flux f, noise sigma) has mean
  E[max(f+n, 0)] = f Phi(f/sigma) + sigma phi(f/sigma)
  and, for f = 0 (the only case used), variance sigma^2 (1/2 - 1/(2 pi)).
  The signal gain is therefore lower than 1.
"""
import specula
specula.init(0)  # Default target device

import math
import unittest

from specula import np
from specula import cpuArray

from specula.loop_control import LoopControl
from specula.data_objects.pixels import Pixels
from specula.data_objects.pupdata import PupData
from specula.data_objects.subap_data import SubapData
from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams
from specula.processing_objects.ciao_ciao_slopec import CiaoCiaoSlopec
from specula.processing_objects.cur_wfs_slopec import CurWfsSlopec
from specula.processing_objects.double_roof_slopec import DoubleRoofSlopec
from specula.processing_objects.pyr_slopec import PyrSlopec
from specula.processing_objects.zwfs_slopec import ZwfsSlopec
from specula.processing_objects.sh_slopec import ShSlopec

from test.specula_testlib import cpu_and_gpu

SEED = 12345
SIGMA = 1.0
NP_SUB = 4  # SH subaperture size (pixels)


def _phi(u):
    return math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi)


def _Phi(u):
    return 0.5 * (1 + math.erf(u / math.sqrt(2)))


def clamped_mean(f, sigma=SIGMA):
    """E[max(f + n, 0)], n ~ N(0, sigma^2)."""
    u = f / sigma
    return f * _Phi(u) + sigma * _phi(u)


def _run(slopec, frame, xp, target_device_idx):
    pixels = Pixels(*frame.shape, target_device_idx=target_device_idx)
    pixels.pixels = xp.asarray(frame, dtype=xp.float32)
    pixels.generation_time = 1
    slopec.inputs['in_pixels'].set(pixels)
    slopec.check_ready(1)
    slopec.trigger()
    slopec.post_trigger()
    return slopec.outputs['out_slopes']


def _run2(slopec, frame1, frame2, xp, target_device_idx):
    """Same as _run, for the two-input CurWfsSlopec."""
    for name, frame in (('in_pixels1', frame1), ('in_pixels2', frame2)):
        pixels = Pixels(*frame.shape, target_device_idx=target_device_idx)
        pixels.pixels = xp.asarray(frame, dtype=xp.float32)
        pixels.generation_time = 1
        slopec.inputs[name].set(pixels)
    slopec.check_ready(1)
    slopec.trigger()
    slopec.post_trigger()
    return slopec.outputs['out_slopes']


# ---------------------------------------------------------------- SH helpers

def make_sh(n_subaps, target_device_idx, xp, **kwargs):
    """ShSlopec on a frame of shape (n_subaps*NP_SUB, NP_SUB): one subaperture per NP_SUB rows."""
    idxs = np.arange(n_subaps * NP_SUB * NP_SUB).reshape(n_subaps, NP_SUB * NP_SUB)
    subapdata = SubapData(idxs=idxs, display_map=np.arange(n_subaps), nx=n_subaps, ny=1,
                          target_device_idx=target_device_idx)
    return ShSlopec(subapdata, target_device_idx=target_device_idx, **kwargs)


def sh_frame(subap_pixels):
    """(n_subaps, NP_SUB, NP_SUB) -> frame (n_subaps*NP_SUB, NP_SUB)."""
    n = subap_pixels.shape[0]
    return subap_pixels.reshape(n * NP_SUB, NP_SUB)


def sh_weights(slopec):
    w = cpuArray(slopec._weights)  # rows: denominator mask, x weights, y weights
    return w[0].reshape(NP_SUB, NP_SUB), w[1].reshape(NP_SUB, NP_SUB), w[2].reshape(NP_SUB, NP_SUB)


def sh_reference(subap_pixels, wx, wy):
    """Float64 reference: (sx, sy) = weighted sums / total, zero below 1e-3 * mean(total)."""
    tot = subap_pixels.sum(axis=(1, 2))
    sx = (subap_pixels * wx).sum(axis=(1, 2))
    sy = (subap_pixels * wy).sum(axis=(1, 2))
    bad = tot < 1e-3 * tot.mean()
    safe = np.where(bad, 1.0, tot)
    return np.where(bad, 0, sx / safe), np.where(bad, 0, sy / safe)


# ---------------------------------------------------------------- Pyramid helpers

def make_pyr(m, target_device_idx, xp, **kwargs):
    """PyrSlopec with m*m subapertures on a frame (2m, 2m): one pupil per quadrant.

    Pupil 0 is top-right (as required by PupData.single_mask), then 1 top-left,
    2 bottom-left, 3 bottom-right.
    """
    ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()
    fs = 2 * m
    ind_pup = np.stack([ii * fs + jj + m,
                        ii * fs + jj,
                        (ii + m) * fs + jj,
                        (ii + m) * fs + jj + m], axis=1)
    pupdata = PupData(ind_pup=ind_pup, framesize=(fs, fs), target_device_idx=target_device_idx)
    return PyrSlopec(pupdata, target_device_idx=target_device_idx, **kwargs)


def pyr_frame(A, B, C, D):
    """Four (m, m) arrays -> frame (2m, 2m)."""
    m = A.shape[0]
    f = np.zeros((2 * m, 2 * m))
    f[:m, m:] = A
    f[:m, :m] = B
    f[m:, :m] = C
    f[m:, m:] = D
    return f


def pyr_quadrants(frame):
    m = frame.shape[0] // 2
    return frame[:m, m:], frame[:m, :m], frame[m:, :m], frame[m:, m:]


class TestSlopecClampNegative(unittest.TestCase):

    # ------------------------------------------------------------ 1. default unchanged

    @cpu_and_gpu
    def test_sh_clamp_true_equals_clamp_then_compute(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        n = 200
        frame = sh_frame(rng.normal(1.0, SIGMA, (n, NP_SUB, NP_SUB)))
        assert (frame < 0).any()
        clamped = np.maximum(frame, 0)

        # Default is True
        s_default = _run(make_sh(n, target_device_idx, xp), frame, xp, target_device_idx)
        s_true = _run(make_sh(n, target_device_idx, xp, clamp_negative=True), frame, xp, target_device_idx)
        # Explicit clamp done outside, no clamp inside
        s_pre = _run(make_sh(n, target_device_idx, xp, clamp_negative=False), clamped, xp, target_device_idx)

        np.testing.assert_array_equal(cpuArray(s_default.slopes), cpuArray(s_true.slopes))
        np.testing.assert_allclose(cpuArray(s_true.slopes), cpuArray(s_pre.slopes), rtol=1e-6, atol=1e-7)

        # and both agree with the float64 reference
        slopec = make_sh(n, target_device_idx, xp)
        _, wx, wy = sh_weights(slopec)
        sx, sy = sh_reference(clamped.reshape(n, NP_SUB, NP_SUB), wx, wy)
        np.testing.assert_allclose(cpuArray(s_true.xslopes), sx, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpuArray(s_true.yslopes), sy, rtol=1e-4, atol=1e-5)

    @cpu_and_gpu
    def test_sh_no_negative_pixels_same_result(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        n = 50
        frame = sh_frame(rng.uniform(0.1, 10, (n, NP_SUB, NP_SUB)))
        s_t = _run(make_sh(n, target_device_idx, xp, clamp_negative=True), frame, xp, target_device_idx)
        s_f = _run(make_sh(n, target_device_idx, xp, clamp_negative=False), frame, xp, target_device_idx)
        np.testing.assert_array_equal(cpuArray(s_t.slopes), cpuArray(s_f.slopes))

    @cpu_and_gpu
    def test_pyr_clamp_true_equals_clamp_then_compute(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        m = 20
        frame = rng.normal(1.0, SIGMA, (2 * m, 2 * m))
        assert (frame < 0).any()
        clamped = np.maximum(frame, 0)
        for kwargs in ({}, {'norm_factor': 3.0}, {'shlike': True}):
            with self.subTest(**kwargs):
                s_default = _run(make_pyr(m, target_device_idx, xp, **kwargs), frame, xp, target_device_idx)
                s_true = _run(make_pyr(m, target_device_idx, xp, clamp_negative=True, **kwargs),
                              frame, xp, target_device_idx)
                s_pre = _run(make_pyr(m, target_device_idx, xp, clamp_negative=False, **kwargs),
                             clamped, xp, target_device_idx)
                np.testing.assert_array_equal(cpuArray(s_default.slopes), cpuArray(s_true.slopes))
                np.testing.assert_array_equal(cpuArray(s_true.slopes), cpuArray(s_pre.slopes))

    @cpu_and_gpu
    def test_pyr_no_negative_pixels_same_result(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        m = 10
        frame = rng.uniform(0.1, 10, (2 * m, 2 * m))
        for kwargs in ({}, {'norm_factor': 3.0}, {'shlike': True}):
            with self.subTest(**kwargs):
                s_t = _run(make_pyr(m, target_device_idx, xp, clamp_negative=True, **kwargs),
                           frame, xp, target_device_idx)
                s_f = _run(make_pyr(m, target_device_idx, xp, clamp_negative=False, **kwargs),
                           frame, xp, target_device_idx)
                np.testing.assert_array_equal(cpuArray(s_t.slopes), cpuArray(s_f.slopes))

    # ------------------------------------------------------------ 2-3. gain

    @cpu_and_gpu
    def test_pyr_gain_unbiased_without_clamp_and_lowered_with_clamp(self, target_device_idx, xp):
        """Constant norm_factor: sx = (A+B-C-D)/norm is linear without clamp."""
        rng = np.random.default_rng(SEED)
        m = 200  # 40000 subapertures with identical statistics
        norm = 1.0
        # Noiseless pupil fluxes of a small displacement, comparable to sigma
        fA, fB, fC, fD = 1.0 * SIGMA, 0.8 * SIGMA, 0.4 * SIGMA, 0.2 * SIGMA
        flux = [np.full((m, m), f) for f in (fA, fB, fC, fD)]
        frame = pyr_frame(*flux) + rng.normal(0, SIGMA, (2 * m, 2 * m))

        sx_lin = (fA + fB - fC - fD) / norm
        sy_lin = (fB + fC - fA - fD) / norm
        eA, eB, eC, eD = (clamped_mean(f) for f in (fA, fB, fC, fD))
        sx_clamp = (eA + eB - eC - eD) / norm
        sy_clamp = (eB + eC - eA - eD) / norm
        # Standard error of the mean of one slope: 2 sigma / norm / sqrt(N)
        tol = 5 * 2 * SIGMA / norm / m

        s_f = _run(make_pyr(m, target_device_idx, xp, norm_factor=norm, clamp_negative=False),
                   frame, xp, target_device_idx)
        s_t = _run(make_pyr(m, target_device_idx, xp, norm_factor=norm, clamp_negative=True),
                   frame, xp, target_device_idx)

        np.testing.assert_allclose(cpuArray(s_f.xslopes).mean(), sx_lin, atol=tol)
        np.testing.assert_allclose(cpuArray(s_f.yslopes).mean(), sy_lin, atol=tol)
        np.testing.assert_allclose(cpuArray(s_t.xslopes).mean(), sx_clamp, atol=tol)
        np.testing.assert_allclose(cpuArray(s_t.yslopes).mean(), sy_clamp, atol=tol)
        # The clamp lowers the gain (gain ~0.72 here), well beyond the MC error
        gain = cpuArray(s_t.xslopes).mean() / sx_lin
        self.assertLess(gain, 0.8)
        self.assertGreater(sx_lin - cpuArray(s_t.xslopes).mean(), 3 * tol)

    @cpu_and_gpu
    def test_sh_gain_unbiased_without_clamp_and_lowered_with_clamp(self, target_device_idx, xp):
        """
        The SH denominator depends on the noise, so the slope is not exactly linear.
        Its fluctuation is made negligible with two mirror-symmetric bright
        columns (x weights w and -w) of flux F >> sigma: they add 8 F to the
        denominator (noise ~ 4 sigma / (8 F) = 1%) and cancel in the numerator.
        The signal is in the low-flux pixels: column 3 has flux f, column 0 zero.
        Expected slope = E[sum w p] / E[sum p], with E[p] = f (no clamp) or
        clamped_mean(f) (clamp); the ratio-of-means bias is O(1e-4) relative.
        """
        rng = np.random.default_rng(SEED)
        n = 20000
        F = 50 * SIGMA
        f = 0.5 * SIGMA
        flux = np.zeros((NP_SUB, NP_SUB))
        flux[:, 1] = F
        flux[:, 2] = F
        flux[:, 3] = f
        frame = sh_frame(flux[None] + rng.normal(0, SIGMA, (n, NP_SUB, NP_SUB)))

        s_f = _run(make_sh(n, target_device_idx, xp, clamp_negative=False), frame, xp, target_device_idx)
        s_t = _run(make_sh(n, target_device_idx, xp, clamp_negative=True), frame, xp, target_device_idx)
        slopec = make_sh(n, target_device_idx, xp)
        _, wx, _ = sh_weights(slopec)

        def expected(mean_pix):
            return (wx * mean_pix).sum() / mean_pix.sum()

        mean_lin = flux
        mean_clamp = np.vectorize(clamped_mean)(flux)
        sx_lin, sx_clamp = expected(mean_lin), expected(mean_clamp)
        # Std of one slope ~ sigma sqrt(sum w^2) / total; take the measured one for the tolerance
        sx_f = cpuArray(s_f.xslopes)
        sx_t = cpuArray(s_t.xslopes)
        tol_f = 5 * sx_f.std() / math.sqrt(n)
        tol_t = 5 * sx_t.std() / math.sqrt(n)

        np.testing.assert_allclose(sx_f.mean(), sx_lin, atol=tol_f)
        np.testing.assert_allclose(sx_t.mean(), sx_clamp, atol=tol_t)
        # Clamp: gain well below 1 (about 0.6)
        self.assertLess(sx_t.mean() / sx_lin, 0.8)
        self.assertGreater(sx_t.mean() / sx_lin, 0.4)
        # Unbiased without clamp, to better than 10% of the signal
        self.assertAlmostEqual(sx_f.mean() / sx_lin, 1.0, delta=0.1)

    # ------------------------------------------------------------ 4. noise

    @cpu_and_gpu
    def test_pyr_noise_variance(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        m = 200
        sigma = 3.0
        norm = 2.0
        frame = rng.normal(0, sigma, (2 * m, 2 * m))
        s_f = _run(make_pyr(m, target_device_idx, xp, norm_factor=norm, clamp_negative=False),
                   frame, xp, target_device_idx)
        s_t = _run(make_pyr(m, target_device_idx, xp, norm_factor=norm, clamp_negative=True),
                   frame, xp, target_device_idx)
        N = m * m
        # Relative scatter of a variance estimate: sqrt(2/N) = 0.7%; tolerance 5%
        var_exp = 4 * sigma**2 / norm**2
        for s in (s_f.xslopes, s_f.yslopes):
            np.testing.assert_allclose(cpuArray(s).var(), var_exp, rtol=0.05)
            np.testing.assert_allclose(cpuArray(s).mean(), 0, atol=5 * math.sqrt(var_exp / N))
        # With clamp: var[max(n, 0)] = sigma^2 (1/2 - 1/(2 pi)), mean sigma/sqrt(2 pi) cancels in A+B-C-D
        var_clamp = 4 * sigma**2 * (0.5 - 1 / (2 * math.pi)) / norm**2
        np.testing.assert_allclose(cpuArray(s_t.xslopes).var(), var_clamp, rtol=0.05)
        self.assertLess(cpuArray(s_t.xslopes).var(), 0.8 * cpuArray(s_f.xslopes).var())

    # ------------------------------------------------------------ 5. SH denominators

    @cpu_and_gpu
    def test_sh_negative_or_small_denominator_gives_zero_slopes(self, target_device_idx, xp):
        ramp = np.linspace(-1, 1, NP_SUB)[None, :] * np.ones((NP_SUB, 1))  # non-zero numerator
        subaps = np.empty((6, NP_SUB, NP_SUB))
        for k in range(4):
            subaps[k] = 100 + (k + 1) * ramp          # regular, positive
        subaps[4] = -10 + 3 * ramp                    # negative total (-160)
        subaps[5] = 1e-3 + 1e-4 * ramp                # total 0.016 < 1e-3 * mean (~1)
        frame = sh_frame(subaps)

        for clamp in (False, True):
            with self.subTest(clamp_negative=clamp):
                slopec = make_sh(6, target_device_idx, xp, clamp_negative=clamp)
                s = _run(slopec, frame, xp, target_device_idx)
                sl = cpuArray(s.slopes)
                self.assertTrue(np.all(np.isfinite(sl)))
                xs, ys = cpuArray(s.xslopes), cpuArray(s.yslopes)
                self.assertEqual(xs[4], 0)
                self.assertEqual(xs[5], 0)
                self.assertEqual(ys[4], 0)
                self.assertEqual(ys[5], 0)
                # Regular subapertures are non-zero and match the reference
                _, wx, wy = sh_weights(slopec)
                ref = np.maximum(subaps, 0) if clamp else subaps
                rx, ry = sh_reference(ref, wx, wy)
                self.assertTrue(np.all(xs[:4] != 0))
                np.testing.assert_allclose(xs[:4], rx[:4], rtol=1e-4)
                np.testing.assert_allclose(ys[:4], ry[:4], atol=1e-6)

    @cpu_and_gpu
    def test_sh_negative_denominator_not_flipped(self, target_device_idx, xp):
        """Regression: a negative denominator must not give a (huge, sign-flipped) slope."""
        subaps = np.full((3, NP_SUB, NP_SUB), 50.0)
        subaps[2] = -1.0
        subaps[2, :, 3] = -0.5  # negative total, non-zero x numerator
        s = _run(make_sh(3, target_device_idx, xp, clamp_negative=False), sh_frame(subaps), xp,
                 target_device_idx)
        xs = cpuArray(s.xslopes)
        self.assertEqual(xs[2], 0)
        self.assertTrue(np.all(np.isfinite(cpuArray(s.slopes))))

    # ------------------------------------------------------------ 6. pyramid shlike

    @cpu_and_gpu
    def test_pyr_shlike_zero_flux_gives_zero_slopes(self, target_device_idx, xp):
        m = 2  # 4 subapertures
        # Per subaperture (A, B, C, D)
        vals = np.array([[4.0, 3.0, 2.0, 1.0],     # regular, flux 10
                         [-4.0, -3.0, -2.0, -1.0], # negative flux, non-zero numerator
                         [2.0, -1.0, 0.0, -1.0],   # flux exactly 0, non-zero numerator
                         [1.0, 2.0, 3.0, 4.0]])    # regular, flux 10
        A, B, C, D = (vals[:, i].reshape(m, m) for i in range(4))
        frame = pyr_frame(A, B, C, D)

        s = _run(make_pyr(m, target_device_idx, xp, shlike=True, clamp_negative=False),
                 frame, xp, target_device_idx)
        xs, ys = cpuArray(s.xslopes), cpuArray(s.yslopes)
        self.assertTrue(np.all(np.isfinite(cpuArray(s.slopes))))
        self.assertEqual(xs[1], 0)
        self.assertEqual(ys[1], 0)
        self.assertEqual(xs[2], 0)
        self.assertEqual(ys[2], 0)
        # Regular subapertures: (A+B-C-D)/flux and (B+C-A-D)/flux
        np.testing.assert_allclose(xs[[0, 3]], [(4 + 3 - 2 - 1) / 10, (1 + 2 - 3 - 4) / 10], rtol=1e-6)
        np.testing.assert_allclose(ys[[0, 3]], [(3 + 2 - 4 - 1) / 10, (2 + 3 - 1 - 4) / 10], atol=1e-6)

    @cpu_and_gpu
    def test_pyr_shlike_clamped_dark_subaperture_is_zero(self, target_device_idx, xp):
        """Default (clamp): fully negative or dark subapertures give zero, not NaN."""
        m = 2
        vals = np.array([[4.0, 3.0, 2.0, 1.0],
                         [-4.0, -3.0, -2.0, -1.0],
                         [0.0, 0.0, 0.0, 0.0],
                         [1.0, 2.0, 3.0, 4.0]])
        A, B, C, D = (vals[:, i].reshape(m, m) for i in range(4))
        s = _run(make_pyr(m, target_device_idx, xp, shlike=True), pyr_frame(A, B, C, D), xp,
                 target_device_idx)
        xs, ys = cpuArray(s.xslopes), cpuArray(s.yslopes)
        self.assertTrue(np.all(np.isfinite(cpuArray(s.slopes))))
        np.testing.assert_array_equal(xs[[1, 2]], 0)
        np.testing.assert_array_equal(ys[[1, 2]], 0)

    # ------------------------------------------------------------ pyramid global normalization

    @staticmethod
    def _pyr_frame_from_vals(vals):
        m = int(round(math.sqrt(len(vals))))
        return m, pyr_frame(*(vals[:, i].reshape(m, m) for i in range(4)))

    @cpu_and_gpu
    def test_pyr_global_norm_negative_total_gives_zero_slopes(self, target_device_idx, xp):
        vals = np.array([[-1.0, -4.0, -2.0, -6.0],   # non-zero numerators
                         [1.0, -5.0, -3.0, -2.0],
                         [-2.0, 1.0, -7.0, -1.0],
                         [-3.0, -1.0, 2.0, -4.0]])
        assert vals.sum() < 0
        m, frame = self._pyr_frame_from_vals(vals)
        s = _run(make_pyr(m, target_device_idx, xp, clamp_negative=False), frame, xp, target_device_idx)
        sl = cpuArray(s.slopes)
        self.assertTrue(np.all(np.isfinite(sl)))
        np.testing.assert_array_equal(sl, 0)

    @cpu_and_gpu
    def test_pyr_global_norm_zero_total_gives_zero_slopes(self, target_device_idx, xp):
        vals = np.array([[2.0, -1.0, 0.0, -1.0],     # non-zero numerators, total 0
                         [-3.0, 1.0, 1.0, 1.0],
                         [1.0, 1.0, -4.0, 2.0],
                         [0.0, 3.0, -2.0, -1.0]])
        assert vals.sum() == 0
        m, frame = self._pyr_frame_from_vals(vals)
        for clamp in (False, True):
            with self.subTest(clamp_negative=clamp):
                s = _run(make_pyr(m, target_device_idx, xp, clamp_negative=clamp), frame, xp,
                         target_device_idx)
                sl = cpuArray(s.slopes)
                self.assertTrue(np.all(np.isfinite(sl)))
                if not clamp:
                    np.testing.assert_array_equal(sl, 0)

    @cpu_and_gpu
    def test_pyr_global_norm_positive_total_with_negative_pixels(self, target_device_idx, xp):
        vals = np.array([[5.0, 4.0, -1.0, -2.0],
                         [6.0, -3.0, 2.0, 1.0],
                         [-2.0, 7.0, 3.0, 4.0],
                         [1.0, 2.0, 3.0, -1.0]])
        total = vals.sum()
        assert total > 0 and (vals < 0).any()
        m, frame = self._pyr_frame_from_vals(vals)
        A, B, C, D = vals.T
        norm = total / len(vals)
        s = _run(make_pyr(m, target_device_idx, xp, clamp_negative=False), frame, xp, target_device_idx)
        np.testing.assert_allclose(cpuArray(s.xslopes), (A + B - C - D) / norm, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(cpuArray(s.yslopes), (B + C - A - D) / norm, rtol=1e-5, atol=1e-6)

    # ------------------------------------------------------------ all classes accept the option

    @cpu_and_gpu
    def test_all_classes_store_clamp_negative(self, target_device_idx, xp):
        d = target_device_idx
        pupdata = PupData(target_device_idx=d)
        pupdata.ind_pup = xp.array([[1, 3, 6, 8], [15, 16, 21, 24]], dtype=int)
        pupdata.framesize = (4, 4)
        mask = Pupilstop(SimulParams(pixel_pupil=32, pixel_pitch=1.0),
                         input_mask=np.ones((32, 32), dtype=np.float32))
        builders = {
            'sh': lambda **kw: make_sh(4, d, xp, **kw),
            'pyr': lambda **kw: make_pyr(2, d, xp, **kw),
            'doubleroof': lambda **kw: DoubleRoofSlopec(pupdata, target_device_idx=d, **kw),
            'zwfs': lambda **kw: ZwfsSlopec(pup_diam=3, ccd_size=5, target_device_idx=d, **kw),
            'curwfs': lambda **kw: CurWfsSlopec(diameter=8, ccd_size=(10, 10), target_device_idx=d, **kw),
            'ciaociao': lambda **kw: CiaoCiaoSlopec(wavelength_in_nm=2200.0, window_x_in_pix=26,
                                                    window_y_in_pix=21, window_sigma_in_pix=3.0,
                                                    pupil_mask=mask, target_device_idx=d, **kw),
        }
        for name, build in builders.items():
            with self.subTest(cls=name):
                self.assertIs(build().clamp_negative, True)
                self.assertIs(build(clamp_negative=True).clamp_negative, True)
                self.assertIs(build(clamp_negative=False).clamp_negative, False)

    # ------------------------------------------------------------ ZWFS

    @staticmethod
    def _zwfs_ref(frame, pup_idx, thr, clamp):
        meta = frame.ravel()[pup_idx].astype(np.float64) - thr
        if clamp:
            meta = np.maximum(meta, 0)
        return meta / (meta.sum() / len(meta))

    @cpu_and_gpu
    def test_zwfs_clamp(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        size, thr = 8, 0.5
        frame = rng.normal(1.0, SIGMA, (size, size))
        for clamp in (False, True):
            with self.subTest(clamp_negative=clamp):
                slopec = ZwfsSlopec(pup_diam=6, ccd_size=size, thr_value=thr, clamp_negative=clamp,
                                    target_device_idx=target_device_idx)
                pup_idx = cpuArray(slopec.pup_idx)
                raw = frame.ravel()[pup_idx] - thr
                assert (raw < 0).any() and raw.sum() > 0
                s = _run(slopec, frame, xp, target_device_idx)
                np.testing.assert_allclose(cpuArray(s.slopes),
                                           self._zwfs_ref(frame.astype(np.float32), pup_idx, thr, clamp),
                                           rtol=1e-4, atol=1e-6)
        # Without negative pixels, True == False
        pos = rng.uniform(1, 5, (size, size))
        s_t = _run(ZwfsSlopec(pup_diam=6, ccd_size=size, clamp_negative=True,
                              target_device_idx=target_device_idx), pos, xp, target_device_idx)
        s_f = _run(ZwfsSlopec(pup_diam=6, ccd_size=size, clamp_negative=False,
                              target_device_idx=target_device_idx), pos, xp, target_device_idx)
        np.testing.assert_array_equal(cpuArray(s_t.slopes), cpuArray(s_f.slopes))

    @cpu_and_gpu
    def test_zwfs_nonpositive_total_gives_zero_slopes(self, target_device_idx, xp):
        size = 8
        probe = ZwfsSlopec(pup_diam=6, ccd_size=size, target_device_idx=target_device_idx)
        pup_idx = cpuArray(probe.pup_idx)
        n = len(pup_idx)
        ramp = np.arange(n) - (n - 1) / 2.0   # exact in single precision, sums to 0
        for name, values in (('negative', ramp - n), ('zero', ramp)):
            frame = np.zeros(size * size)
            frame[pup_idx] = values
            frame = frame.reshape(size, size)
            for clamp in (False, True):
                with self.subTest(total=name, clamp_negative=clamp):
                    slopec = ZwfsSlopec(pup_diam=6, ccd_size=size, clamp_negative=clamp,
                                        target_device_idx=target_device_idx)
                    sl = cpuArray(_run(slopec, frame, xp, target_device_idx).slopes)
                    self.assertTrue(np.all(np.isfinite(sl)))
                    # Without clamp the total is <= 0 in both frames. With clamp it is
                    # zero only for the all-negative frame.
                    if not clamp or name == 'negative':
                        np.testing.assert_array_equal(sl, 0)

    # ------------------------------------------------------------ CurWFS

    @cpu_and_gpu
    def test_curwfs_clamp(self, target_device_idx, xp):
        rng = np.random.default_rng(SEED)
        size = 16
        p1 = rng.normal(1.0, SIGMA, (size, size))
        p2 = rng.normal(1.0, SIGMA, (size, size))
        probe = CurWfsSlopec(diameter=12, ccd_size=(size, size), target_device_idx=target_device_idx)
        pup_idx = cpuArray(probe.pup_idx)
        # Negative summed flux, and zero summed flux, in two pupil pixels
        p1.ravel()[pup_idx[0]], p2.ravel()[pup_idx[0]] = -1.0, -0.5
        p1.ravel()[pup_idx[1]], p2.ravel()[pup_idx[1]] = 1.0, -1.0
        p1, p2 = p1.astype(np.float32), p2.astype(np.float32)

        for clamp in (False, True):
            with self.subTest(clamp_negative=clamp):
                slopec = CurWfsSlopec(diameter=12, ccd_size=(size, size), clamp_negative=clamp,
                                      target_device_idx=target_device_idx)
                s = _run2(slopec, p1, p2, xp, target_device_idx)
                i1 = p1.ravel()[pup_idx].astype(np.float64)
                i2 = p2.ravel()[pup_idx].astype(np.float64)
                if clamp:
                    i1, i2 = np.maximum(i1, 0), np.maximum(i2, 0)
                tot = i1 + i2
                valid = tot >= 1e-9
                ref = np.where(valid, (i1 - i2) / np.where(valid, tot, 1), 0)
                sl = cpuArray(s.slopes)
                self.assertTrue(np.all(np.isfinite(sl)))
                np.testing.assert_allclose(sl, ref, rtol=1e-5, atol=1e-6)
                self.assertEqual(sl[0], 0)  # negative (False) or zero (True) summed flux
                self.assertEqual(sl[1], 0 if not clamp else 1.0)

        pos1 = rng.uniform(0.1, 5, (size, size))
        pos2 = rng.uniform(0.1, 5, (size, size))
        s_t = _run2(CurWfsSlopec(diameter=12, ccd_size=(size, size), clamp_negative=True,
                                 target_device_idx=target_device_idx), pos1, pos2, xp, target_device_idx)
        s_f = _run2(CurWfsSlopec(diameter=12, ccd_size=(size, size), clamp_negative=False,
                                 target_device_idx=target_device_idx), pos1, pos2, xp, target_device_idx)
        np.testing.assert_array_equal(cpuArray(s_t.slopes), cpuArray(s_f.slopes))

    # ------------------------------------------------------------ CiaoCiao

    @cpu_and_gpu
    def test_ciaociao_clamp(self, target_device_idx, xp):
        shape = (64, 64)
        y, x = np.indices(shape)
        interf = 1.0 + np.cos(2 * np.pi * (10.0 / 64 * x + 5.0 / 64 * y))
        mask = Pupilstop(SimulParams(pixel_pupil=64, pixel_pitch=1.0),
                         input_mask=np.ones(shape, dtype=np.float32))

        def build(clamp):
            return CiaoCiaoSlopec(wavelength_in_nm=2200.0, window_x_in_pix=42, window_y_in_pix=37,
                                  window_sigma_in_pix=3.0, pupil_mask=mask, unwrap=False,
                                  clamp_negative=clamp, target_device_idx=target_device_idx)

        noisy = (interf + np.random.default_rng(SEED).normal(0, 0.5, shape)).astype(np.float32)
        assert (noisy < 0).any()
        before = noisy.copy()

        def run_with_input(slopec, frame):
            pixels = Pixels(*shape, target_device_idx=target_device_idx)
            pixels.pixels = xp.asarray(frame)
            pixels.generation_time = pixels.seconds_to_t(1)
            slopec.inputs['in_pixels'].set(pixels)
            loop = LoopControl()  # runs setup(), needed to build the filtering window
            loop.add(slopec, idx=0)
            loop.run(run_time=2, dt=1, t0=1)
            return cpuArray(slopec.outputs['out_slopes'].slopes).copy(), cpuArray(pixels.pixels)

        s_default, in_after = run_with_input(build(True), noisy)
        s_pre, _ = run_with_input(build(False), np.maximum(noisy, 0))
        s_raw, _ = run_with_input(build(False), noisy)

        np.testing.assert_array_equal(in_after, before)   # input not modified
        np.testing.assert_allclose(s_default, s_pre, rtol=1e-5, atol=1e-5)
        self.assertGreater(np.abs(s_default - s_raw).max(), 1e-3)  # the clamp matters here

        clean = interf.astype(np.float32)
        a, _ = run_with_input(build(True), clean)
        b, _ = run_with_input(build(False), clean)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
