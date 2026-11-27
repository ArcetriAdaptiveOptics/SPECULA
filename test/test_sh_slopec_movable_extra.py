import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.sh_slopec_movable import ShSlopecMovable
from test.specula_testlib import cpu_and_gpu

NP_SUB = 32
RADIUS = 2.0


def gaussian_spot(n, sx, sy, sigma=1.5, amp=100.0):
    """Gaussian spot at (sx, sy) px from the subaperture centre (x = column, y = row)."""
    c = (n - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(n) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(n) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def reference_slopes(frame, wx, wy, radius, n):
    """Independent numpy WCoG with the Gaussian window on (wx, wy), slope zero on the window."""
    c = (n - 1) / 2.0
    sigma = 2.0 * radius / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    k = np.arange(n) - c
    mask = np.outer(np.exp(-0.5 * ((k - wy) / sigma) ** 2), np.exp(-0.5 * ((k - wx) / sigma) ** 2))
    mask /= mask.max()
    mask[mask < 1e-6] = 0.0
    tot = (frame * mask).sum()
    px = 2.0 / n
    sx = (frame * mask * k[None, :] * px).sum() / tot - wx * px
    sy = (frame * mask * k[:, None] * px).sum() / tot - wy * px
    return sx, sy


class Rig:
    """One persistent ShSlopecMovable driven step after step (state is kept between calls)."""

    def __init__(self, target_device_idx, xp, n=NP_SUB, cls=ShSlopecMovable, radius=RADIUS,
                 precision=None, **kw):
        self.xp, self.dev, self.n = xp, target_device_idx, n
        idx = np.where(np.ones((n, n)) == 1)
        v = np.zeros((1, n * n), dtype=int)
        v[0] = np.ravel_multi_index(idx, (n, n))
        subap = SubapData(idxs=v, display_map=np.zeros(1, dtype=int), nx=1, ny=1,
                          target_device_idx=target_device_idx)
        self.slopec = cls(subap, weightedPixRad=radius, thr_value=0,
                          target_device_idx=target_device_idx, precision=precision, **kw)
        self.t = 0

    def step(self, frame, window=None, t=None):
        self.t = self.t + 1 if t is None else t
        s = self.slopec
        pixels = Pixels(*np.shape(frame), target_device_idx=self.dev)
        pixels.pixels = self.xp.array(frame)
        pixels.generation_time = self.t
        s.inputs['in_pixels'].set(pixels)
        if window is not None:
            win = BaseValue(value=self.xp.array(window, dtype=float), target_device_idx=self.dev)
            win.generation_time = self.t
            s.inputs['in_window'].set(win)
        s.check_ready(self.t)
        s.trigger()
        s.post_trigger()
        return (cpuArray(s.outputs['out_slopes'].xslopes).copy(),
                cpuArray(s.outputs['out_slopes'].yslopes).copy())


def fresh(target_device_idx, xp, frame, window=None, cls=ShSlopecMovable, **kw):
    return Rig(target_device_idx, xp, n=np.shape(frame)[0], cls=cls, **kw).step(frame, window)


class TestShSlopecMovableExtra(unittest.TestCase):

    @cpu_and_gpu
    def test_matches_independent_reference(self, target_device_idx, xp):
        """Fractional windows, both signs and axes, against a plain-numpy WCoG."""
        for wx, wy, sx, sy in [(2.5, -1.25, 3.0, -0.5), (-6.0, 4.5, -5.0, 5.5), (0.75, 0.0, 1.0, 1.0)]:
            frame = gaussian_spot(NP_SUB, sx, sy)
            gx, gy = fresh(target_device_idx, xp, frame, [wx, wy, 0.0])
            rx, ry = reference_slopes(frame, wx, wy, RADIUS, NP_SUB)
            np.testing.assert_allclose([gx[0], gy[0]], [rx, ry], rtol=1e-5, atol=1e-8)

    @cpu_and_gpu
    def test_fractional_window_on_spot_gives_zero(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 2.5, -1.25)
        sx, sy = fresh(target_device_idx, xp, frame, [2.5, -1.25, 0.0])
        self.assertAlmostEqual(float(sx[0]), 0.0, places=6)
        self.assertAlmostEqual(float(sy[0]), 0.0, places=6)

    @cpu_and_gpu
    def test_odd_np_sub_matches_rolled_frame(self, target_device_idx, xp):
        n, wx, wy = 31, 4, -3
        frame = gaussian_spot(n, wx + 0.6, wy + 0.3)
        gx, gy = fresh(target_device_idx, xp, frame, [float(wx), float(wy), 0.0])
        rolled = np.roll(frame, shift=(-wy, -wx), axis=(0, 1))
        rx, ry = fresh(target_device_idx, xp, rolled, None, cls=ShSlopec)
        np.testing.assert_allclose([gx, gy], [rx, ry], rtol=1e-5, atol=1e-8)

    @cpu_and_gpu
    def test_window_applies_to_all_subapertures(self, target_device_idx, xp):
        """Two subapertures side by side: same window and same slope zero on both."""
        n = NP_SUB
        idx0 = np.ravel_multi_index(np.where(np.ones((n, n)) == 1), (n, 2 * n))
        idxs = np.stack([idx0, idx0 + n])
        subap = SubapData(idxs=idxs, display_map=np.arange(2), nx=2, ny=1, target_device_idx=target_device_idx)
        slopec = ShSlopecMovable(subap, weightedPixRad=RADIUS, thr_value=0, target_device_idx=target_device_idx)
        wx, wy = 5.0, -4.0
        frame = np.hstack([gaussian_spot(n, wx, wy), gaussian_spot(n, wx + 1.0, wy)])
        pixels = Pixels(n, 2 * n, target_device_idx=target_device_idx)
        pixels.pixels = xp.array(frame)
        pixels.generation_time = 1
        win = BaseValue(value=xp.array([wx, wy, 0.0]), target_device_idx=target_device_idx)
        win.generation_time = 1
        slopec.inputs['in_pixels'].set(pixels)
        slopec.inputs['in_window'].set(win)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        sx = cpuArray(slopec.outputs['out_slopes'].xslopes)
        sy = cpuArray(slopec.outputs['out_slopes'].yslopes)
        self.assertEqual(sx.shape, (2,))
        self.assertAlmostEqual(float(sx[0]), 0.0, places=6)
        self.assertAlmostEqual(float(sy[0]), 0.0, places=6)
        self.assertGreater(float(sx[1]), 0.0)              # second spot 1 px to the right of the window
        self.assertAlmostEqual(float(sy[1]), 0.0, places=6)

    # ---- state handling across steps ----

    @cpu_and_gpu
    def test_consecutive_moves_have_no_stale_state(self, target_device_idx, xp):
        """A window sequence on one object equals a fresh object at every step (mask fully recomputed)."""
        frame = gaussian_spot(NP_SUB, 3.0, -2.0)
        rig = Rig(target_device_idx, xp)
        for win in [(6.0, 1.0), (-4.0, 5.0), (6.0, 1.0), (0.0, 0.0), (2.0, -3.0), (0.0, 0.0)]:
            got = rig.step(frame, [win[0], win[1], 0.0])
            ref = fresh(target_device_idx, xp, frame, [win[0], win[1], 0.0])
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9, err_msg=str(win))

    @cpu_and_gpu
    def test_return_to_zero_window_restores_shslopec(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, -1.0, 2.0)
        rig = Rig(target_device_idx, xp)
        rig.step(frame, [7.0, -6.0, 0.0])
        got = rig.step(frame, [0.0, 0.0, 0.0])
        ref = fresh(target_device_idx, xp, frame, None, cls=ShSlopec)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    @cpu_and_gpu
    def test_repeated_identical_step_is_idempotent(self, target_device_idx, xp):
        """The window offset is subtracted once per step, never accumulated."""
        frame = gaussian_spot(NP_SUB, 5.0, 4.0)
        rig = Rig(target_device_idx, xp)
        first = rig.step(frame, [4.0, 3.0, 0.0])
        for _ in range(3):
            np.testing.assert_array_equal(rig.step(frame, [4.0, 3.0, 0.0]), first)

    @cpu_and_gpu
    def test_disconnected_window_after_connection_reverts_to_shslopec(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 1.0, 1.0)
        rig = Rig(target_device_idx, xp)
        rig.step(frame, [6.0, -5.0, 1.0])
        rig.slopec.inputs['in_window'].input_values = []       # link removed
        got = rig.step(frame, None)
        ref = fresh(target_device_idx, xp, frame, None, cls=ShSlopec)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
        self.assertEqual(rig.slopec._w_applied, (0.0, 0.0))
        self.assertFalse(rig.slopec._hold)

    def test_two_instances_do_not_share_window_state(self):
        frame = gaussian_spot(NP_SUB, 3.0, 3.0)
        a, b = Rig(-1, np), Rig(-1, np)
        ra = a.step(frame, [5.0, 5.0, 0.0])
        b.step(frame, [-5.0, -5.0, 0.0])
        np.testing.assert_array_equal(a.step(frame, [5.0, 5.0, 0.0]), ra)
        self.assertEqual(a.slopec._w_applied, (5.0, 5.0))
        self.assertEqual(b.slopec._w_applied, (-5.0, -5.0))

    # ---- hold flag ----

    @cpu_and_gpu
    def test_hold_flag_nonzero(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 3.0, 2.0)
        rig = Rig(target_device_idx, xp)
        sx, _ = rig.step(frame, [1.0, 1.0, 0.0])
        self.assertNotEqual(float(sx[0]), 0.0)
        self.assertFalse(rig.slopec._hold)
        sx, sy = rig.step(frame, [1.0, 1.0, 1.0])
        self.assertEqual((float(sx[0]), float(sy[0])), (0.0, 0.0))
        self.assertTrue(rig.slopec._hold)

    @cpu_and_gpu
    def test_hold_release_gives_slopes_again(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 3.0, 2.0)
        rig = Rig(target_device_idx, xp)
        before = rig.step(frame, [1.0, 1.0, 0.0])
        rig.step(frame, [1.0, 1.0, 1.0])
        after = rig.step(frame, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(after, before)

    @cpu_and_gpu
    def test_two_element_window_means_no_hold(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 3.0, 2.0)
        got = fresh(target_device_idx, xp, frame, [1.0, 1.0])
        ref = fresh(target_device_idx, xp, frame, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(got, ref)

    @cpu_and_gpu
    def test_hold_moves_the_window(self, target_device_idx, xp):
        """Under hold only the slopes are zeroed: the mask is already on the new window, so the first frame
        after the release gives the slopes of that window."""
        frame = gaussian_spot(NP_SUB, 7.0, 0.5)
        rig = Rig(target_device_idx, xp)
        sx, sy = rig.step(frame, [6.0, 0.0, 1.0])
        self.assertEqual((float(sx[0]), float(sy[0])), (0.0, 0.0))
        self.assertEqual(rig.slopec._w_applied, (6.0, 0.0))
        got = rig.step(frame, [6.0, 0.0, 0.0])
        ref = fresh(target_device_idx, xp, frame, [6.0, 0.0, 0.0])
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    @cpu_and_gpu
    def test_hold_on_dark_frame_gives_finite_zeros(self, target_device_idx, xp):
        """During a dropout the raw WCoG is NaN on a dark frame; hold must still output clean zeros."""
        dark = np.zeros((NP_SUB, NP_SUB))
        with np.errstate(all='ignore'):
            sx, sy = fresh(target_device_idx, xp, dark, [3.0, 2.0, 1.0])
        self.assertEqual((float(sx[0]), float(sy[0])), (0.0, 0.0))

    @cpu_and_gpu
    def test_hold_gives_zero_slopes_with_slope_null(self, target_device_idx, xp):
        """BUG (sh_slopec_movable.py:84-86): the slope null ``sn`` is subtracted by Slopec.post_trigger
        after the hold zeroing, so hold outputs ``-sn`` instead of 0 and the loop keeps integrating."""
        sn = Slopes(2, target_device_idx=target_device_idx)
        sn.xslopes = xp.array([0.1])
        sn.yslopes = xp.array([0.2])
        frame = gaussian_spot(NP_SUB, 3.0, 2.0)
        sx, sy = fresh(target_device_idx, xp, frame, [1.0, 1.0, 1.0], sn=sn)
        self.assertEqual((float(sx[0]), float(sy[0])), (0.0, 0.0))

    @cpu_and_gpu
    def test_slope_null_is_subtracted_once_on_top_of_window(self, target_device_idx, xp):
        sn = Slopes(2, target_device_idx=target_device_idx)
        sn.xslopes = xp.array([0.1])
        sn.yslopes = xp.array([0.2])
        frame = gaussian_spot(NP_SUB, 5.0, 4.0)
        base = fresh(target_device_idx, xp, frame, [4.0, 3.0, 0.0])
        got = fresh(target_device_idx, xp, frame, [4.0, 3.0, 0.0], sn=sn)
        np.testing.assert_allclose(got[0], base[0] - 0.1, atol=1e-6)
        np.testing.assert_allclose(got[1], base[1] - 0.2, atol=1e-6)

    # ---- vecWeiPixRadT interplay ----

    @cpu_and_gpu
    def test_vec_wei_pix_rad_t_keeps_the_shifted_window(self, target_device_idx, xp):
        """After the radius changes in time the mask is rebuilt on the same window centre, not reset to 0."""
        vec = xp.asarray([[1.0, 0.2], [3.0, 1.0]])
        frame = gaussian_spot(NP_SUB, 5.0, -4.0)
        rig = Rig(target_device_idx, xp, radius=2.0, vecWeiPixRadT=vec)
        win = [5.0, -4.0, 0.0]
        early = rig.step(frame, win, t=int(0.5e9))             # 0.5 s: row at 0.2 s valid -> radius 1.0
        self.assertAlmostEqual(float(rig.slopec.weighted_pix_rad), 1.0)
        late = rig.step(frame, win, t=int(2e9))                # 2 s: radius 3.0
        self.assertAlmostEqual(float(rig.slopec.weighted_pix_rad), 3.0)
        for got, radius in [(early, 1.0), (late, 3.0)]:
            ref = fresh(target_device_idx, xp, frame, win, radius=radius)
            np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
            rx, ry = reference_slopes(frame, 5.0, -4.0, radius, NP_SUB)
            np.testing.assert_allclose([got[0][0], got[1][0]], [rx, ry], rtol=1e-5, atol=1e-8)

    @cpu_and_gpu
    def test_vec_wei_pix_rad_t_then_window_move(self, target_device_idx, xp):
        """Window changes after the radius switch use the new radius."""
        vec = xp.asarray([[3.0, 0.1]])
        frame = gaussian_spot(NP_SUB, -3.0, 4.0)
        rig = Rig(target_device_idx, xp, radius=2.0, vecWeiPixRadT=vec)
        rig.step(frame, [0.0, 0.0, 0.0], t=int(1e9))
        got = rig.step(frame, [-3.0, 4.0, 0.0], t=int(2e9))
        ref = fresh(target_device_idx, xp, frame, [-3.0, 4.0, 0.0], radius=3.0)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    def test_vec_wei_pix_rad_t_zero_radius_with_moved_window(self):
        """BUG (sh_slopec_movable.py:55-58): a radius of 0 from vecWeiPixRadT (accepted by ShSlopec
        as 'no weighting') makes sigma = 0 in _shift_weights -> NaN mask and NaN slopes. The
        constructor only rejects weightedPixRad == 0, not a later 0 from the time table."""
        vec = np.asarray([[2.0, 0.0], [0.0, 1.0]])
        frame = gaussian_spot(NP_SUB, 5.0, 3.0)
        rig = Rig(-1, np, radius=2.0, vecWeiPixRadT=vec)
        rig.step(frame, [5.0, 3.0, 0.0], t=int(0.5e9))
        with np.errstate(all='ignore'):
            sx, sy = rig.step(frame, [5.0, 3.0, 0.0], t=int(2e9))
        self.assertTrue(np.isfinite(sx).all() and np.isfinite(sy).all())

    # ---- precision and dtype ----

    def test_single_precision_matches_double(self):
        frame = gaussian_spot(NP_SUB, 4.0, -3.0)
        win = [3.0, -2.0, 0.0]
        d = Rig(-1, np, precision=0)
        s = Rig(-1, np, precision=1)
        gd = d.step(frame, win)
        gs = s.step(frame.astype(np.float32), win)
        self.assertEqual(s.slopec.dtype, np.float32)
        self.assertEqual(s.slopec.mask_weighted.dtype, np.float32)
        self.assertEqual(s.slopec.xweights.dtype, np.float32)
        self.assertEqual(s.slopec._w.dtype, np.float32)
        np.testing.assert_allclose(gs[0], gd[0], atol=1e-5)
        np.testing.assert_allclose(gs[1], gd[1], atol=1e-5)
        # slope dtype must not be promoted relative to the parent class
        ref = Rig(-1, np, cls=ShSlopec, precision=1).step(frame.astype(np.float32))
        self.assertEqual(gs[0].dtype, ref[0].dtype)

    def test_shifted_weights_are_consistent_with_mask(self):
        rig = Rig(-1, np)
        rig.step(gaussian_spot(NP_SUB, 0, 0), [3.0, -2.0, 0.0])
        s = rig.slopec
        n = NP_SUB
        self.assertEqual(s.mask_weighted.shape, (n, n))
        # the combined array used by ShSlopec for the slopes holds the shifted weights
        np.testing.assert_array_equal(s._weights[0], s.mask_weighted.ravel())
        np.testing.assert_array_equal(s._weights[1], s.xweights.ravel())
        self.assertAlmostEqual(float(s.mask_weighted.max()), 1.0, places=6)
        # mask peak on the window: row = y = -2 + c, column = x = 3 + c
        row, col = np.unravel_index(np.argmax(s.mask_weighted), (n, n))
        c = (n - 1) / 2.0
        self.assertLess(abs(col - c - 3.0), 0.51)
        self.assertLess(abs(row - c + 2.0), 0.51)
        np.testing.assert_array_equal(s.xweights == 0, s.mask_weighted == 0)


if __name__ == '__main__':
    unittest.main()
