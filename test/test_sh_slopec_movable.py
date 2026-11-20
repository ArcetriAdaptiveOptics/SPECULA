import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.sh_slopec_movable import ShSlopecMovable
from test.specula_testlib import cpu_and_gpu

NP_SUB = 32
RADIUS = 2.0


def gaussian_spot(n, sx, sy, sigma=1.5, amp=100.0):
    """Gaussian spot centred at (sx, sy) px from the subaperture centre (x = column, y = row)."""
    c = (n - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(n) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(n) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


class TestShSlopecMovable(unittest.TestCase):

    def single_subap(self, target_device_idx, n=NP_SUB):
        idx = np.where(np.ones((n, n)) == 1)
        v = np.zeros((1, n * n), dtype=int)
        v[0] = np.ravel_multi_index(idx, (n, n))
        return SubapData(idxs=v, display_map=np.zeros(1, dtype=int), nx=1, ny=1,
                         target_device_idx=target_device_idx)

    def run_slopec(self, cls, frame, window, target_device_idx, xp, t=1):
        slopec = cls(self.single_subap(target_device_idx), weightedPixRad=RADIUS, thr_value=0,
                     target_device_idx=target_device_idx)
        pixels = Pixels(NP_SUB, NP_SUB, target_device_idx=target_device_idx)
        pixels.pixels = xp.array(frame)
        pixels.generation_time = t
        slopec.inputs['in_pixels'].set(pixels)
        if window is not None:
            win = BaseValue(value=xp.array(window, dtype=float), target_device_idx=target_device_idx)
            win.generation_time = t
            slopec.inputs['in_window'].set(win)
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()
        return (float(cpuArray(slopec.outputs['out_slopes'].xslopes)[0]),
                float(cpuArray(slopec.outputs['out_slopes'].yslopes)[0]))

    @cpu_and_gpu
    def test_unconnected_matches_shslopec(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 1.3, -0.7)
        ref = self.run_slopec(ShSlopec, frame, None, target_device_idx, xp)
        got = self.run_slopec(ShSlopecMovable, frame, None, target_device_idx, xp)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    @cpu_and_gpu
    def test_zero_window_matches_shslopec(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, -2.1, 0.9)
        ref = self.run_slopec(ShSlopec, frame, None, target_device_idx, xp)
        got = self.run_slopec(ShSlopecMovable, frame, [0.0, 0.0, 0.0], target_device_idx, xp)
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)

    @cpu_and_gpu
    def test_spot_on_window_gives_zero_slope(self, target_device_idx, xp):
        """A spot exactly at the window centre reads ~0 whatever the window position (x = column, y = row)."""
        for wx, wy in [(6.0, 0.0), (0.0, -5.0), (-4.0, 7.0)]:
            frame = gaussian_spot(NP_SUB, wx, wy)
            sx, sy = self.run_slopec(ShSlopecMovable, frame, [wx, wy, 0.0], target_device_idx, xp)
            self.assertAlmostEqual(sx, 0.0, places=6)
            self.assertAlmostEqual(sy, 0.0, places=6)

    @cpu_and_gpu
    def test_equivalent_to_shifted_frame(self, target_device_idx, xp):
        """Window at integer (wx, wy) on a frame == centred window on the frame rolled by (-wy, -wx)."""
        wx, wy = 5, -3
        frame = gaussian_spot(NP_SUB, wx + 0.8, wy - 0.4)
        got = self.run_slopec(ShSlopecMovable, frame, [float(wx), float(wy), 0.0], target_device_idx, xp)
        rolled = np.roll(frame, shift=(-wy, -wx), axis=(0, 1))
        ref = self.run_slopec(ShSlopec, rolled, None, target_device_idx, xp)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-8)

    @cpu_and_gpu
    def test_slope_is_relative_and_sign_follows_offset(self, target_device_idx, xp):
        wx, wy = 4.0, 3.0
        frame = gaussian_spot(NP_SUB, wx + 1.0, wy - 1.0)
        sx, sy = self.run_slopec(ShSlopecMovable, frame, [wx, wy, 0.0], target_device_idx, xp)
        self.assertGreater(sx, 0.0)      # spot at larger column than the window centre
        self.assertLess(sy, 0.0)         # spot at smaller row
        # calibrated gain: |slope| below the geometric offset, in units of 2/np_sub per px
        self.assertLess(abs(sx), 1.0 * 2.0 / NP_SUB)

    @cpu_and_gpu
    def test_hold_zeroes_the_slopes(self, target_device_idx, xp):
        frame = gaussian_spot(NP_SUB, 1.0, 1.0)
        sx, sy = self.run_slopec(ShSlopecMovable, frame, [3.0, -2.0, 1.0], target_device_idx, xp)
        self.assertEqual(sx, 0.0)
        self.assertEqual(sy, 0.0)

    def test_unsupported_modes_raise(self):
        subap = self.single_subap(-1)
        with self.assertRaises(ValueError):
            ShSlopecMovable(subap, weightedPixRad=RADIUS, windowing=True, target_device_idx=-1)
        with self.assertRaises(ValueError):
            ShSlopecMovable(subap, weightedPixRad=0.0, target_device_idx=-1)
        with self.assertRaises(ValueError):
            ShSlopecMovable(subap, weightedPixRad=RADIUS, exp_weight=2.0, target_device_idx=-1)

    def test_window_input_registered(self):
        self.assertIn('in_window', ShSlopecMovable.input_names())
        self.assertIn('in_pixels', ShSlopecMovable.input_names())
        slopec = ShSlopecMovable(self.single_subap(-1), weightedPixRad=RADIUS, target_device_idx=-1)
        self.assertTrue(slopec.inputs['in_window'].optional)


if __name__ == '__main__':
    unittest.main()
