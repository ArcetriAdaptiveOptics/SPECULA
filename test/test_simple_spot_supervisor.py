import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.processing_objects.simple_spot_supervisor import SimpleSpotSupervisor
from test.specula_testlib import cpu_and_gpu

N = 64
NOISE = 1.0


def spot(sx, sy, amp=20.0, sigma=1.0):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    kw.setdefault('look_frames', 4)
    kw.setdefault('snr_thr', 6.0)
    kw.setdefault('cmd_latency', 2)
    return SimpleSpotSupervisor(np_sub=N, cmd_to_px=[[1, 0], [0, 1]], target_device_idx=target_device_idx, **kw)


class TestSimpleSpotSupervisor(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N)) + 5.0      # with a background pedestal

    def test_lost_spot_is_brought_back(self):
        sup = make()
        moved = [sup.process_frame(spot(15.0, -10.0) + self.noise()) for _ in range(4)]
        self.assertEqual(moved, [False, False, False, True])
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.ff_px, [15.0, -10.0], atol=0.6)  # integer-pixel peak, centre at .5

    def test_spot_in_window_is_left_alone(self):
        sup = make()
        for _ in range(40):
            sup.process_frame(spot(1.0, -1.5) + self.noise())
        self.assertEqual(sup.n_moves, 0)
        self.assertGreater(sup.state[2], 6.0)                          # detected, but inside the capture radius

    def test_pure_noise_never_moves(self):
        sup = make(look_frames=1)
        for _ in range(500):
            sup.process_frame(self.noise())
        self.assertEqual(sup.n_moves, 0)
        self.assertLess(sup.state[2], 6.0)

    def test_faint_spot_below_threshold_does_not_move(self):
        sup = make()
        for _ in range(40):
            sup.process_frame(spot(15.0, -10.0, amp=0.5) + self.noise())
        self.assertEqual(sup.n_moves, 0)

    def test_window_sees_the_spot_vetoes_the_move(self):
        """A significant peak near the reference vetoes a move to a brighter peak elsewhere."""
        sup = make()
        for _ in range(4):
            sup.process_frame(spot(0.5, 0.5) + spot(15.0, -10.0, amp=40.0) + self.noise())
        self.assertEqual(sup.n_moves, 0)
        np.testing.assert_allclose(sup.state[:2], [15.0, -10.0], atol=0.6)

    def test_frames_after_a_move_are_discarded(self):
        """The cmd_latency frames after a move still show the spot at the old position: they must not enter the
        next look (no second move)."""
        sup = make(look_frames=2, cmd_latency=3)
        frames = [spot(15.0, -10.0) + self.noise() for _ in range(2 + 3)] + [spot(0.0, 0.0) + self.noise()
                                                                            for _ in range(4)]
        moved = [sup.process_frame(f) for f in frames]
        self.assertEqual(sum(moved), 1)
        self.assertTrue(moved[1])

    @cpu_and_gpu
    def test_trigger_outputs(self, target_device_idx, xp):
        cmd_to_px = [[0.5, 0.0], [0.0, -0.25]]
        sup = SimpleSpotSupervisor(look_frames=2, snr_thr=6.0, cmd_latency=1, np_sub=N, cmd_to_px=cmd_to_px,
                                   target_device_idx=target_device_idx)
        pix = Pixels(N, N, target_device_idx=target_device_idx)
        sup.inputs['in_pixels'].set(pix)
        sup.setup()
        for t in (1, 2):
            pix.pixels = xp.asarray(spot(15.0, -10.0) + self.noise(), dtype=pix.pixels.dtype)
            pix.generation_time = t
            sup.check_ready(t)
            sup.trigger()
            sup.post_trigger()
        ff = cpuArray(sup.outputs['out_feedforward'].value)
        np.testing.assert_allclose(ff, np.linalg.inv(cmd_to_px) @ sup.ff_px, rtol=1e-5)
        self.assertEqual(sup.outputs['out_feedforward'].generation_time, 2)
        st = cpuArray(sup.outputs['out_state'].value)
        self.assertEqual(st[4], 1)

    def test_parameters_and_names(self):
        with self.assertRaises(ValueError):
            make(look_frames=0)
        with self.assertRaises(ValueError):
            make(cmd_latency=-1)
        self.assertEqual(set(SimpleSpotSupervisor.input_names()), {'in_pixels'})
        self.assertEqual(set(SimpleSpotSupervisor.output_names()), {'out_feedforward', 'out_state'})
