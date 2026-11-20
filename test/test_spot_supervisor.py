import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.processing_objects.spot_supervisor import SpotSupervisor
from test.specula_testlib import cpu_and_gpu

N = 64
PEDESTAL = 50.0
NOISE = 2.5
AMP = 40.0          # SNR of a look >> 7.5 for a sigma = 1 px spot


def spot(sx, sy, amp=AMP, sigma=1.0):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    kw.setdefault('cmd_latency', 3)
    kw.setdefault('look_frames', 4)
    kw.setdefault('capture_radius', 4.0)
    kw.setdefault('k_absent', 2)
    kw.setdefault('cmd_to_px', [[1, 0], [0, 1]])
    return SpotSupervisor(np_sub=N, target_device_idx=target_device_idx, **kw)


class TestSpotSupervisor(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def noise(self):
        return PEDESTAL + NOISE * self.rng.standard_normal((N, N))

    @staticmethod
    def u_ff_on_mirror(sup):
        """Feedforward reaching the pixels on the next frame j = sup.frame: the one in force after frame j - L
        (zero before). Keeps a history of sup.u_ff after each frame in sup._test_ffh (index = frame)."""
        ffh = sup.__dict__.setdefault('_test_ffh', [])
        j = sup.frame
        return ffh[j - sup.cmd_latency] if j >= sup.cmd_latency else np.zeros(2)

    @staticmethod
    def record_ff(sup):
        sup._test_ffh.append(sup.u_ff.copy())

    def step(self, sup, f):
        """process_frame + record of the feedforward history."""
        self.u_ff_on_mirror(sup)                                      # make sure the history exists
        sup.process_frame(f)
        self.record_ff(sup)

    def feed(self, sup, n, *spots):
        """Process n frames with the given (x, y, amp) spots shifted by the feedforward as it reaches the pixels
        (cmd_latency frames after it was issued); no loop command."""
        for _ in range(n):
            f = self.noise()
            uff = self.u_ff_on_mirror(sup)
            for x, y, amp in spots:
                f = f + spot(x - uff[0], y - uff[1], amp)
            self.step(sup, f)

    # 1
    def test_lost_spot_move_confirm_and_return(self):
        """A bright far spot moves the window onto it at the first look; the next look confirms (u_ff = window
        offset) and the window returns to zero cmd_latency - 1 frames after the confirmation."""
        sup = make()
        ws = []
        for _ in range(12):
            self.feed(sup, 1, (15.0, -10.0, AMP))
            ws.append(sup.w.copy())
            if len(ws) == 3:
                np.testing.assert_array_equal(sup.w, [0, 0])           # no move before the look completes
        np.testing.assert_allclose(ws[3], [15.0, -10.0], atol=0.6)     # integer-pixel peak
        np.testing.assert_array_equal(ws[2], [0, 0])
        moved = ws[3]
        for w in ws[4:9]:
            np.testing.assert_array_equal(w, moved)                    # held until the return
        np.testing.assert_array_equal(ws[9], [0, 0])                   # confirm at frame 7, return at 7 + 3 - 1
        np.testing.assert_allclose(sup.u_ff, moved)
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.n_aborts, 0)

    def test_no_feedforward_before_confirmation(self):
        """u_ff stays zero until the look after the move completes."""
        sup = make()
        self.feed(sup, 4, (15.0, -10.0, AMP))
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_array_equal(sup.u_ff, [0, 0])
        self.feed(sup, 3, (15.0, -10.0, AMP))
        np.testing.assert_array_equal(sup.u_ff, [0, 0])
        self.feed(sup, 1, (15.0, -10.0, AMP))
        self.assertGreater(np.linalg.norm(sup.u_ff), 10.0)

    # 2
    def test_spot_inside_capture_radius_never_moves(self):
        """A spot within the capture radius is left to the WCoG: no move, no dropout."""
        sup = make()
        self.feed(sup, 40, (1.0, -1.5, AMP))
        self.assertEqual(sup.n_moves, 0)
        self.assertFalse(sup.dropout)

    # 3
    def test_pure_noise_dropout_at_second_absent_look_without_move(self):
        """Noise only (empty window): hold at the k_absent-th empty look, never a move."""
        sup = make()
        self.feed(sup, 4)
        self.assertFalse(sup.dropout)
        self.feed(sup, 4)
        self.assertTrue(sup.dropout)
        self.feed(sup, 40)
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.n_moves, 0)

    # 4
    def _faint_spot_setup(self, pos):
        """Same noise at every look (constant s) plus a spot at pos with an amplitude tuned so that s is mid-band
        between presence_thr and snr_thr. Returns (frames, amp)."""
        frames = [self.noise() for _ in range(4)]

        def s_of(amp):
            sup = make()
            for f in frames:
                sup.process_frame(f + spot(pos[0], pos[1], amp))
            return sup.state[2]

        amp = 10.0
        for _ in range(3):                                            # s is ~linear in amp: refine to the mid-band
            amp *= 0.5 * (6.0 + 7.5) / s_of(amp)
        sup = make()
        self.assertGreater(s_of(amp), sup.presence_thr)
        self.assertLess(s_of(amp), sup.snr_thr)
        return frames, amp

    def test_faint_spot_outside_window_holds_without_move(self):
        """presence_thr < s < snr_thr, spot outside the window: the window is empty, so the hold starts after
        k_absent looks, but the spot is too faint to move the window."""
        pos = (15.0, -10.0)
        frames, amp = self._faint_spot_setup(pos)
        sup = make()
        for _ in range(10):
            for f in frames:
                sup.process_frame(f + spot(pos[0], pos[1], amp))
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.n_moves, 0)

    def test_faint_spot_inside_window_no_hold_no_move(self):
        """presence_thr < s < snr_thr, spot inside the window: the window is occupied (no hold) and the spot is
        within the capture radius (no move)."""
        pos = (1.0, 1.0)
        frames, amp = self._faint_spot_setup(pos)
        sup = make()
        for _ in range(10):
            for f in frames:
                sup.process_frame(f + spot(pos[0], pos[1], amp))
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.miss, 0)
        self.assertEqual(sup.n_moves, 0)

    def test_step_moves_at_next_look_without_hold(self):
        """Spot in the window, then a step to a far position: the first complete look after the step has one empty
        look only (miss = 1 < k_absent = 2), so no hold, and it moves the window; the next look confirms; the window
        returns when the feedforward reaches the pixels. No hold at any frame."""
        sup = make()
        far = (15.0, -10.0, AMP)
        dropouts = []

        def run(n, *spots):
            for _ in range(n):
                self.feed(sup, 1, *spots)
                dropouts.append(sup.dropout)

        run(4, (0.0, 0.0, AMP))                                       # look 1 (frames 0-3): spot in the window
        self.assertEqual(sup.miss, 0)
        run(3, far)
        np.testing.assert_array_equal(sup.w, [0, 0])                  # look 2 not complete yet
        run(1, far)                                                   # frame 7: look 2 holds the step only
        self.assertEqual(sup.miss, 1)                                 # empty window, but k_absent = 2 not reached
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.n_moves, 1)                              # moved without waiting for the hold
        np.testing.assert_allclose(sup.w, [15.0, -10.0], atol=0.6)
        w_moved = sup.w.copy()
        np.testing.assert_array_equal(sup.u_ff, [0, 0])
        run(3, far)                                                   # frames 8-10
        np.testing.assert_array_equal(sup.w, w_moved)
        run(1, far)                                                   # frame 11: confirmation look
        self.assertEqual(sup.miss, 0)
        np.testing.assert_allclose(sup.u_ff, w_moved)
        np.testing.assert_array_equal(sup.w, w_moved)
        run(2, far)                                                   # frames 12, 13: return after frame 13
        np.testing.assert_array_equal(sup.w, [0, 0])
        run(20, far)                                                  # spot back in the window thanks to the ff
        self.assertFalse(any(dropouts))
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.n_aborts, 0)
        np.testing.assert_array_equal(sup.w, [0, 0])
        self.assertEqual(sup.miss, 0)

    def test_move_during_hold_and_confirmation_ends_it(self):
        """Noise only until the hold starts, then a bright far spot: the window moves while held; the confirmation
        look (spot at the new window) ends the hold and issues the feedforward."""
        sup = make()
        far = (15.0, -10.0, AMP)
        self.feed(sup, 8)
        self.assertTrue(sup.dropout)
        self.feed(sup, 4, far)                                        # frames 8-11: moving look
        self.assertEqual(sup.n_moves, 1)
        self.assertTrue(sup.dropout)                                  # still held: the window was empty
        np.testing.assert_allclose(sup.w, [15.0, -10.0], atol=0.6)
        np.testing.assert_array_equal(sup.u_ff, [0, 0])
        self.feed(sup, 4, far)                                        # frames 12-15: confirmation look
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.miss, 0)
        self.assertEqual(sup.n_aborts, 0)
        np.testing.assert_allclose(sup.u_ff, [15.0, -10.0], atol=0.6)

    # 5
    def test_local_veto_comparable_near_spot(self):
        """A near peak at 0.83 of the far one (> q_thr) vetoes the move."""
        sup = make()
        self.feed(sup, 4, (0.0, 0.0, 250.0), (15.0, -10.0, 300.0))
        self.assertEqual(sup.n_moves, 0)
        self.assertGreater(sup.state[3] / sup.state[2], sup.q_thr)

    def test_local_veto_weak_near_spot(self):
        """A near peak at 0.2 of the far one (< q_thr) does not veto."""
        sup = make()
        self.feed(sup, 4, (0.0, 0.0, 60.0), (15.0, -10.0, 300.0))
        self.assertEqual(sup.n_moves, 1)
        self.assertLess(sup.state[3] / sup.state[2], sup.q_thr)

    def test_no_near_spot_moves(self):
        """No near spot: the far one is taken."""
        sup = make()
        self.feed(sup, 4, (15.0, -10.0, 300.0))
        self.assertEqual(sup.n_moves, 1)

    # 6
    def test_spot_just_outside_capture_radius_moves(self):
        """Its own wing inside the disk is not a 3x3 maximum: a spot at capture_radius + 1 is recovered."""
        sup = make(capture_radius=4.0)
        self.feed(sup, 4, (5.0, 0.0, 300.0))
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.w, [5.0, 0.0], atol=0.6)

    # 7
    def test_confirmation_abort_restores_window(self):
        """If the spot is gone at the new window the move is aborted: window back, no feedforward."""
        sup = make(k_absent=3)
        self.feed(sup, 4, (15.0, -10.0, AMP))
        self.assertEqual(sup.n_moves, 1)
        self.feed(sup, 4)
        np.testing.assert_array_equal(sup.w, [0, 0])
        self.assertEqual(sup.n_aborts, 1)
        np.testing.assert_array_equal(sup.u_ff, [0, 0])
        self.assertFalse(sup.dropout)

    # 8
    def test_dropout_released_by_present_look(self):
        """After a hold, the first look with a spot in the window ends it."""
        sup = make()
        self.feed(sup, 12)
        self.assertTrue(sup.dropout)
        self.feed(sup, 4, (1.0, 1.0, AMP))
        self.assertFalse(sup.dropout)

    # 9
    def test_look_restarts_at_window_changes_and_no_move_during_return(self):
        """look_count is 0 right after a move and after the return; no new move while the return is pending."""
        sup = make(look_frames=2, cmd_latency=6)
        far = (15.0, -10.0, AMP)
        self.feed(sup, 2, far)                                        # move
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.look_count, 0)
        self.feed(sup, 1, far)
        self.assertEqual(sup.look_count, 1)
        self.feed(sup, 1, far)                                        # confirm at frame 3, return at frame 8
        self.assertEqual(len(sup.w_queue), 1)
        self.assertEqual(sup.frame, 4)
        w_moved = sup.w.copy()
        # a second far spot while the return is in flight must not move the window
        for _ in range(3):
            self.feed(sup, 1, (-15.0, 12.0, 300.0))
            if sup.frame < 9:
                np.testing.assert_array_equal(sup.w, w_moved)
                self.assertEqual(sup.n_moves, 1)
        self.feed(sup, 1, (-15.0, 12.0, 300.0))                      # frame 7
        self.assertEqual(sup.n_moves, 1)
        self.feed(sup, 1, (-15.0, 12.0, 300.0))                      # frame 8: return
        np.testing.assert_array_equal(sup.w, [0, 0])
        self.assertEqual(sup.look_count, 0)

    # 10
    @cpu_and_gpu
    def test_trigger_path(self, target_device_idx, xp):
        """Outputs follow the internal state: ff = inv(cmd_to_px) @ u_ff, window, telemetry, out_look = mean frame
        of the last complete look (zero before the first one), no refresh without new pixels."""
        cmd = [[0.5, 0.0], [0.0, -0.25]]
        sup = make(target_device_idx=target_device_idx, cmd_to_px=cmd)
        pix = Pixels(N, N, target_device_idx=target_device_idx)
        sup.inputs['in_pixels'].set(pix)
        sup.setup()

        t = 0
        frames = []
        for t in range(12):
            uff = self.u_ff_on_mirror(sup)
            f = self.noise() + spot(15.0 - uff[0], -10.0 - uff[1])
            frames.append(f)
            pix.pixels = xp.asarray(f, dtype=pix.pixels.dtype)
            pix.generation_time = t
            sup.check_ready(t)
            sup.trigger()
            sup.post_trigger()
            self.record_ff(sup)
            if t == 1:
                np.testing.assert_array_equal(cpuArray(sup.outputs['out_look'].value), 0)
            if t == 3:
                np.testing.assert_allclose(cpuArray(sup.outputs['out_look'].value), np.mean(frames, axis=0),
                                           rtol=1e-5)
                win = cpuArray(sup.outputs['out_window'].value)
                np.testing.assert_allclose(win[:2], [15.0, -10.0], atol=0.6)
                self.assertEqual(win[2], 0)
        ff = cpuArray(sup.outputs['out_feedforward'].value)
        np.testing.assert_allclose(ff, np.linalg.inv(cmd) @ sup.u_ff, rtol=1e-5)
        self.assertGreater(np.linalg.norm(ff), 10.0)
        np.testing.assert_allclose(cpuArray(sup.outputs['out_window'].value), [0, 0, 0], atol=1e-6)
        st = cpuArray(sup.outputs['out_state'].value)
        self.assertEqual(len(st), 7)
        self.assertEqual(st[5], 1)

        # a step in which the pixels were not refreshed leaves the outputs untouched
        gt = {k: v.generation_time for k, v in sup.outputs.items()}
        frame_before = sup.frame
        self.assertFalse(sup.check_ready(t + 1))                      # nothing refreshed: not even triggered
        sup.inputs_changed = True                                     # another input refreshed: step is forced
        sup.trigger()
        sup.post_trigger()
        self.assertEqual(sup.frame, frame_before)
        for k, v in sup.outputs.items():
            self.assertEqual(v.generation_time, gt[k])

    @cpu_and_gpu
    def test_out_hold_follows_window_hold_flag(self, target_device_idx, xp):
        """out_hold (for the controller's in_freeze) equals the hold flag of out_window at every step: 0 while the
        spot is present, 1 after k_absent noise-only looks."""
        sup = make(target_device_idx=target_device_idx)
        pix = Pixels(N, N, target_device_idx=target_device_idx)
        sup.inputs['in_pixels'].set(pix)
        sup.setup()
        holds = []
        for t in range(16):
            f = self.noise() + (spot(0.0, 0.0) if t < 4 else 0)
            pix.pixels = xp.asarray(f, dtype=pix.pixels.dtype)
            pix.generation_time = t
            sup.check_ready(t)
            sup.trigger()
            sup.post_trigger()
            hold = float(cpuArray(sup.outputs['out_hold'].value)[0])
            self.assertEqual(hold, float(cpuArray(sup.outputs['out_window'].value)[2]))
            self.assertEqual(sup.outputs['out_hold'].generation_time, t)
            holds.append(hold)
        self.assertEqual(holds[:11], [0.0] * 11)                     # looks 2, 3 absent: dropout at frame 11
        self.assertEqual(holds[11:], [1.0] * 5)

    # 12
    D = (15.0, -10.0)

    def test_feedforward_does_not_create_spurious_jump(self):
        """Still disturbance: after the confirmed move and the delayed feedforward there is no second move, no
        abort, and the spot stays within capture_radius of the window at every following look."""
        sup = make()
        far = (self.D[0], self.D[1], AMP)
        self.feed(sup, 12, far)
        self.assertEqual(sup.n_moves, 1)
        self.assertGreater(np.linalg.norm(sup.u_ff), 10.0)
        for _ in range(40):
            self.feed(sup, 1, far)
            if sup.look_count == 0 and sup.frame >= 14:
                self.assertLessEqual(np.linalg.norm(np.array(sup.state[:2]) - sup.w), sup.capture_radius)
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.n_aborts, 0)
        self.assertFalse(sup.dropout)

    # 11
    def test_parameter_validation(self):
        """cmd_latency, look_frames, k_absent must be >= 1."""
        for bad in ('cmd_latency', 'look_frames', 'k_absent'):
            with self.assertRaises(ValueError):
                make(**{bad: 0})

    def test_input_output_names(self):
        """Declared interface."""
        self.assertEqual(set(SpotSupervisor.input_names()), {'in_pixels'})
        self.assertEqual(set(SpotSupervisor.output_names()),
                         {'out_window', 'out_hold', 'out_feedforward', 'out_state', 'out_look'})


if __name__ == '__main__':
    unittest.main()
