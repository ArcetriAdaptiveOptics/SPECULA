import unittest

import numpy as np

from iteras.lib.frame_stacker import FrameStacker


def frames(first, n, channels=2, side=3):
    """n frames, each filled with its own time index (first, first+1, ...)."""
    t = np.arange(first, first + n, dtype=np.float32)
    return np.broadcast_to(t[:, None, None, None], (n, channels, side, side)).copy()


def frame_times(stacked, channels=2):
    """Time index of each stacked frame, per sample: (B, n_frames)."""
    return stacked[:, ::channels, 0, 0]


class TestFrameStacker(unittest.TestCase):

    def test_stacks_current_then_previous_frames_repeating_the_first_at_start(self):
        out = FrameStacker(4, np)(frames(0, 5))
        self.assertEqual(out.shape, (5, 8, 3, 3))
        np.testing.assert_array_equal(frame_times(out), [[0, 0, 0, 0],
                                                         [1, 0, 0, 0],
                                                         [2, 1, 0, 0],
                                                         [3, 2, 1, 0],
                                                         [4, 3, 2, 1]])

    def test_history_carries_over_to_the_next_batch(self):
        stacker = FrameStacker(4, np)
        stacker(frames(0, 5))
        np.testing.assert_array_equal(frame_times(stacker(frames(5, 2))), [[5, 4, 3, 2],
                                                                          [6, 5, 4, 3]])

    def test_one_frame_at_a_time_matches_whole_batch(self):
        batch = FrameStacker(3, np)(frames(0, 6))
        one_by_one = FrameStacker(3, np)
        singles = np.concatenate([one_by_one(frames(i, 1)) for i in range(6)])
        np.testing.assert_array_equal(singles, batch)

    def test_single_channel_frames_without_channel_axis(self):
        x = frames(0, 4, channels=1)[:, 0]          # (B, H, W)
        out = FrameStacker(2, np)(x)
        self.assertEqual(out.shape, (4, 2, 3, 3))
        np.testing.assert_array_equal(frame_times(out, channels=1), [[0, 0], [1, 0], [2, 1], [3, 2]])

    def test_n_frames_one_is_a_no_op(self):
        x = frames(0, 4)
        self.assertIs(FrameStacker(1, np)(x), x)

    def test_invalid_n_frames(self):
        with self.assertRaises(ValueError):
            FrameStacker(0, np)


if __name__ == '__main__':
    unittest.main()
