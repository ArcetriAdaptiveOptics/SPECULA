import json
import os
import tempfile
import unittest

try:
    import torch
    import numpy as np
    from iteras.lib.cnn_checkpoint import (NETWORK_KEYS, build_network, calibration_factor,
                                           check_same_network, load_stats, load_trained_network,
                                           network_input, predict, save_checkpoint, stats_filename)
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
NETWORK = dict(nmodes=NMODES, input_channels=2, n_frames=1, channels=4, depth=2,
               conv_block_type=0, head_type='pooled', head_grid=32, dropout=0.0, channel_product=False)
STATS = {'meanp': 0.0, 'stdp': 1.0, 'meanmodes': [0.0] * NMODES, 'stdmodes': [1.0] * NMODES}


def save(path, **network):
    net = dict(NETWORK, **network)
    model = build_network(net)
    save_checkpoint(model, path, dict(STATS, network=net))
    return model, net


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestCnnCheckpoint(unittest.TestCase):

    def test_stats_filename_is_next_to_the_weights(self):
        self.assertEqual(stats_filename('a/b/net.pth'), 'a/b/net_stats.json')
        self.assertEqual(stats_filename('a/b/net_dp0.000.pth'), 'a/b/net_dp0.000_stats.json')

    def test_the_network_is_rebuilt_from_the_checkpoint_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'sub', 'net.pth')    # missing directories are created
            model, net = save(path, head_type='spatial', head_grid=16, n_frames=4)
            loaded, stats = load_trained_network(path)
            self.assertFalse(loaded.training)
            self.assertEqual(stats['network'], net)
            for k, v in model.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[k], v)

    def test_every_architecture_argument_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save(path)
            self.assertEqual(set(load_stats(path)['network']), set(NETWORK_KEYS))

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            with self.assertRaisesRegex(FileNotFoundError, 'Model file'):
                load_trained_network(path)
            torch.save(build_network(NETWORK).state_dict(), path)
            with self.assertRaisesRegex(FileNotFoundError, 'Statistics file'):
                load_trained_network(path)

    def test_a_checkpoint_without_its_architecture_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            save_checkpoint(build_network(NETWORK), path, STATS)
            with self.assertRaisesRegex(ValueError, 'does not record the network'):
                load_trained_network(path)

    def test_resuming_with_a_different_network_names_every_difference(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            _, net = save(path)
            check_same_network(path, net)
            with self.assertRaises(ValueError) as cm:
                check_same_network(path, dict(net, depth=3, n_frames=4))
            self.assertIn('depth=3 (checkpoint: 2)', str(cm.exception))
            self.assertIn('n_frames=4 (checkpoint: 1)', str(cm.exception))

    def test_older_checkpoints_get_the_defaults_of_newer_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'net.pth')
            old = {k: v for k, v in NETWORK.items() if k != 'channel_product'}
            save_checkpoint(build_network(old), path, dict(STATS, network=old))
            check_same_network(path, NETWORK)
            _, stats = load_trained_network(path)
            self.assertFalse(stats['network']['channel_product'])

    def test_network_input(self):
        x = np.arange(2 * 2 * 3 * 3, dtype=float).reshape(2, 2, 3, 3)
        np.testing.assert_array_equal(network_input(x, NETWORK), x)
        np.testing.assert_array_equal(network_input(x[:, 0], dict(NETWORK, input_channels=1)), x[:, 0])
        np.testing.assert_array_equal(network_input(x, dict(NETWORK, input_channels=1, channel_product=True)),
                                      x[:, 1] * x[:, 0])
        with self.assertRaisesRegex(ValueError, 'input_channels=1'):
            network_input(x, dict(NETWORK, input_channels=1))
        with self.assertRaisesRegex(ValueError, '2 channels'):
            network_input(x[:, 0], dict(NETWORK, input_channels=1, channel_product=True))

    def test_no_checkpoint_yet_is_not_checked(self):
        check_same_network('/nonexistent/net.pth', NETWORK)

    def test_predict_in_chunks_matches_one_pass_and_denormalizes(self):
        model = build_network(NETWORK).eval()
        x = torch.randn(7, 2, 16, 16)
        mean, std = torch.arange(NMODES) * 1.0, torch.full((NMODES,), 2.0)
        with torch.no_grad():
            expected = model(x) * std + mean
        torch.testing.assert_close(predict(model, x, mean, std, chunk=3), expected)
        self.assertFalse(predict(model.train(), x, mean, std).requires_grad)
        self.assertFalse(model.training)      # left in eval mode

    def test_calibration_factor_undoes_the_measured_shrinkage(self):
        stats = dict(STATS, mode_gain=[1.0, 0.5, 0.25, 0.1, -0.3])
        # 1/gain where the network predicts the mode, capped at 3, and left
        # alone where it barely does (gain <= 0.2): amplifying those would
        # mostly amplify noise
        self.assertEqual(list(calibration_factor(stats, NMODES)), [1.0, 2.0, 3.0, 1.0, 1.0])
        self.assertIsNone(calibration_factor(STATS, NMODES))                           # no measurement
        self.assertIsNone(calibration_factor(dict(STATS, mode_gain=[1.0]), NMODES))    # wrong length


if __name__ == '__main__':
    unittest.main()
