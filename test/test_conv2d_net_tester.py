import io
import os
import json
import tempfile
import unittest
import contextlib
import unittest.mock

import specula
specula.init(0)  # Default target device

import numpy as np

from specula.base_value import BaseValue

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from iteras.lib.efficient_u_net import UNetRegressor
    from iteras.lib import cnn_checkpoint
    from iteras.lib.cnn_checkpoint import build_network
    from iteras.processing_objects.conv2d_net_tester import Conv2dNetTester
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
CHANNELS = 4
DEPTH = 1
H = W = 32
BATCH = 4


def save_checkpoint(net_path, stats_path=None, nmodes=NMODES, channels=CHANNELS, depth=DEPTH,
                    input_channels=2, n_frames=1, head_type='pooled', head_grid=32, **stats):
    """A checkpoint as Conv2dNetTrainer writes it: weights plus a stats file
    recording the network, zero-mean / unit-std normalization, and any extra
    stats given. stats_path is where that file lands (derived from net_path)."""
    network = dict(nmodes=nmodes, input_channels=input_channels, n_frames=n_frames,
                   channels=channels, depth=depth, conv_block_type=0, head_type=head_type,
                   head_grid=head_grid, dropout=0.0)
    model = build_network(network)
    cnn_checkpoint.save_checkpoint(model, net_path, dict(
        {'network': network, 'meanp': 0.0, 'stdp': 1.0,
          'meanmodes': [0.0] * nmodes, 'stdmodes': [1.0] * nmodes}, **stats))
    return model


def build_tester(tmp_dir, network_name='net.pth', nmodes=NMODES, channels=CHANNELS,
                  depth=DEPTH, input_channels=2, target_device_idx=-1, label_offset=1, **kwargs):
    """A tester of a new checkpoint; the labels fed by the helpers below have
    a placeholder first column, hence label_offset=1."""
    net_path = os.path.join(tmp_dir, network_name)
    stats_path = net_path.replace('.pth', '_stats.json')
    save_checkpoint(net_path, stats_path, nmodes=nmodes, channels=channels, depth=depth,
                    input_channels=input_channels)
    tester = Conv2dNetTester(network_filename=net_path, target_device_idx=target_device_idx,
                             label_offset=label_offset, **kwargs)
    return tester, net_path, stats_path


def feed_batch(tester, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    x_val.generation_time = x_val.seconds_to_t(1)
    y_val.generation_time = y_val.seconds_to_t(1)
    tester.inputs['input_2d_batch'].set(x_val)
    tester.inputs['labels'].set(y_val)
    tester.check_ready(1)


def batch_errors(tester, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    """The errors of the last prediction on the batch feed_batch() makes with
    these arguments."""
    rng = np.random.default_rng(seed)
    rng.standard_normal((batch, 2, h, w))
    labels = rng.standard_normal((batch, nmodes + 1)).astype(np.float32)[:, 1:]
    return np.asarray(tester.outputs['prediction'].value) - labels


def logged(mock_method):
    """All the messages passed to a mocked logger method, one per line."""
    return '\n'.join(str(c.args[0]) for c in mock_method.call_args_list)


def feed_batch_with_gain_mod(tester, gain_mod, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    """Like feed_batch(), but with an explicit 'gain_mod' input (shape
    (batch,) or (batch, 1)) connected alongside input_2d_batch/labels."""
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    g_val = BaseValue(value=np.asarray(gain_mod, dtype=np.float32).reshape(batch, 1))
    t = x_val.seconds_to_t(1)
    x_val.generation_time = t
    y_val.generation_time = t
    g_val.generation_time = t
    tester.inputs['input_2d_batch'].set(x_val)
    tester.inputs['labels'].set(y_val)
    tester.inputs['gain_mod'].set(g_val)
    tester.check_ready(1)


def feed_batch_with_baseline(tester, labels, baseline, batch=BATCH, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=labels.astype(np.float32))
    b_val = BaseValue(value=baseline.astype(np.float32))
    t = x_val.seconds_to_t(1)
    x_val.generation_time = t
    y_val.generation_time = t
    b_val.generation_time = t
    tester.inputs['input_2d_batch'].set(x_val)
    tester.inputs['labels'].set(y_val)
    tester.inputs['baseline'].set(b_val)
    tester.check_ready(1)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTesterConstruction(unittest.TestCase):

    def test_missing_model_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                Conv2dNetTester(network_filename=os.path.join(d, 'nope.pth'))

    def test_missing_stats_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            model = UNetRegressor(
                input_channels=1, output_size=NMODES, base_channels=CHANNELS,
                dropout_level=0.0, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            with self.assertRaises(FileNotFoundError):
                Conv2dNetTester(network_filename=net_path)

    def test_successful_construction_loads_stats_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            self.assertEqual(tester.meanp, 0.0)
            self.assertEqual(tester.stdp, 1.0)
            np.testing.assert_allclose(tester.meanmodes, [0.0] * NMODES)
            np.testing.assert_allclose(tester.stdmodes, [1.0] * NMODES)
            self.assertEqual(
                set(tester.outputs.keys()),
                {'loss', 'prediction'})
            self.assertTrue(tester.model.training is False)  # eval() was called


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTesterTrigger(unittest.TestCase):

    def test_input_channels_two_uses_raw_multichannel_input(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d, input_channels=2)
            self.assertEqual(tester.model.encoders[0].block[0].in_channels, 2)

            feed_batch(tester)
            tester.trigger()
            self.assertEqual(tester.outputs['prediction'].value.shape, (BATCH, NMODES))

    def test_input_channels_one_uses_a_single_map_per_frame_as_is(self):
        # out_slopes_map of slopes from intensity buffers one map per frame,
        # with no channel axis; the two-channel product convention must not
        # kick in (see Conv2dNetTrainer.network_input).
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d, input_channels=1)
            rng = np.random.default_rng(0)
            x_val = BaseValue(value=rng.standard_normal((BATCH, H, W)).astype(np.float32))
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            x_val.generation_time = y_val.generation_time = x_val.seconds_to_t(1)
            tester.inputs['input_2d_batch'].set(x_val)
            tester.inputs['labels'].set(y_val)
            tester.check_ready(1)
            tester.trigger()

            self.assertEqual(tester.count, BATCH)
            self.assertEqual(tester.outputs['prediction'].value.shape, (BATCH, NMODES))

    def test_trigger_computes_predictions_and_updates_stats(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester)
            tester.trigger()

            self.assertEqual(tester.count, BATCH)
            self.assertEqual(tester.outputs['prediction'].value.shape, (BATCH, NMODES))
            np.testing.assert_allclose(tester.outputs['loss'].value,
                                       np.mean(batch_errors(tester) ** 2), rtol=1e-5)

    def test_denormalization_uses_correct_affine_formula(self):
        """Regression test: predictions must be denormalized as
        pred_norm * std + mean, not pred_norm * (std + mean) -- the latter
        is only numerically indistinguishable from the former when mean=0,
        which is why it previously slipped through with all-zero meanmodes
        in the other tests here."""
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.meanmodes = np.array([10.0, -5.0, 2.0, 0.0, 3.0])
            tester.stdmodes = np.array([2.0, 1.0, 0.5, 4.0, 1.5])

            normalized_output = torch.zeros(BATCH, NMODES)
            tester.model = lambda x: normalized_output

            feed_batch(tester)
            tester.trigger()

            expected = np.tile(tester.meanmodes, (BATCH, 1))  # pred_norm == 0
            np.testing.assert_allclose(tester.outputs['prediction'].value, expected)

    def test_trigger_noop_when_inputs_missing(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.local_inputs['input_2d_batch'] = None
            tester.local_inputs['labels'] = None
            tester.trigger()
            self.assertEqual(tester.count, 0)

    def test_gain_mod_filters_out_below_threshold_samples(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch_with_gain_mod(tester, gain_mod=[1.0, 1.0, 0.0, 0.5])
            tester.trigger()

            # Default gain_mod_threshold=1.0: only the two full-gain samples
            # (indices 0, 1) are kept; the zero and the partial-gain one are
            # dropped.
            self.assertEqual(tester.count, 2)
            self.assertEqual(tester.outputs['prediction'].value.shape, (2, NMODES))

    def test_gain_mod_all_below_threshold_skips_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch_with_gain_mod(tester, gain_mod=[0.0, 0.0, 0.0, 0.0])
            tester.trigger()

            self.assertEqual(tester.count, 0)
            self.assertFalse(np.any(tester.sums['e2']))

    def test_gain_mod_unconnected_keeps_all_samples(self):
        # gain_mod is optional: with nothing wired to it, behavior is
        # unchanged from before this feature existed.
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester)
            tester.trigger()

            self.assertEqual(tester.count, BATCH)

    def test_gain_mod_custom_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.gain_mod_threshold = 0.5
            feed_batch_with_gain_mod(tester, gain_mod=[1.0, 0.5, 0.4, 0.0])
            tester.trigger()

            self.assertEqual(tester.count, 2)  # 1.0 and 0.5 both pass >= 0.5

    def test_multiple_triggers_accumulate_count_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, seed=1)
            tester.trigger()
            feed_batch(tester, seed=2)
            tester.trigger()

            self.assertEqual(tester.count, 2 * BATCH)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestResidualLearning(unittest.TestCase):
    """When a 'baseline' input is connected, the (denormalized) network
    output is a residual and must be added back to the baseline to form
    the final reported prediction, comparable to the full true labels."""

    def test_baseline_input_is_optional_and_unconnected_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            self.assertTrue(tester.inputs['baseline'].optional)

    def test_prediction_adds_baseline_back_to_network_output(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.meanmodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            tester.stdmodes = np.ones(NMODES)
            tester.model = lambda x: torch.zeros(BATCH, NMODES)  # predicted residual == meanmodes

            rng = np.random.default_rng(3)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            baseline = rng.standard_normal((BATCH, NMODES))

            feed_batch_with_baseline(tester, labels, baseline)
            tester.trigger()

            expected = np.tile(tester.meanmodes, (BATCH, 1)) + baseline
            np.testing.assert_allclose(tester.outputs['prediction'].value, expected, atol=1e-5)

    def test_error_is_computed_against_full_labels_not_residual(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            tester.meanmodes = np.zeros(NMODES)
            tester.stdmodes = np.ones(NMODES)

            rng = np.random.default_rng(4)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            baseline = rng.standard_normal((BATCH, NMODES))
            # network predicts the exact residual -> perfect reconstruction
            exact_residual = labels[:, 1:NMODES + 1] - baseline
            tester.model = lambda x: torch.tensor(exact_residual, dtype=torch.float32)

            feed_batch_with_baseline(tester, labels, baseline)
            tester.trigger()

            np.testing.assert_allclose(
                tester.outputs['prediction'].value, labels[:, 1:NMODES + 1], atol=1e-5)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTesterFinalize(unittest.TestCase):

    def test_finalize_with_no_data_prints_message(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            with unittest.mock.patch.object(tester.logger, 'warning') as mock_warning:
                tester.finalize()
            self.assertIn('No data processed', logged(mock_warning))

    def test_finalize_computes_expected_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, seed=3)
            tester.trigger()

            # Ground truth, derived directly from the per-sample error that
            # trigger() recorded, independent of the accumulator internals.
            error = batch_errors(tester, seed=3)
            expected_mae = np.mean(np.abs(error), axis=0)
            expected_rmse = np.sqrt(np.mean(error**2, axis=0))

            with unittest.mock.patch.object(tester.logger, 'info') as mock_info:
                tester.finalize()

            self.assertIn('Test statistics over 4 samples', logged(mock_info))
            self.assertEqual(tester.smape.shape, (NMODES,))
            np.testing.assert_allclose(tester.mean_absolute_error, expected_mae, rtol=1e-5)
            np.testing.assert_allclose(tester.root_mean_squared_error, expected_rmse, rtol=1e-5)

    def test_finalize_handles_varying_batch_sizes_across_triggers(self):
        with tempfile.TemporaryDirectory() as d:
            tester, _, _ = build_tester(d)
            feed_batch(tester, batch=4, seed=1)
            tester.trigger()
            errors = [batch_errors(tester, batch=4, seed=1)]
            feed_batch(tester, batch=7, seed=2)
            tester.trigger()
            errors.append(batch_errors(tester, batch=7, seed=2))

            with unittest.mock.patch.object(tester.logger, 'info'):
                tester.finalize()

            self.assertEqual(tester.count, 11)
            self.assertEqual(tester.mean_absolute_error.shape, (NMODES,))
            expected_mae = np.mean(np.abs(np.concatenate(errors, axis=0)), axis=0)
            np.testing.assert_allclose(tester.mean_absolute_error, expected_mae, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
