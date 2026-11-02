import io
import os
import json
import tempfile
import unittest
import unittest.mock
import contextlib

import specula
specula.init(0)  # Default target device

import numpy as np

from specula import cpuArray
from specula.base_value import BaseValue

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from iteras.lib.efficient_u_net import UNetRegressor, SpatialRegressionHead
    from iteras.lib import cnn_checkpoint
    from iteras.lib.cnn_checkpoint import build_network
    from iteras.processing_objects.conv2d_net_trainer import (
        Conv2dNetTrainer,
        EarlyStopping,
        to_torch,
    )
    from iteras.processing_objects.conv2d_net_tester import Conv2dNetTester
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 20
CHANNELS = 4
DEPTH = 2
H = W = 32
BATCH = 8


def build_trainer(tmp_dir, network_name='testnet.pth', nmodes=NMODES, channels=CHANNELS,
                   depth=DEPTH, epoch_len=1, patience=600, val_split=0.25,
                   load_from_file=False, conv_block_type=0, input_channels=2,
                   label_offset=1, loss_delta=20.0, target_device_idx=-1,
                   **kwargs):
    network_filename = os.path.join(tmp_dir, network_name)
    return Conv2dNetTrainer(
        network_filename=network_filename,
        nmodes=nmodes,
        channels=channels,
        depth=depth,
        epoch_len=epoch_len,
        patience=patience,
        val_split=val_split,
        load_from_file=load_from_file,
        conv_block_type=conv_block_type,
        input_channels=input_channels,
        label_offset=label_offset,
        loss_delta=loss_delta,
        target_device_idx=target_device_idx,
        **kwargs,
    )


def feed_batch(trainer, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=rng.standard_normal((batch, nmodes + 1)).astype(np.float32))
    x_val.generation_time = x_val.seconds_to_t(1)
    y_val.generation_time = y_val.seconds_to_t(1)
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.check_ready(1)


def logged(mock_method):
    """All the messages passed to a mocked logger method, one per line."""
    return '\n'.join(str(c.args[0]) for c in mock_method.call_args_list)


def linear_batch(rng, w, n, side=8):
    """Batch whose labels (after a placeholder column, label_offset=1) are an
    exact linear function of the in-pupil pixels of a (n, 2, side, side)
    input."""
    yy, xx = np.mgrid[:side, :side] - (side - 1) / 2
    pupil = (xx ** 2 + yy ** 2) < (side / 2) ** 2
    x = np.zeros((n, 2, side, side), np.float32)
    x[:, :, pupil] = rng.standard_normal((n, 2, int(pupil.sum())))
    modes = x[:, :, pupil].reshape(n, -1) @ w + 3.0
    labels = np.concatenate([np.zeros((n, 1)), modes], axis=1).astype(np.float32)
    return x, labels


def feed_arrays(trainer, x, labels, t_seconds=1):
    x_val = BaseValue(value=x)
    y_val = BaseValue(value=labels)
    t = x_val.seconds_to_t(t_seconds)
    x_val.generation_time = y_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.check_ready(t)


def feed_batch_with_baseline(trainer, labels, baseline, batch=BATCH, h=H, w=W, seed=0):
    """Like feed_batch(), but with explicit labels/baseline arrays
    (shape (batch, n)) instead of random ones, and a 'baseline' input."""
    rng = np.random.default_rng(seed)
    x_val = BaseValue(value=rng.standard_normal((batch, 2, h, w)).astype(np.float32))
    y_val = BaseValue(value=labels.astype(np.float32))
    b_val = BaseValue(value=baseline.astype(np.float32))
    t = x_val.seconds_to_t(1)
    x_val.generation_time = t
    y_val.generation_time = t
    b_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['baseline'].set(b_val)
    trainer.check_ready(1)


def feed_batch_with_gain_mod(trainer, gain_mod, batch=BATCH, nmodes=NMODES, h=H, w=W, seed=0):
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
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['gain_mod'].set(g_val)
    trainer.check_ready(1)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestNetworkFilenameHandling(unittest.TestCase):

    def test_pth_is_appended_without_extension(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet', nmodes=3, depth=1)
            self.assertEqual(trainer.network_filename, os.path.join(d, 'mynet.pth'))
            self.assertEqual(trainer.stats_filename, os.path.join(d, 'mynet_stats.json'))

    def test_filename_is_used_verbatim(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='mynet.pt', nmodes=3, depth=1)
            self.assertEqual(trainer.network_filename, os.path.join(d, 'mynet.pt'))


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerLoading(unittest.TestCase):

    def _save_checkpoint(self, path, stats_path=None, nmodes=NMODES, channels=CHANNELS, depth=DEPTH):
        """A checkpoint matching build_trainer's defaults (stats_path is where
        its stats land, derived from path)."""
        network = dict(nmodes=nmodes, input_channels=2, n_frames=1, channels=channels,
                       depth=depth, conv_block_type=0, head_type='pooled', head_grid=32,
                       dropout=0.01)
        model = build_network(network)
        cnn_checkpoint.save_checkpoint(model, path, {
            'network': network, 'meanp': 0.5, 'stdp': 1.5,
            'meanmodes': [0.1] * nmodes, 'stdmodes': [1.1] * nmodes})
        return model

    def test_missing_checkpoint_falls_back_to_new_model(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='nofile_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            # Falls back gracefully: a fresh model is created, no stats loaded.
            self.assertIsNone(trainer.meanp)
            self.assertIsNone(trainer.stdp)
            self.assertIsInstance(trainer.model, torch.nn.Module)

    def test_successful_load_restores_weights_and_stats(self):
        with tempfile.TemporaryDirectory() as d:
            # filename already tagged so Conv2dNetTrainer won't rewrite it
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            source_model = self._save_checkpoint(net_path, stats_path)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)

            self.assertEqual(trainer.meanp, 0.5)
            self.assertEqual(trainer.stdp, 1.5)
            # Regression: meanmodes/stdmodes must be plain xp arrays (matching
            # how trigger() uses them), not torch tensors -- a previous bug
            # stored them as torch.tensor(...), which this test's old
            # trainer.meanmodes.numpy() call happened to paper over.
            self.assertNotIsInstance(trainer.meanmodes, torch.Tensor)
            self.assertNotIsInstance(trainer.stdmodes, torch.Tensor)
            np.testing.assert_allclose(trainer.meanmodes, [0.1] * NMODES)
            np.testing.assert_allclose(trainer.stdmodes, [1.1] * NMODES)

            loaded_state = trainer._inner_model().state_dict()
            source_state = source_model.state_dict()
            for key in source_state:
                torch.testing.assert_close(loaded_state[key].cpu(), source_state[key].cpu())

    def test_successful_load_preserves_stats_across_first_trigger(self):
        # The loaded normalization must carry on from where it was, not be
        # re-estimated from scratch on this one batch. The first batch after
        # resuming has no time interval to weigh an update by, so it leaves
        # the statistics exactly as loaded.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            self._save_checkpoint(net_path, stats_path)

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            loaded_meanmodes = np.array(trainer.meanmodes, copy=True)

            feed_batch(trainer, seed=7)
            trainer.trigger()

            np.testing.assert_allclose(trainer.meanmodes, loaded_meanmodes)

    def test_checkpoint_of_a_different_network_raises_instead_of_being_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing.pth')
            self._save_checkpoint(net_path, depth=DEPTH + 1)
            with self.assertRaisesRegex(ValueError, f'depth={DEPTH} \\(checkpoint: {DEPTH + 1}\\)'):
                build_trainer(d, network_name='existing.pth', load_from_file=True)

    def test_checkpoint_without_stats_file_warns_and_sets_none(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'existing_m20_ch4_dp0.010.pth')
            model = UNetRegressor(
                input_channels=2, output_size=NMODES, base_channels=CHANNELS,
                dropout_level=0.01, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            # no stats file written on purpose

            trainer = build_trainer(d, network_name='existing_m20_ch4_dp0.010.pth',
                                     load_from_file=True)
            self.assertIsNone(trainer.meanp)
            self.assertIsNone(trainer.stdp)
            self.assertIsNone(trainer.meanmodes)
            self.assertIsNone(trainer.stdmodes)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerDevice(unittest.TestCase):

    def test_cpu_object_trains_on_cpu(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, target_device_idx=-1)
            self.assertEqual(trainer.device.type, 'cpu')

    @unittest.skipIf(not TORCH_AVAILABLE or specula.cp is None or not torch.cuda.is_available(), "no GPU")
    def test_gpu_object_trains_on_its_gpu_only(self):
        # Not on every visible GPU: DataParallel is opt-in (data_parallel)
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, target_device_idx=0)
            self.assertEqual(str(trainer.device), 'cuda:0')
            self.assertNotIsInstance(trainer.model, torch.nn.DataParallel)

    def test_cpu_object_ignores_data_parallel(self):
        with tempfile.TemporaryDirectory() as d, \
                unittest.mock.patch.object(torch.cuda, 'device_count', return_value=2):
            trainer = build_trainer(d, target_device_idx=-1, data_parallel=True)
            self.assertEqual(trainer.device.type, 'cpu')
            self.assertNotIsInstance(trainer.model, torch.nn.DataParallel)

    @unittest.skipIf(not TORCH_AVAILABLE or specula.cp is None or not torch.cuda.is_available()
                     or torch.cuda.device_count() < 2, "fewer than 2 GPUs")
    def test_data_parallel_starts_from_the_object_gpu(self):
        # DataParallel wants the model on device_ids[0]
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, target_device_idx=1, data_parallel=True)
            self.assertIsInstance(trainer.model, torch.nn.DataParallel)
            self.assertEqual(trainer.model.device_ids[0], 1)
            self.assertEqual(sorted(trainer.model.device_ids), list(range(torch.cuda.device_count())))


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestNormalization(unittest.TestCase):
    """The input normalization is set from the first batch and kept; the
    per-mode output normalization follows a moving average over simulated
    time, and every update rescales the output layers so the denormalized
    predictions don't move."""

    def _feed(self, trainer, labels_scale=1.0, labels_offset=0.0, t_seconds=1, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((BATCH, 2, H, W)).astype(np.float32)
        labels = (rng.standard_normal((BATCH, NMODES + 1)) * labels_scale + labels_offset).astype(np.float32)
        feed_arrays(trainer, x, labels, t_seconds=t_seconds)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.trigger()
        return x, labels[:, 1:NMODES + 1]

    def test_norm_timescale_must_be_positive(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (0.0, -1.0):
                with self.assertRaises(ValueError):
                    build_trainer(d, norm_timescale=bad)

    def test_input_normalization_is_set_from_the_first_batch_and_kept(self):
        with tempfile.TemporaryDirectory() as d:
            # input_channels=2: the buffered maps are the network input as is
            trainer = build_trainer(d, epoch_len=1, input_channels=2)
            x, _ = self._feed(trainer, t_seconds=1)
            self.assertAlmostEqual(trainer.meanp, float(np.mean(x)), places=5)
            self.assertAlmostEqual(trainer.stdp, float(np.std(x)), places=4)
            first = (trainer.meanp, trainer.stdp)
            self._feed(trainer, labels_scale=5.0, t_seconds=5, seed=1)
            self.assertEqual((trainer.meanp, trainer.stdp), first)

    def test_output_statistics_follow_a_moving_average_over_simulated_time(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, norm_timescale=2.0)
            _, y1 = self._feed(trainer, t_seconds=1)
            np.testing.assert_allclose(cpuArray(trainer.meanmodes), y1.mean(0), atol=1e-5)
            _, y2 = self._feed(trainer, labels_offset=10.0, t_seconds=3, seed=1)
            alpha = 1 - np.exp(-2.0 / 2.0)          # 2 s elapsed, 2 s time constant
            m1 = (1 - alpha) * y1.mean(0) + alpha * y2.mean(0)
            m2 = (1 - alpha) * (y1.astype(np.float64) ** 2).mean(0) + alpha * (y2.astype(np.float64) ** 2).mean(0)
            np.testing.assert_allclose(cpuArray(trainer.meanmodes), m1, rtol=1e-4)
            np.testing.assert_allclose(cpuArray(trainer.stdmodes), np.sqrt(m2 - m1 ** 2), rtol=1e-4)

    def test_updates_leave_the_denormalized_predictions_unchanged(self):
        # The point of the rescaling: the statistics move, the predictions
        # don't. Checked for both heads (one output layer, and two summed).
        for head in ('pooled', 'spatial'):
            with self.subTest(head=head), tempfile.TemporaryDirectory() as d:
                trainer = build_trainer(d, epoch_len=1, head_type=head,
                                        input_channels=2, norm_timescale=1.0)
                x, _ = self._feed(trainer, t_seconds=1)
                model = trainer._inner_model().eval()
                inp = to_torch((x - trainer.meanp) / trainer.stdp, trainer.device)

                def predict():
                    with torch.no_grad():
                        return (model(inp) * to_torch(trainer.stdmodes, trainer.device)
                                + to_torch(trainer.meanmodes, trainer.device)).cpu().numpy()

                before = predict()
                old_std = cpuArray(trainer.stdmodes).copy()
                trainer.current_time = trainer.seconds_to_t(4)      # 3 s later: a big update
                rng = np.random.default_rng(2)
                trainer._update_output_normalization(rng.standard_normal((BATCH, NMODES)) * 7 + 20)
                self.assertGreater(np.abs(cpuArray(trainer.stdmodes) / old_std - 1).max(), 0.5)
                np.testing.assert_allclose(predict(), before, rtol=1e-4, atol=1e-4)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestLossDelta(unittest.TestCase):
    """loss_delta is the Huber-loss transition point (physical mode
    units); it must default to 20.0, be validated, and actually be passed
    through to the underlying nn.HuberLoss."""

    def test_default_matches_documented_value(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            self.assertEqual(trainer.loss_delta, 20.0)

    def test_rejects_non_positive_values(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_trainer(d, loss_delta=0.0)
            with self.assertRaises(ValueError):
                build_trainer(d, loss_delta=-5.0)

    def test_custom_delta_is_used_by_loss_fn(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, loss_delta=5.0)
            self.assertIsInstance(trainer.loss_fn, torch.nn.HuberLoss)
            self.assertEqual(trainer.loss_fn.delta, 5.0)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestEarlyStopping(unittest.TestCase):

    def test_first_call_never_stops(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        self.assertFalse(es(1.0))

    def test_improvement_resets_counter(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        es(1.0)
        self.assertFalse(es(0.5))   # improvement
        self.assertFalse(es(0.9))   # worse, counter=1
        self.assertFalse(es(0.4))   # improvement, counter reset
        self.assertFalse(es(0.9))   # worse, counter=1

    def test_stops_after_patience_exceeded(self):
        es = EarlyStopping(patience=2, min_delta=1e-3)
        es(1.0)                     # baseline
        self.assertFalse(es(1.0))   # no improvement, counter=1
        self.assertTrue(es(1.0))    # no improvement, counter=2 >= patience


def constant_output_model(nmodes):
    """A model that always outputs zeros in normalized space, regardless of
    the input, so that the physical-space loss can be computed by hand
    (defined here, not at module level, since torch may be missing)."""
    class ConstantOutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # A dummy trainable parameter, multiplied by zero, so the
            # output still has a grad_fn (loss.backward() requires this)
            # while never actually changing the (constant) output value.
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.zeros(x.shape[0], nmodes, dtype=x.dtype) + 0.0 * self.dummy
    return ConstantOutputModel()


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestPhysicalSpaceLoss(unittest.TestCase):
    """The training loss must be a plain (unweighted) Huber loss computed
    in physical (de-normalized) mode units: preds*stdmodes+meanmodes vs.
    the true physical labels -- not the old per-mode-weighted MSE in
    normalized/z-scored space, and not a plain (outlier-sensitive) MSE in
    physical units either (see loss_delta's docstring)."""

    def test_loss_is_plain_huber_in_physical_units(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, val_split=0.25)
            trainer.model = constant_output_model(NMODES)
            trainer.device = torch.device('cpu')

            rng = np.random.default_rng(0)
            x_val = BaseValue(value=rng.standard_normal((BATCH, 2, H, W)).astype(np.float32))
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            labels = cpuArray(y_val.value)[:, 1:NMODES + 1]  # label_offset=1 default
            t = x_val.seconds_to_t(1)
            x_val.generation_time = t
            y_val.generation_time = t
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)

            torch.manual_seed(1234)
            trainer.trigger()

            # The training rows: all but the most recent val_split of them
            n_val = max(1, int(BATCH * 0.25))
            n_train = BATCH - n_val
            train_idx = np.arange(n_train)

            # First trigger: meanmodes/stdmodes are computed directly from
            # the full physical-space batch of labels.
            meanmodes = cpuArray(trainer.meanmodes)
            preds_physical = np.broadcast_to(meanmodes, (n_train, NMODES))
            diff = preds_physical - labels[train_idx]
            abs_diff = np.abs(diff)
            delta = trainer.loss_delta
            # nn.HuberLoss's own formula (mean reduction): quadratic below
            # delta, linear beyond it -- see loss_delta's docstring.
            per_entry = np.where(abs_diff < delta,
                                  0.5 * diff ** 2,
                                  delta * (abs_diff - 0.5 * delta))
            expected_loss = np.mean(per_entry)

            self.assertAlmostEqual(trainer.loss, expected_loss, places=4)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetTrainerTrigger(unittest.TestCase):

    def test_trigger_noop_when_inputs_not_set(self):
        # Exercise trigger()'s defensive "no data yet" branch directly:
        # going through check_ready()/setup() would raise, since the inputs
        # are declared as non-optional.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            trainer.local_inputs['input_2d_batch'] = None
            trainer.local_inputs['labels'] = None
            trainer.trigger()
            self.assertEqual(trainer.step_count, 0)

    def test_trigger_noop_when_should_stop(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            feed_batch(trainer)
            trainer.should_stop = True
            trainer.trigger()
            self.assertEqual(trainer.step_count, 0)

    def test_input_channels_two_uses_raw_multichannel_input(self):
        # With input_channels=2, the buffered (B, 2, H, W) value must be fed
        # to the network as-is (two input channels), not collapsed into a
        # single-channel product as with the default input_channels=1.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=2)
            self.assertEqual(trainer._inner_model().encoders[0].block[0].in_channels, 2)

            feed_batch(trainer)
            trainer.trigger()
            self.assertEqual(trainer.step_count, 1)

    def test_channel_product_takes_the_product_of_a_two_channel_value(self):
        # A (B, 2, H, W) buffered value whose two channels (e.g. amplitude
        # and phase) are multiplied into the single-channel network input.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=1, channel_product=True)
            self.assertEqual(trainer._inner_model().encoders[0].block[0].in_channels, 1)

            rng = np.random.default_rng(0)
            value = rng.standard_normal((BATCH, 2, H, W)).astype(np.float32)
            expected = value[:, 1] * value[:, 0]
            x_val = BaseValue(value=value)
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            x_val.generation_time = y_val.generation_time = x_val.seconds_to_t(1)
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertAlmostEqual(float(trainer.meanp), float(np.mean(expected)), places=4)

    def test_channel_product_needs_one_input_channel(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_trainer(d, input_channels=2, channel_product=True)

    def test_input_channels_must_match_the_maps(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=1)
            feed_batch(trainer)       # (B, 2, H, W) maps
            with self.assertRaisesRegex(ValueError, 'channel'):
                trainer.trigger()

    def test_labels_narrower_than_label_offset_plus_nmodes(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, label_offset=2)
            feed_batch(trainer)       # NMODES + 1 labels
            with self.assertRaisesRegex(ValueError, 'label_offset'):
                trainer.trigger()

    def test_maps_too_small_for_the_depth(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, depth=3)
            feed_batch(trainer, h=4, w=4)
            with self.assertRaisesRegex(ValueError, 'too small for depth=3'):
                trainer.trigger()

    def test_input_channels_one_uses_a_single_map_per_frame_as_is(self):
        # A buffered value with one map per frame and no channel axis
        # (out_slopes_map of slopes from intensity): used unchanged, with
        # the channel axis added for the network.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, input_channels=1)
            rng = np.random.default_rng(0)
            value = rng.standard_normal((BATCH, H, W)).astype(np.float32)
            x_val = BaseValue(value=value)
            y_val = BaseValue(value=rng.standard_normal((BATCH, NMODES + 1)).astype(np.float32))
            x_val.generation_time = y_val.generation_time = x_val.seconds_to_t(1)
            trainer.inputs['input_2d_batch'].set(x_val)
            trainer.inputs['labels'].set(y_val)
            trainer.check_ready(1)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertAlmostEqual(float(trainer.meanp), float(np.mean(value)), places=4)
            self.assertEqual(trainer.val_inputs.shape[1:], (1, H, W))

    def test_mode_gain_moving_average_uses_mode_gain_alpha(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, mode_gain_alpha=0.1)
            targets = torch.randn(200, NMODES, dtype=torch.float64)
            trainer._update_mode_gain(0.5 * targets, targets)     # first: the gain itself
            np.testing.assert_allclose(trainer.mode_gain, 0.5, atol=1e-9)
            trainer._update_mode_gain(1.0 * targets, targets)     # then 0.9 old + 0.1 new
            np.testing.assert_allclose(trainer.mode_gain, 0.55, atol=1e-9)

    def test_mode_gain_alpha_out_of_range_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                build_trainer(d, mode_gain_alpha=0.0)

    def test_mode_gain_is_measured_and_saved(self):
        # A model whose output is a known fraction of the target must come out
        # with that fraction as its per-mode gain in the stats file.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            for i in range(10):     # first periodic save at step 10
                feed_batch(trainer, seed=i)
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.trigger()
            shrink = 0.5
            targets = trainer.val_targets
            mean_t = to_torch(trainer.meanmodes, targets.device)
            std_t = to_torch(trainer.stdmodes, targets.device)

            class Shrinking(torch.nn.Module):
                """Predicts the targets, shrunk towards their mean by `shrink`,
                in the normalized space the trainer denormalizes from."""
                def forward(self, x):
                    return shrink * (targets - mean_t) / std_t

            trainer.model = Shrinking()
            trainer.mode_gain = None
            trainer._measure_mode_gain(trainer.val_inputs, targets, mean_t, std_t)
            np.testing.assert_allclose(trainer.mode_gain, np.full(NMODES, shrink), atol=1e-4)

            trainer.step_count = 10
            with contextlib.redirect_stdout(io.StringIO()):
                trainer._save(1.0)
            with open(trainer.stats_filename) as f:
                np.testing.assert_allclose(json.load(f)['mode_gain'], np.full(NMODES, shrink), atol=1e-4)

    def test_trigger_runs_one_training_step(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)
            self.assertIsNotNone(trainer.val_inputs)
            self.assertIsNotNone(trainer.meanmodes)
            self.assertTrue(np.isfinite(trainer.loss))
            # Saving is periodic (every 10 steps), not gated on improvement
            # -- see test_model_saved_every_10_steps_not_on_improvement.
            self.assertFalse(os.path.isfile(trainer.network_filename))

    def test_model_saved_every_10_steps_not_on_improvement(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            for i in range(9):
                feed_batch(trainer, seed=i)
                trainer.trigger()
            self.assertEqual(trainer.step_count, 9)
            self.assertFalse(os.path.isfile(trainer.network_filename))

            feed_batch(trainer, seed=9)
            trainer.trigger()
            self.assertEqual(trainer.step_count, 10)
            self.assertTrue(os.path.isfile(trainer.network_filename))
            self.assertTrue(os.path.isfile(trainer.stats_filename))

    def test_saved_stats_min_loss_reflects_this_saves_eval_loss_not_best_ever(self):
        # Regression test: after switching from "save on improvement" to
        # "save every 10 steps", the persisted min_loss must reflect
        # *this save's* eval_loss (matching the weights actually being
        # written), not self.min_loss (best-ever seen, which can now
        # refer to a step other than the one just saved). Otherwise a
        # resumed run's first evaluation looks like an unexplained
        # regression against a number these exact weights never produced.
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            # Pretend an impossibly low FVU was already seen (as if from
            # a different, no-longer-current set of weights) -- it must
            # not leak into what gets saved for the weights saved here.
            trainer.best_val_fvu = -100.0

            for i in range(10):
                feed_batch(trainer, seed=i)
                trainer.trigger()

            with open(trainer.stats_filename) as f:
                stats = json.load(f)
            # Real values are ordinary positive numbers; -100 would only
            # appear if the best-ever value leaked into the save.
            self.assertGreater(stats['min_loss'], -100.0)
            self.assertGreater(stats['val_fvu'], 0.0)
            self.assertEqual(trainer.best_val_fvu, -100.0)

    def test_trigger_creates_missing_parent_directory_before_saving(self):
        with tempfile.TemporaryDirectory() as d:
            nested_dir = os.path.join(d, 'does', 'not', 'exist', 'yet')
            trainer = build_trainer(nested_dir, epoch_len=1)
            self.assertFalse(os.path.isdir(nested_dir))

            for i in range(10):
                feed_batch(trainer, seed=i)
                trainer.trigger()

            self.assertTrue(os.path.isfile(trainer.network_filename))
            self.assertTrue(os.path.isfile(trainer.stats_filename))

    def test_second_trigger_accumulates_validation_set(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, val_split=0.25)
            feed_batch(trainer, seed=1)
            trainer.trigger()
            trainer.post_trigger()
            n_val_after_first = trainer.val_inputs.shape[0]

            feed_batch(trainer, seed=2)
            trainer.trigger()
            trainer.post_trigger()
            n_val_after_second = trainer.val_inputs.shape[0]

            self.assertGreater(n_val_after_second, n_val_after_first)

    def test_post_trigger_sets_loss_output(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()
            trainer.post_trigger()
            self.assertAlmostEqual(
                trainer.outputs['loss'].value, trainer.loss, places=5)

    def test_post_trigger_before_any_trigger_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            # check_ready() marks inputs as changed; post_trigger() itself
            # must tolerate the missing 'loss' attribute (no trigger() yet).
            feed_batch(trainer)
            trainer.post_trigger()

    def test_finalize_runs_without_error(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()
            trainer.post_trigger()

            with unittest.mock.patch.object(trainer.logger, 'info') as mock_info:
                trainer.finalize()
            self.assertIn('Training complete', logged(mock_info))


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestResidualLearning(unittest.TestCase):
    """When a 'baseline' input is connected, the network must be trained
    on labels - baseline instead of the raw labels."""

    def test_baseline_input_is_optional_and_unconnected_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d)
            self.assertTrue(trainer.inputs['baseline'].optional)

    def test_zero_residual_when_baseline_equals_labels(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            rng = np.random.default_rng(0)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            # baseline equal to the true labels over the sliced mode range
            # (label_offset=1 by default): residual must be exactly zero.
            baseline = np.zeros((BATCH, NMODES))
            baseline[:] = labels[:, 1:NMODES + 1]

            feed_batch_with_baseline(trainer, labels, baseline)
            trainer.trigger()

            # meanmodes/stdmodes are computed directly from the (residual)
            # training target, independent of the network itself.
            np.testing.assert_allclose(trainer.meanmodes, np.zeros(NMODES), atol=1e-6)
            np.testing.assert_allclose(trainer.stdmodes, np.full(NMODES, 1e-8), atol=1e-9)

    def test_nonzero_residual_matches_labels_minus_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            rng = np.random.default_rng(1)
            labels = rng.standard_normal((BATCH, NMODES + 1))
            baseline = rng.standard_normal((BATCH, NMODES))

            feed_batch_with_baseline(trainer, labels, baseline)
            trainer.trigger()

            expected_residual = labels[:, 1:NMODES + 1] - baseline
            np.testing.assert_allclose(
                trainer.meanmodes, np.mean(expected_residual, axis=0), atol=1e-5)

    def test_baseline_offset_selects_correct_columns(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1, nmodes=3)
            trainer.baseline_offset = 2
            rng = np.random.default_rng(2)
            labels = rng.standard_normal((BATCH, 4))
            # baseline has extra leading columns that must be skipped
            baseline_full = rng.standard_normal((BATCH, 5))

            feed_batch_with_baseline(trainer, labels, baseline_full, batch=BATCH)
            trainer.trigger()

            expected_residual = labels[:, 1:4] - baseline_full[:, 2:5]
            np.testing.assert_allclose(
                trainer.meanmodes, np.mean(expected_residual, axis=0), atol=1e-5)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestGainModFiltering(unittest.TestCase):
    """When a 'gain_mod' input is connected, samples where the loop wasn't
    at full gain must be dropped from the batch before training (mirrors
    Conv2dNetTester's gain_mod_threshold)."""

    def test_gain_mod_filters_out_below_threshold_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch_with_gain_mod(trainer, gain_mod=[1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
            trainer.trigger()

            # Default gain_mod_threshold=1.0: only the 6 full-gain samples
            # are kept; the zero and the partial-gain one are dropped.
            self.assertIsNotNone(trainer.meanmodes)
            self.assertEqual(trainer.meanmodes.shape, (NMODES,))

    def test_gain_mod_all_below_threshold_skips_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch_with_gain_mod(trainer, gain_mod=[0.0] * BATCH)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 0)
            self.assertIsNone(trainer.meanmodes)

    def test_gain_mod_unconnected_keeps_all_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            feed_batch(trainer)
            trainer.trigger()

            self.assertEqual(trainer.step_count, 1)

    def test_gain_mod_custom_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, epoch_len=1)
            trainer.gain_mod_threshold = 0.5
            feed_batch_with_gain_mod(trainer, gain_mod=[1.0, 0.5, 0.4, 0.0, 1.0, 0.5, 0.4, 0.0])
            trainer.trigger()

            # 1.0 and 0.5 both pass >= 0.5 -> 4 of the 8 samples kept.
            self.assertEqual(trainer.step_count, 1)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestResume(unittest.TestCase):

    def test_learning_rate_continues_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            opts = dict(network_name='net.pth', input_channels=2, lr_decay=0.9)
            trainer = build_trainer(d, **opts)
            with contextlib.redirect_stdout(io.StringIO()):
                for i in range(10):     # saved at step 10
                    feed_batch(trainer, seed=i)
                    trainer.trigger()
            lr = trainer.optimizer.param_groups[0]['lr']
            self.assertAlmostEqual(lr, 1e-3 * 0.9 ** 10)
            with contextlib.redirect_stdout(io.StringIO()):
                resumed = build_trainer(d, load_from_file=True, **opts)
            self.assertAlmostEqual(resumed.optimizer.param_groups[0]['lr'], lr)

    def _pupil_weights(self, rng, side=8):
        yy, xx = np.mgrid[:side, :side] - (side - 1) / 2
        n_pix = 2 * int(((xx ** 2 + yy ** 2) < (side / 2) ** 2).sum())
        return rng.standard_normal((n_pix, NMODES)) * 10

    def test_checkpoint_resumes_with_its_stats_and_loads_in_tester(self):
        with tempfile.TemporaryDirectory() as d:
            opts = dict(network_name='net.pth', input_channels=2)
            trainer = build_trainer(d, **opts)
            rng = np.random.default_rng(4)
            w = self._pupil_weights(rng)
            with contextlib.redirect_stdout(io.StringIO()):
                t = 0
                while trainer.step_count < 10:  # first periodic save
                    t += 1
                    feed_arrays(trainer, *linear_batch(rng, w, 100), t_seconds=t)
                    trainer.trigger()
                resumed = build_trainer(d, load_from_file=True, **opts)
            with open(trainer.stats_filename) as f:
                saved_stats = json.load(f)
            np.testing.assert_allclose(cpuArray(resumed.meanmodes), saved_stats['meanmodes'], rtol=1e-6)
            self.assertEqual(resumed.stdp, saved_stats['stdp'])

            # the tester needs nothing but the file name
            tester = Conv2dNetTester(network_filename=trainer.network_filename, target_device_idx=-1)
            saved = torch.load(trainer.network_filename, map_location='cpu')
            for k, v in tester.model.state_dict().items():
                torch.testing.assert_close(v, saved[k])


def time_tagged_batch(n, offset=0, h=H, w=W, seed=0):
    """Random inputs; labels whose first trained mode (label_offset=1) is
    offset + the row's position in time, to track which rows went where."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2, h, w)).astype(np.float32)
    labels = rng.standard_normal((n, NMODES + 1)).astype(np.float32)
    labels[:, 1] = offset + np.arange(n)
    return x, labels


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestReplayAndGradClip(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, replay_size=100, replay_subsample=2, replay_ratio=1.0,
                    replay_batch=4)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def _feed(self, trainer, n=32, offset=0, t_seconds=1):
        with contextlib.redirect_stdout(io.StringIO()):
            feed_arrays(trainer, *time_tagged_batch(n, offset, seed=offset), t_seconds=t_seconds)
            trainer.trigger()

    def test_replay_is_off_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, input_channels=2)
            self._feed(trainer)
            self.assertIsNone(trainer.replay_x)
            self.assertEqual(trainer.step_count, 1)

    def test_buffer_gets_every_nth_training_row_in_time_order(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            self._feed(trainer)
            # 32 rows, 8 to validation, 24 training rows -> every 2nd = 12.
            self.assertEqual(trainer.replay_count, 12)
            t = trainer.replay_y[:12, 0].numpy()
            self.assertTrue(np.all(np.diff(t) >= 2))   # time-ordered, subsampled
            val_t = trainer.val_targets[:, 0].cpu().tolist()
            self.assertFalse(set(t.tolist()) & set(val_t))   # no validation rows

    def test_validation_is_the_most_recent_block_of_the_batch(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            self._feed(trainer)
            # 32 rows in time order, val_split 0.25: the last 8 validate
            np.testing.assert_array_equal(trainer.val_targets[:, 0].cpu().numpy(), np.arange(24, 32))

    def test_validation_keeps_the_most_recent_val_size_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, val_size=12)
            for k in range(3):
                self._feed(trainer, offset=100 * k, t_seconds=k + 1)
            # 8 validation rows per batch (the last 8 of 32): batch 0's are
            # gone, the last 4 of batch 1 and all of batch 2 remain
            np.testing.assert_array_equal(trainer.val_targets[:, 0].cpu().numpy(),
                                          np.r_[128:132, 224:232])
            with self.assertRaises(ValueError):
                self._build(d, val_size=0)

    def test_mode_gain_is_measured_on_the_training_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            with unittest.mock.patch.object(trainer, '_update_mode_gain') as update:
                self._feed(trainer)
            targets = update.call_args.args[1]
            # 32 rows in time order: the first 24 train, the last 8 validate
            np.testing.assert_array_equal(targets[:, 0].cpu().numpy(), np.arange(24))

    def test_step_line_reports_the_validation_fvu(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            with unittest.mock.patch.object(trainer.logger, 'info') as mock_info:
                feed_arrays(trainer, *time_tagged_batch(32))
                trainer.trigger()
            self.assertGreater(trainer.val_fvu, 0.0)
            self.assertEqual(trainer.best_val_fvu, trainer.val_fvu)
            self.assertIn(f'(FVU {trainer.val_fvu:.3f}) | Best FVU', logged(mock_info))

    def test_ring_buffer_keeps_the_most_recent_samples(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, replay_size=20)
            for k in range(3):
                self._feed(trainer, offset=100 * k, t_seconds=k + 1)
            self.assertEqual(trainer.replay_count, 20)
            t = trainer.replay_y[:, 0].numpy()
            self.assertGreaterEqual(t.min(), 100)          # batch 0 fully overwritten
            self.assertEqual(int(np.sum(t >= 200)), 12)   # all of batch 2 kept

    def test_each_trigger_draws_replay_ratio_samples_per_new_sample(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, replay_ratio=2.0)
            with unittest.mock.patch.object(trainer.optimizer, 'step',
                                            wraps=trainer.optimizer.step) as step:
                self._feed(trainer)
            # 12 new samples in the buffer, 2 draws each, 4 per minibatch
            self.assertEqual(step.call_count, 6)

    def test_at_least_one_step_per_trigger(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, replay_ratio=0.01)
            with unittest.mock.patch.object(trainer.optimizer, 'step',
                                            wraps=trainer.optimizer.step) as step:
                self._feed(trainer)
            self.assertEqual(step.call_count, 1)

    def test_replay_ratio_must_be_positive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self._build(d, replay_ratio=0)

    def test_single_channel_legacy_input_works_with_replay(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, input_channels=1, channel_product=True)
            self._feed(trainer)
            self.assertEqual(trainer.step_count, 1)

    def test_lr_decay_per_trigger_with_floor_and_no_plateau_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, lr_decay=0.5)
            with unittest.mock.patch.object(trainer.plateau_scheduler, 'step') as plateau:
                self._feed(trainer, t_seconds=1)
                self._feed(trainer, offset=100, t_seconds=2)
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]['lr'], 1e-3 * 0.25)
            plateau.assert_not_called()

            floored = self._build(d, network_name='f.pth', lr_decay=0.01)
            for k in range(3):
                self._feed(floored, offset=100 * k, t_seconds=k + 1)
            self.assertAlmostEqual(floored.optimizer.param_groups[0]['lr'], 0.5e-5)

    def test_lr_decay_is_validated(self):
        with tempfile.TemporaryDirectory() as d:
            for bad in (0.0, 1.0, 1.5):
                with self.assertRaises(ValueError):
                    self._build(d, lr_decay=bad)

    def test_grad_clip_default_and_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(build_trainer(d).grad_clip_value, 1.0)
            trainer = self._build(d, grad_clip_value=0)
            self.assertEqual(trainer.grad_clip_value, float('inf'))
            with unittest.mock.patch('torch.nn.utils.clip_grad_norm_',
                                     wraps=torch.nn.utils.clip_grad_norm_) as clip:
                self._feed(trainer)
            self.assertTrue(clip.called)
            self.assertTrue(all(c.args[1] == float('inf') for c in clip.call_args_list))


def feed_with_gain(trainer, gain, offset=0, t_seconds=1):
    """Time-tagged batch (see time_tagged_batch) with an explicit per-frame
    gain_mod sequence."""
    gain = np.asarray(gain, dtype=np.float32)
    x, labels = time_tagged_batch(len(gain), offset, seed=offset)
    x_val, y_val = BaseValue(value=x), BaseValue(value=labels)
    g_val = BaseValue(value=gain.reshape(-1, 1))
    t = x_val.seconds_to_t(t_seconds)
    x_val.generation_time = y_val.generation_time = g_val.generation_time = t
    trainer.inputs['input_2d_batch'].set(x_val)
    trainer.inputs['labels'].set(y_val)
    trainer.inputs['gain_mod'].set(g_val)
    trainer.check_ready(t)
    with contextlib.redirect_stdout(io.StringIO()) as out:
        trainer.trigger()
    return out.getvalue()


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestTransientRegime(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, replay_size=1000, replay_subsample=1, replay_ratio=0.5,
                    replay_batch=4, gain_mod_threshold=1.0)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def test_frames_since_reclose_counts_and_carries_over(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            np.testing.assert_array_equal(
                trainer._frames_since_reclose(np.array([1, 1, 0, 0.5, 1, 1, 1.0])), [1, 2, 0, 0, 1, 2, 3])
            np.testing.assert_array_equal(trainer._frames_since_reclose(np.array([1, 0, 1])), [4, 0, 1])

    def test_counter_advances_on_frames_that_are_filtered_out(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            feed_with_gain(trainer, [0.0] * 8)            # nothing kept, trigger returns early
            self.assertEqual(trainer._frames_at_full_gain, 0)
            feed_with_gain(trainer, [1.0] * 8, offset=100, t_seconds=2)
            self.assertEqual(trainer._frames_at_full_gain, 8)

    def test_weight_zero_drops_transient_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.0)
            feed_with_gain(trainer, [1.0] * 32)
            # frames since full gain are 1..32: only 6..32 (time tags 5..31)
            # are kept, split between training and validation.
            kept = set(trainer.replay_y[:trainer.replay_count, 0].tolist())
            kept |= set(trainer.val_targets[:, 0].cpu().tolist())
            self.assertEqual(kept, set(float(t) for t in range(5, 32)))

    def test_weight_between_zero_and_one_downweights_transient_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.25)
            feed_with_gain(trainer, [1.0] * 32)
            t = trainer.replay_y[:trainer.replay_count, 0]
            w = trainer.replay_w[:trainer.replay_count]
            torch.testing.assert_close(w[t <= 4], torch.full_like(w[t <= 4], 0.25))
            torch.testing.assert_close(w[t > 4], torch.ones_like(w[t > 4]))

    def test_weighted_loss_is_a_weighted_mean_of_per_sample_huber(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, loss_delta=1.0)
            p = torch.tensor([[0.5, 3.0], [0.0, 0.0]])
            y = torch.zeros(2, 2)
            w = torch.tensor([1.0, 3.0])
            per_sample = torch.tensor([(0.125 + 2.5) / 2, 0.0])
            expected = (w * per_sample).sum() / w.sum()
            self.assertAlmostEqual(trainer._loss(p, y, w, None).item(), expected.item(), places=6)
            self.assertAlmostEqual(trainer._loss(p, y, None, None).item(), trainer.loss_fn(p, y).item(), places=6)

    def test_settle_frames_without_gain_mod_warns_once_and_trains(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, settle_frames=5, transient_weight=0.0)
            with unittest.mock.patch.object(trainer.logger, 'warning') as mock_warning:
                for k in range(2):
                    feed_arrays(trainer, *time_tagged_batch(32, 100 * k, seed=k), t_seconds=k + 1)
                    trainer.trigger()
            self.assertEqual(logged(mock_warning).count("settle_frames needs the 'gain_mod' input"), 1)
            self.assertEqual(trainer.step_count, 2)

    def test_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for kwargs in (dict(settle_frames=-1), dict(transient_weight=1.5), dict(transient_weight=-0.1)):
                with self.assertRaises(ValueError):
                    self._build(d, **kwargs)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestFrameStackingAndNormLoss(unittest.TestCase):

    def _build(self, d, **kwargs):
        opts = dict(input_channels=2, n_frames=4, replay_size=1000, replay_subsample=1,
                    replay_ratio=0.5, replay_batch=4)
        opts.update(kwargs)
        return build_trainer(d, **opts)

    def _feed_time_frames(self, trainer, gain, first=0, t_seconds=1):
        """Frames whose pixels all equal their time index, labels tagged
        with it too (first mode, label_offset=1), and a gain_mod sequence."""
        n = len(gain)
        t = np.arange(first, first + n, dtype=np.float32)
        x = np.broadcast_to(t[:, None, None, None], (n, 2, H, W)).copy()
        labels = np.random.default_rng(first).standard_normal((n, NMODES + 1)).astype(np.float32)
        labels[:, 1] = t
        x_val, y_val = BaseValue(value=x), BaseValue(value=labels)
        g_val = BaseValue(value=np.asarray(gain, dtype=np.float32).reshape(-1, 1))
        ts = x_val.seconds_to_t(t_seconds)
        x_val.generation_time = y_val.generation_time = g_val.generation_time = ts
        trainer.inputs['input_2d_batch'].set(x_val)
        trainer.inputs['labels'].set(y_val)
        trainer.inputs['gain_mod'].set(g_val)
        trainer.check_ready(ts)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.trigger()

    def test_network_input_has_input_channels_times_n_frames(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            inner = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            self.assertEqual(inner.encoders[0].block[0].in_channels, 8)

    def test_stacks_true_previous_frames_before_filtering(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d)
            # Frames 0-1 at zero gain are dropped as samples, but frames 2-3
            # must still carry them as their history.
            self._feed_time_frames(trainer, [0.0, 0.0] + [1.0] * 30)
            self._feed_time_frames(trainer, [1.0] * 32, first=32, t_seconds=2)
            n = trainer.replay_count
            t = trainer.replay_y[:n, 0]
            stacked_times = trainer.replay_x[:n, ::2, 0, 0]      # one channel per frame
            for time, times in zip(t.tolist(), stacked_times.tolist()):
                self.assertEqual(times, [max(time - k, 0) for k in range(4)])
            self.assertIn(2.0, t.tolist() + trainer.val_targets[:, 0].cpu().tolist())
            self.assertIn(32.0, t.tolist() + trainer.val_targets[:, 0].cpu().tolist())

    def test_n_frames_is_saved_and_the_tester_takes_it_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, network_name='stack.pth')
            for k in range(10):   # first periodic save at step 10
                self._feed_time_frames(trainer, [1.0] * 32, first=32 * k, t_seconds=k + 1)
            with open(trainer.stats_filename) as f:
                self.assertEqual(json.load(f)['network']['n_frames'], 4)

            tester = Conv2dNetTester(network_filename=trainer.network_filename, target_device_idx=-1)
            self.assertEqual(tester.stacker.n_frames, 4)
            x = BaseValue(value=np.random.default_rng(0).standard_normal((8, 2, H, W)).astype(np.float32))
            y = BaseValue(value=np.random.default_rng(1).standard_normal((8, NMODES + 1)).astype(np.float32))
            x.generation_time = y.generation_time = x.seconds_to_t(1)
            tester.inputs['input_2d_batch'].set(x)
            tester.inputs['labels'].set(y)
            tester.check_ready(1)
            tester.trigger()
            self.assertEqual(tester.count, 8)

    def test_norm_loss_adds_the_per_mode_normalized_mse(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = self._build(d, loss_delta=1.0, norm_loss_weight=2.0)
            p = torch.tensor([[0.5, 3.0], [1.0, 0.0]])
            y = torch.zeros(2, 2)
            std = torch.tensor([1.0, 10.0])
            huber = torch.nn.functional.huber_loss(p, y, delta=1.0)
            normalized = ((p / std) ** 2).mean()
            self.assertAlmostEqual(trainer._loss(p, y, None, std).item(),
                                   (huber + 2.0 * normalized).item(), places=6)

    def test_reported_train_loss_stays_plain_nm(self):
        with tempfile.TemporaryDirectory() as d:
            plain = self._build(d, network_name='a.pth', replay_size=0, epoch_len=1, n_frames=1)
            weighted = self._build(d, network_name='b.pth', replay_size=0, epoch_len=1, n_frames=1,
                                   norm_loss_weight=1000.0)
            weighted.model.load_state_dict(plain.model.state_dict())
            for trainer in (plain, weighted):
                torch.manual_seed(0)
                self._feed_time_frames(trainer, [1.0] * 32)
            # Same weights, same data, same split: the reported (pre-step)
            # loss is the same nm loss, despite the huge extra term.
            self.assertAlmostEqual(plain.loss, weighted.loss, places=4)

    def test_invalid_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for kwargs in (dict(n_frames=0), dict(norm_loss_weight=-1.0)):
                with self.assertRaises(ValueError):
                    self._build(d, **kwargs)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestToTorchHelper(unittest.TestCase):

    def test_to_torch_converts_numpy_array(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        t = to_torch(arr)
        self.assertEqual(t.dtype, torch.float32)
        np.testing.assert_allclose(t.numpy(), arr)


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestSpatialHeadTrainerRoundTrip(unittest.TestCase):

    def _train(self, trainer, n_steps=10):
        rng = np.random.default_rng(0)
        for k in range(n_steps):
            x = BaseValue(value=rng.standard_normal((16, 2, H, W)).astype(np.float32))
            y = BaseValue(value=rng.standard_normal((16, NMODES + 1)).astype(np.float32))
            t = x.seconds_to_t(k + 1)
            x.generation_time = y.generation_time = t
            trainer.inputs['input_2d_batch'].set(x)
            trainer.inputs['labels'].set(y)
            trainer.check_ready(t)
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.trigger()

    def _tester(self, filename, head_type):
        return Conv2dNetTester(network_filename=filename, target_device_idx=-1)

    def test_head_type_is_saved_and_reloaded_and_resuming_checks_it(self):
        with tempfile.TemporaryDirectory() as d:
            trainer = build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                                    head_type='spatial')
            inner = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
            self.assertIsInstance(inner.regressor, SpatialRegressionHead)
            self._train(trainer)   # first periodic save at step 10
            with open(trainer.stats_filename) as f:
                self.assertEqual(json.load(f)['network']['head_type'], 'spatial')

            tester = self._tester(trainer.network_filename, 'spatial')
            self.assertIsInstance(tester.model.regressor, SpatialRegressionHead)
            saved = torch.load(trainer.network_filename, map_location='cpu')
            for k, v in tester.model.state_dict().items():
                torch.testing.assert_close(v, saved[k])

            # Resuming with the other head must stop, not silently start a
            # new model that would overwrite the checkpoint.
            with self.assertRaisesRegex(ValueError, "head_type='pooled' \\(checkpoint: 'spatial'\\)"):
                build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                              load_from_file=True)
            resumed = build_trainer(d, network_name='spatial.pth', input_channels=2, depth=3,
                                    head_type='spatial', load_from_file=True)
            inner = resumed.model.module if hasattr(resumed.model, 'module') else resumed.model
            for k, v in inner.state_dict().items():
                torch.testing.assert_close(v.cpu(), saved[k])


if __name__ == '__main__':
    unittest.main()
