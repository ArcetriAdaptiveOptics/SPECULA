import os
import json
import tempfile
import unittest

import specula
specula.init(0)  # Default target device

import numpy as np

from specula.base_value import BaseValue
from specula.data_objects.slopes import Slopes

# torch is an optional dependency (see pyproject.toml's "nn" extra): skip
# every test in this module rather than failing collection when it's absent.
try:
    import torch
    from iteras.lib.efficient_u_net import UNetRegressor
    from iteras.lib import cnn_checkpoint
    from iteras.lib.cnn_checkpoint import build_network
    from specula.processing_objects.conv2d_net_rec import Conv2dNetRec
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NMODES = 5
CHANNELS = 4
DEPTH = 1
MASK_SIDE = 8
N_VALID = 6  # number of valid subapertures


def save_checkpoint(net_path, stats_path=None, nmodes=NMODES, channels=CHANNELS, depth=DEPTH,
                    input_channels=2, n_frames=1, head_type='pooled', head_grid=32,
                    channel_product=False, **stats):
    """A checkpoint as Conv2dNetTrainer writes it: weights plus a stats file
    recording the network, zero-mean / unit-std normalization, and any extra
    stats given. stats_path is where that file lands (derived from net_path)."""
    network = dict(nmodes=nmodes, input_channels=input_channels, n_frames=n_frames,
                   channels=channels, depth=depth, conv_block_type=0, head_type=head_type,
                   head_grid=head_grid, dropout=0.0, channel_product=channel_product)
    model = build_network(network)
    cnn_checkpoint.save_checkpoint(model, net_path, dict(
        {'network': network, 'meanp': 0.0, 'stdp': 1.0,
          'meanmodes': [0.0] * nmodes, 'stdmodes': [1.0] * nmodes}, **stats))
    return model


def build_rec(tmp_dir, network_name='net.pth', nmodes=NMODES, channels=CHANNELS, depth=DEPTH,
              input_channels=2, channel_product=False, target_device_idx=-1, **kwargs):
    net_path = os.path.join(tmp_dir, network_name)
    stats_path = net_path.replace('.pth', '_stats.json')
    save_checkpoint(net_path, stats_path, nmodes=nmodes, channels=channels, depth=depth,
                    input_channels=input_channels, channel_product=channel_product)
    rec = Conv2dNetRec(network_filename=net_path, target_device_idx=target_device_idx, **kwargs)
    return rec, net_path, stats_path


def build_slopes(target_device_idx, seed=0):
    """A Slopes object with a valid single_mask/display_map so get2d()
    returns a real (2, MASK_SIDE, MASK_SIDE) map, as a PyrSlopec/ShSlopec
    output would."""
    rng = np.random.default_rng(seed)
    slopes = Slopes(length=2 * N_VALID, interleave=False, target_device_idx=target_device_idx)
    slopes.slopes[:] = slopes.xp.asarray(rng.standard_normal(2 * N_VALID))
    slopes.single_mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    slopes.display_map = np.arange(N_VALID)
    return slopes


def run_once(rec, slopes, target_device_idx):
    slopes.generation_time = rec.seconds_to_t(1)
    rec.inputs['in_slopes'].set(slopes)
    rec.setup()
    t = rec.seconds_to_t(1)
    rec.check_ready(t)
    rec.trigger()
    rec.post_trigger()
    from specula import cpuArray
    return cpuArray(rec.outputs['out_modes'].value).copy()


class TestConv2dNetRecConstruction(unittest.TestCase):

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_missing_model_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                Conv2dNetRec(network_filename=os.path.join(d, 'nope.pth'), target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_missing_stats_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            model = UNetRegressor(
                input_channels=2, output_size=NMODES, base_channels=CHANNELS,
                dropout_level=0.0, conv_block_type=0, depth=DEPTH,
            )
            torch.save(model.state_dict(), net_path)
            with self.assertRaises(FileNotFoundError):
                Conv2dNetRec(network_filename=net_path, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_successful_construction(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            self.assertEqual(rec.model.encoders[0].block[0].in_channels, 2)
            self.assertEqual(set(rec.outputs.keys()), {'out_modes'})
            self.assertTrue(rec.model.training is False)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_sanity_check_passes_with_baseline_wired(self):
        # Regression test: sanity_check() (run by Simul/LoopControl before
        # any real simulation) failed here because Conv2dNetRec inherits
        # BaseModalrec.input_names(), which doesn't declare 'baseline', so
        # the framework rejected it as an undeclared input the moment a
        # real YAML actually wired one up -- something no direct-Python
        # unit test here caught until this one.
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            baseline = BaseValue(value=np.zeros(NMODES, dtype=np.float32))
            rec.inputs['baseline'].set(baseline)
            rec.sanity_check()  # must not raise

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_n_frames_stacks_the_previous_slope_maps(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            model = save_checkpoint(net_path, stats_path, input_channels=2, n_frames=2)

            rec = Conv2dNetRec(network_filename=net_path, target_device_idx=-1)
            first, second = build_slopes(-1, seed=0), build_slopes(-1, seed=1)
            out1 = run_once(rec, first, -1)
            out2 = run_once(rec, second, -1)

            m1 = np.asarray(first.get2d(), dtype=np.float32)
            m2 = np.asarray(second.get2d(), dtype=np.float32)
            model.eval()
            with torch.no_grad():
                # meanp/stdp = 0/1 and meanmodes/stdmodes = 0/1 in the stats:
                # the output is the raw network output on the stacked maps.
                exp1 = model(torch.from_numpy(np.concatenate([m1, m1])[None])).numpy()[0]
                exp2 = model(torch.from_numpy(np.concatenate([m2, m1])[None])).numpy()[0]
            np.testing.assert_allclose(out1, exp1, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(out2, exp2, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_spatial_head_checkpoint_is_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            model = save_checkpoint(net_path, stats_path, depth=2, head_type='spatial')

            rec = Conv2dNetRec(network_filename=net_path, target_device_idx=-1)
            slopes = build_slopes(-1)
            out = run_once(rec, slopes, -1)
            x = torch.from_numpy(np.asarray(slopes.get2d(), dtype=np.float32))[None]
            with torch.no_grad():
                expected = model.eval()(x).numpy()[0]
            np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-5)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_calibrate_gain_scales_the_prediction_per_mode(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            stats_path = net_path.replace('.pth', '_stats.json')
            save_checkpoint(net_path, stats_path, meanmodes=[1.0] * NMODES,
                            # last two: not correctable / no shrink
                            mode_gain=[1.0, 0.5, 0.25, 0.1, 1.0])

            rec = Conv2dNetRec(network_filename=net_path, calibrate_gain=True, target_device_idx=-1)
            rec.meanmodes = np.ones(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.full((1, NMODES), 2.0)   # -> 3.0 before calibration
            out = run_once(rec, build_slopes(-1), -1)
            # the deviation from meanmodes (2.0) is multiplied by 1/gain, capped at 3
            np.testing.assert_allclose(out, 1.0 + np.array([2.0, 4.0, 6.0, 2.0, 2.0]), atol=1e-5)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_calibrate_gain_without_a_measurement_raises(self):
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            save_checkpoint(net_path, net_path.replace('.pth', '_stats.json'))
            with self.assertRaisesRegex(ValueError, 'no per-mode gain'):
                Conv2dNetRec(network_filename=net_path, calibrate_gain=True, target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_sanity_check_passes_without_baseline_wired(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.sanity_check()  # must not raise: baseline is optional


@unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
class TestConv2dNetRecTrigger(unittest.TestCase):

    def test_trigger_produces_modes_of_correct_shape(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            slopes = build_slopes(target_device_idx=-1)
            out = run_once(rec, slopes, target_device_idx=-1)
            self.assertEqual(out.shape, (NMODES,))
            self.assertTrue(np.all(np.isfinite(out)))

    def test_without_baseline_output_is_bare_denormalized_network_output(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.meanmodes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            rec.stdmodes = np.array([2.0, 1.0, 0.5, 4.0, 1.5])
            rec.model = lambda x: torch.zeros(1, NMODES)

            slopes = build_slopes(target_device_idx=-1)
            out = run_once(rec, slopes, target_device_idx=-1)

            np.testing.assert_allclose(out, rec.meanmodes, atol=1e-5)

    def test_baseline_is_added_back_to_network_output(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.meanmodes = np.zeros(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.zeros(1, NMODES)  # predicted residual == 0

            baseline_vals = np.array([10.0, -1.0, 0.5, 3.0, -2.0])
            baseline = BaseValue(value=baseline_vals.astype(np.float32))
            slopes = build_slopes(target_device_idx=-1)
            baseline.generation_time = slopes.generation_time = 1
            rec.inputs['baseline'].set(baseline)

            out = run_once(rec, slopes, target_device_idx=-1)
            np.testing.assert_allclose(out, baseline_vals, atol=1e-5)

    def test_baseline_offset_selects_correct_columns(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d, baseline_offset=2)
            rec.meanmodes = np.zeros(NMODES)
            rec.stdmodes = np.ones(NMODES)
            rec.model = lambda x: torch.zeros(1, NMODES)

            baseline_full = np.arange(10.0)  # [0..9]
            baseline = BaseValue(value=baseline_full.astype(np.float32))
            slopes = build_slopes(target_device_idx=-1)
            baseline.generation_time = slopes.generation_time = 1
            rec.inputs['baseline'].set(baseline)

            out = run_once(rec, slopes, target_device_idx=-1)
            np.testing.assert_allclose(out, baseline_full[2:2 + NMODES], atol=1e-5)


class TestConv2dNetRecInput(unittest.TestCase):

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_channel_product_network_gets_the_training_input(self):
        # With channel_product the trainer feeds x/y slope maps as their
        # per-pixel product (network_input): so must the reconstructor.
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d, input_channels=1, channel_product=True)
            seen = []
            rec.model = lambda x: (seen.append(x.clone()), torch.zeros(1, NMODES))[1]
            slopes = build_slopes(target_device_idx=-1)
            maps = np.asarray(slopes.get2d())
            run_once(rec, slopes, target_device_idx=-1)
            self.assertEqual(tuple(seen[0].shape), (1, 1, MASK_SIDE, MASK_SIDE))
            np.testing.assert_allclose(seen[0].numpy()[0, 0], maps[1] * maps[0], atol=1e-6)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_input_channels_must_match_the_slope_maps(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d, input_channels=1)
            with self.assertRaisesRegex(ValueError, 'channel'):
                run_once(rec, build_slopes(target_device_idx=-1), target_device_idx=-1)

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    def test_slopes_list_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rec, _, _ = build_rec(d)
            rec.inputs['in_slopes_list'].set([build_slopes(target_device_idx=-1)])
            with self.assertRaisesRegex(ValueError, 'in_slopes_list'):
                rec.setup()


class TestConv2dNetRecDevice(unittest.TestCase):

    @unittest.skipIf(not TORCH_AVAILABLE, "torch is not installed")
    @unittest.skipIf(specula.cp is None or not TORCH_AVAILABLE or not torch.cuda.is_available(),
                     "no GPU")
    def test_gpu_output_matches_cpu(self):
        # Same checkpoint and slopes: the network on the GPU (cupy <-> torch
        # through DLPack) must give the CPU result, with gain calibration on.
        with tempfile.TemporaryDirectory() as d:
            net_path = os.path.join(d, 'net.pth')
            # 16 channels: with CHANNELS=4 the bottleneck attention has 4 // 8 = 0
            # channels, an empty conv that torch hands to triton on CUDA.
            save_checkpoint(net_path, channels=16, mode_gain=[0.5, 0.8, 1.0, 0.9, 0.6],
                            meanmodes=[1.0, -2.0, 0.5, 0.0, 3.0],
                            stdmodes=[2.0, 1.0, 0.5, 4.0, 1.5])
            outputs = []
            for device in (-1, 0):
                rec = Conv2dNetRec(network_filename=net_path, calibrate_gain=True,
                                   target_device_idx=device)
                outputs.append(run_once(rec, build_slopes(target_device_idx=device), device))
            self.assertEqual(str(rec.torch_device), 'cuda:0')
            # cuDNN uses TF32 for float32 convolutions by default (~1e-3 relative)
            np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
